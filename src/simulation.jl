# TOML-driven simulation factory and time loop.
#
# Entry point: Carina.run(input_file)
#
# create_simulation(dict)  reads a parsed TOML dict and builds a
#   SingleDomainSimulation ready to evolve.
#
# evolve!(sim)  runs the full time loop and writes Exodus output.

import Adapt
import FiniteElementContainers as FEC
import ConstitutiveModels as CM
import TOML

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

# Human-readable label for the device log line.
_backend_label(b::KA.Backend) =
    b isa KA.CPU ? "CPU" : "GPU [" * string(nameof(typeof(b))) * "]"

function _log_block_material(block_name::AbstractString, cm_name::AbstractString,
                              density::Float64, props_inputs::Dict)
    parts = String["density = " * Printf.@sprintf("%.3e", density)]
    for key in sort!(collect(keys(props_inputs)))
        push!(parts, key * " = " * Printf.@sprintf("%.3e", Float64(props_inputs[key])))
    end
    _carina_log(0, :setup, "Block \"$block_name\": $cm_name : " * join(parts, ", "))
end

# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

"""
    Carina.run(input_file; backend=KA.CPU())

Load `input_file` (TOML), create a simulation, run it, and close the output file.
`backend` is the `KernelAbstractions.Backend` to run on (default `KA.CPU()`).
Command-line device selection is handled by the `bin/carina` launcher.
"""
function run(input_file::String; backend::KA.Backend=KA.CPU())
    open_log_file(input_file)
    try
        t_start = time()
        _carina_log(0, :carina, "BEGIN SIMULATION")
        _carina_log(0, :setup,  "Reading from $input_file")

        dict = TOML.parsefile(input_file)
        sim_type = get(dict, "type", "single")::String
        if sim_type == "single"
            sim = create_simulation(dict, dirname(abspath(input_file));
                                    backend=backend)
            evolve!(sim)
            FEC.close(sim.post_processor)
        else
            error("Simulation type \"$sim_type\" not yet supported. Only \"single\" is implemented.")
        end

        _carina_log(0, :done, "Simulation complete")
        _carina_log(0, :time, "Total wall time = $(format_time(time() - t_start))")
        _carina_log(0, :carina, "END SIMULATION")
        return sim
    finally
        close_log_file()
    end
end

# ---------------------------------------------------------------------------
# create_simulation
# ---------------------------------------------------------------------------

"""
    create_simulation(dict, basedir=""; backend=KA.CPU()) -> SingleDomainSimulation

Parse a TOML dict (already loaded) and return a fully initialised simulation.
`basedir` is used to resolve relative file paths inside the input.
`backend` is the `KernelAbstractions.Backend` to run on.
"""
function create_simulation(dict::Dict{String,Any}, basedir::String="";
                            backend::KA.Backend=KA.CPU())
    _validate_keys(dict, _TOPLEVEL_KEYS, "top-level input")
    _carina_log(0, :device, _backend_label(backend))

    input_mesh  = _resolve(dict, "input_mesh_file",  basedir)
    output_file = _resolve(dict, "output_mesh_file", basedir)
    _carina_log(0, :setup, "Input:  $input_mesh")
    _carina_log(0, :setup, "Output: $output_file")

    t_setup = time()

    block_name, cm, density, props_inputs = _parse_material_section(dict)
    props   = create_solid_mechanics_properties(cm, props_inputs)
    physics = SolidMechanics(cm, density)
    cm_name = replace(string(typeof(cm)), r"^.*\." => "")  # strip module prefix
    _log_block_material(block_name, cm_name, density, props_inputs)

    mesh    = @carina_phase "Reading mesh" FEC.UnstructuredMesh(input_mesh)
    n_nodes = size(mesh.nodal_coords, 2)
    n_elems = sum(size(mesh.element_conns[k], 2) for k in keys(mesh.element_conns))
    _carina_logf(0, :setup, "Mesh:    %d nodes, %d elements", n_nodes, n_elems)

    q_type, q_order = _parse_quadrature(dict)
    V       = @carina_timed "Function space" FEC.FunctionSpace(
                                mesh, FEC.H1Field, FEC.Lagrange, q_type;
                                q_degree=q_order)
    # Opt the explicit central-difference path into FEC's matrix-free assembler
    # mode.  Skips ~7 GB of sparse-matrix preallocation (sparsity pattern + the
    # mass/stiffness value buffers) on a 530 k-DOF mesh, which is unused on a
    # central-difference run.  Implicit paths (Newmark/QuasiStatic) still need
    # the assembled stiffness for the Jacobi preconditioner so they stay in
    # the matrix-bearing default; they still benefit from the FEC change that
    # dropped the dead damping/hessian buffers.
    matrix_free = _integrator_is_matrix_free(dict)
    asm_cpu = @carina_phase "Building assembler" FEC.SparseMatrixAssembler(
                                         FEC.VectorFunction(V, "displ");
                                         use_condensed=false,
                                         matrix_free=matrix_free)

    # Dirichlet BCs must be enforced by DOF elimination, never by penalty.
    # Penalty enforcement (use_condensed=true) pollutes the spectrum with
    # artificial eigenvalues ~10^6 × tr(K)/n, degrading CG convergence by
    # orders of magnitude.  This assertion guards against regression.
    @assert !FEC._is_condensed(asm_cpu.dof) "Carina requires use_condensed=false (DOF elimination, not penalty)"

    dbcs = _parse_dirichlet_bcs(dict)
    nbcs, point_load_entries = _parse_neumann_bcs(dict)
    bfs  = _parse_body_forces(dict)

    t0, tf, dt, times = _parse_times(dict)
    raw_oi = get(dict, "output_interval", nothing)
    output_interval = raw_oi === nothing ? dt : Float64(raw_oi)
    num_stops = round(Int, (tf - t0) / output_interval) + 1
    controller = TimeController(t0, tf, output_interval, t0, t0, num_stops, 0)
    if output_interval ≈ dt
        _carina_logf(0, :setup, "Time:    [%.2e, %.2e], Δt = %.2e, %d steps",
                     t0, tf, dt, num_stops - 1)
    else
        _carina_logf(0, :setup, "Time:    [%.2e, %.2e], Δt = %.2e, output every %.2e (%d stops)",
                     t0, tf, dt, output_interval, num_stops - 1)
    end

    p_cpu = @carina_phase "Building parameters" FEC.create_parameters(
        mesh, asm_cpu, physics, props;
        dirichlet_bcs = dbcs,
        neumann_bcs   = nbcs,
        sources       = bfs,
        times         = times,
    )
    n_dofs = length(asm_cpu.dof)
    n_free = length(asm_cpu.dof.unknown_dofs)
    _carina_logf(0, :setup, "DOFs:    %d total, %d free, %d constrained",
                 n_dofs, n_free, n_dofs - n_free)

    # Build point loads (Neumann BCs on node sets) — needs finalized dof
    _init_point_loads!(_build_point_loads(point_load_entries, mesh, asm_cpu.dof),
                       p_cpu.coords.data)

    output_spec = _parse_output_spec(dict)
    is_dynamic  = _is_dynamic_integrator(dict)

    # Build VectorFunction list for PostProcessor (all nodal vars in one call).
    nodal_vars = _build_nodal_vars(V, output_spec, is_dynamic)
    rec_names  = _recovered_nodal_var_names(physics, output_spec)
    pp = @carina_timed "Post-processor + output DB" FEC.PostProcessor(
                           mesh, output_file, nodal_vars...;
                           extra_nodal_names = rec_names)

    # Register element variable names (stress, F, IVs at QPs) before first write.
    el_names = _element_var_names(asm_cpu, physics, output_spec)
    if !isempty(el_names)
        Exodus.write_names(pp.field_output_db, Exodus.ElementVariable, el_names)
    end

    if backend isa KA.CPU
        asm = asm_cpu
        p   = p_cpu
    else
        _carina_log(0, :setup, "Transferring to GPU...")
        asm = FEC.to_backend(backend, asm_cpu)
        p   = FEC.to_backend(backend, p_cpu)
    end

    @carina_timed "Assembly cache" _init_assembly_cache!(asm_cpu, cm isa CM.LinearElastic)
    _cpu_asm_ref[]    = asm_cpu
    _cpu_params_ref[] = p_cpu
    _backend_ref[]    = backend
    integrator = @carina_phase "Building integrator and solver" _parse_integrator(
                               dict, asm, asm_cpu, p_cpu, controller, backend)
    _carina_logf(0, :setup, "Solver:  %s", _solver_description(integrator))

    # Evaluate Dirichlet BC values at t=0 so _update_for_assembly! can set
    # constrained DOFs correctly before IC application and initial acceleration.
    # Only CPU parameters needed — GPU field is synced via _update_for_assembly!.
    FEC.update_bc_values!(p_cpu, asm_cpu)

    t0 = controller.initial_time
    @carina_timed "Initial conditions" begin
        _apply_initial_displacement_ics!(integrator, mesh, asm_cpu, p, p_cpu,
                                          _parse_displacement_ics(dict), backend, t0)
        _apply_initial_velocity_ics!(integrator, mesh, asm_cpu, p_cpu,
                                      _parse_velocity_ics(dict), t0)
        # Traveling-wave entries come after the explicit u/v ICs so that, on
        # the union of touched DOFs, the symbolic v₀ = −c·∂u₀/∂s overrides any
        # zero-velocity default the explicit lists may have left in place.
        _apply_initial_traveling_wave_ics!(integrator, mesh, asm_cpu, p, p_cpu,
                                            _parse_traveling_wave_ics(dict), backend, t0)
        _compute_initial_acceleration!(integrator, asm_cpu, p_cpu)
        # Propagate g(t_0), g'(t_0), g''(t_0) into the BC slots of the
        # full-DOF integrator state so the t=0 output and any subsequent
        # read of integrator.V[BC] / integrator.A[BC] see the prescribed
        # values rather than the zero-init buffer.  No-op for the
        # quasi-static integrator (no V/A).  Subsequent steps re-do this
        # inside predict! using bc_cache values at t_{n+1}.
        _propagate_dirichlet_bcs_to_state!(integrator, p)
    end
    @carina_timed "Initial equilibrium" _compute_initial_equilibrium!(integrator, p)

    # Build recovery data for L2 projection (CPU-only)
    recovery_data = @carina_timed "Recovery data" _build_recovery_data(
                                  output_spec.recovery, asm_cpu, p_cpu)

    _carina_log(0, :setup, "Setup complete ($(format_time(time() - t_setup)))")

    n_steps = controller.num_stops - 1
    sim = SingleDomainSimulation(p, p_cpu, asm_cpu, integrator, pp,
                                  controller, backend, output_spec, recovery_data)

    # Write initial state (step 1, t=0).
    @carina_phase "Writing initial state" write_output!(sim, 1)
    pct_digits = max(0, Int(ceil(log10(n_steps))) - 2)
    pct_fmt_init = "[0/%d, %." * string(pct_digits) * "f%%] : Time = %.2e"
    _carina_logf(0, :stop, pct_fmt_init, n_steps, 0.0, controller.initial_time)
    _carina_log(0, :output, output_file)

    return sim
end

# ---------------------------------------------------------------------------
# evolve!
# ---------------------------------------------------------------------------

"""
    evolve!(sim::SingleDomainSimulation)

Run the full time loop, writing Exodus output at every output stop.
The controller's `control_step` equals the output interval; the integrator
subcycles within each interval using its own (possibly adaptive) time step.
"""
function evolve!(sim::SingleDomainSimulation)
    (; params, post_processor, controller, integrator) = sim
    n_steps = controller.num_stops - 1
    is_explicit = integrator isa CentralDifferenceIntegrator

    # Dynamic percentage format: 0 decimals for ≤100 steps, 1 for ≤1000, etc.
    pct_digits = max(0, Int(ceil(log10(n_steps))) - 2)
    pct_base = "[%d/%d, %." * string(pct_digits) * "f%%] : Time = %.2e : |U|_max = %.2e"
    output_basename = basename(post_processor.field_output_db.file_name)

    output_step = 2  # step 1 is the initial frame written in create_simulation
    t_batch = time()  # wall time since last output

    for _ in 1:n_steps
        _advance_controller!(controller)
        t_prev = controller.prev_time
        t_stop = controller.time

        # Reset FEC clock to start of this control interval
        params.times.time_current = t_prev

        _subcycle!(sim, t_stop, is_explicit)

        t   = controller.time
        pct = 100.0 * controller.stop / n_steps

        write_output!(sim, output_step)
        output_step += 1
        u_max = maximum(abs, params.field.data)
        wall_elapsed = time() - t_batch
        if wall_elapsed > 0.01
            _carina_logf(0, :stop, pct_base * " : wall = %s",
                controller.stop, n_steps, pct, t, u_max, format_time(wall_elapsed))
        else
            _carina_logf(0, :stop, pct_base,
                controller.stop, n_steps, pct, t, u_max)
        end
        _carina_log(0, :output, output_basename)
        t_batch = time()
    end
end

function _advance_controller!(c::TimeController)
    c.prev_time = c.time
    c.stop     += 1
    c.time      = c.initial_time + c.stop * c.control_step
end

function _subcycle!(sim, target::Float64, is_explicit::Bool=false)
    (; params, integrator) = sim
    last_log_wall = -Inf  # for explicit: throttle to once per second of wall time

    while true
        t  = FEC.current_time(params.times)
        dt = _adjusted_step(t, integrator.time_step, target)
        params.times.Δt = dt

        if is_explicit
            wall = time()
            if wall - last_log_wall >= 1.0
                _carina_logf(4, :advance, "[%.2e, %.2e] : Δt = %.2e", t, t + dt, dt)
                last_log_wall = wall
            end
        else
            _carina_logf(4, :advance, "[%.2e, %.2e] : Δt = %.2e", t, t + dt, dt)
        end

        _pre_step_hook!(integrator, sim)
        _advance_one_step!(sim)

        isapprox(FEC.current_time(params.times), target;
                 rtol=1e-6, atol=1e-12) && break
    end
end

function _adjusted_step(t::Float64, dt::Float64, t_stop::Float64)::Float64
    gap = t_stop - t
    gap <= 0.0 && return 0.0
    t_next = t + dt
    return t_stop - t_next <= 0.5 * dt ? gap : dt
end

function _advance_one_step!(sim)
    (; params, integrator) = sim
    _save_state!(integrator, params)

    while true
        integrator.failed[] = false

        # Defense-in-depth: evaluate! catches math errors from constitutive
        # models and returns false (→ flag path).  This outer catch handles
        # the same errors if they escape from any other point in evolve!.
        try
            FEC.evolve!(integrator, params)
        catch e
            e isa _MATH_ERRORS || rethrow()
            _carina_logf(4, :solve, "Caught %s during evolve! — treating as step failure",
                         typeof(e))
            integrator.failed[] = true
        end

        if !integrator.failed[]
            _increase_step!(integrator, params)
            break
        end

        # Step failed: restore state, reduce step, undo time advance, retry.
        # Use the actual dt (params.times.Δt) for reduction, not ig.time_step,
        # which may have grown to max_dt via successful-step increases.
        actual_dt = params.times.Δt
        _restore_state!(integrator, params)
        integrator.time_step = actual_dt   # sync before halving
        _decrease_step!(integrator, params)
        params.times.time_current -= actual_dt
        params.times.Δt = integrator.time_step
    end
end
