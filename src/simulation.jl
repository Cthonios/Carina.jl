# YAML-driven simulation factory and time loop.
#
# Entry point: Carina.run(yaml_file)
#
# create_simulation(dict)  reads a parsed YAML dict and builds a
#   SingleDomainSimulation ready to evolve.
#
# evolve!(sim)  runs the full time loop and writes Exodus output.

import Adapt
import FiniteElementContainers as FEC
import ConstitutiveModels as CM
import YAML

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

"""
    Carina.best_device() -> String

Return the best available compute device as a string: `"rocm"` if a
functional AMD GPU is found, `"cuda"` if a functional NVIDIA GPU is found,
or `"cpu"` otherwise.
"""
function best_device()
    try AMDGPU.functional() && return "rocm" catch end
    try CUDA.functional()   && return "cuda"  catch end
    return "cpu"
end

# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

"""
    Carina.run(yaml_file; device=nothing)

Load `yaml_file`, create a simulation, run it, and close the output file.
`device` (optional) overrides the `device:` key in the YAML.  Accepted values:
`"cpu"`, `"rocm"`, `"cuda"`.
"""
function run(yaml_file::String; device::Union{String,Nothing}=nothing)
    t_start = time()
    _carina_log(0, :carina, "BEGIN SIMULATION")
    _carina_log(0, :setup,  "Reading from $yaml_file")

    dict = YAML.load_file(yaml_file; dicttype=Dict{String,Any})
    sim_type = lowercase(get(dict, "type", "single"))
    if sim_type == "single"
        sim = create_simulation(dict, dirname(abspath(yaml_file));
                                device_override=device)
        Base.invokelatest(evolve!, sim)
        FEC.close(sim.post_processor)
    else
        error("Simulation type \"$sim_type\" not yet supported. Only \"single\" is implemented.")
    end

    _carina_log(0, :done, "Simulation complete")
    _carina_log(0, :time, "Total wall time = $(format_time(time() - t_start))")
    _carina_log(0, :carina, "END SIMULATION")
    return sim
end

# ---------------------------------------------------------------------------
# create_simulation
# ---------------------------------------------------------------------------

"""
    create_simulation(dict, basedir=""; device_override=nothing) -> SingleDomainSimulation

Parse a YAML dict (already loaded) and return a fully initialised simulation.
`basedir` is used to resolve relative file paths inside the YAML.
`device_override` (a string) takes priority over the `device:` YAML key.
"""
function create_simulation(dict::Dict{String,Any}, basedir::String="";
                            device_override::Union{String,Nothing}=nothing)
    _validate_keys(dict, _TOPLEVEL_KEYS, "top-level input")
    device_str = device_override !== nothing ? lowercase(device_override) :
                 lowercase(get(dict, "device", "cpu"))
    device = if device_str == "rocm"
        :rocm
    elseif device_str == "cuda"
        :cuda
    else
        :cpu
    end
    if device == :rocm
        AMDGPU.functional() ||
            error("device: rocm requested but no functional AMD GPU found.")
    elseif device == :cuda
        CUDA.functional() ||
            error("device: cuda requested but no functional NVIDIA GPU found.")
    end
    _carina_log(0, :device, device == :rocm ? "ROCm GPU" :
                             device == :cuda ? "CUDA GPU" : "CPU")

    input_mesh  = _resolve(dict, "input mesh file",  basedir)
    output_file = _resolve(dict, "output mesh file", basedir)
    _carina_log(0, :setup, "Input:  $input_mesh")
    _carina_log(0, :setup, "Output: $output_file")

    t_setup = time()

    cm, density, props_inputs = _parse_material_section(dict)
    props   = create_solid_mechanics_properties(cm, props_inputs)
    physics = SolidMechanics(cm, density)
    cm_name = replace(string(typeof(cm)), r"^.*\." => "")  # strip module prefix
    _carina_logf(0, :setup, "Material: %s (ρ = %.1f kg/m³)", cm_name, density)

    mesh    = FEC.UnstructuredMesh(input_mesh)
    n_nodes = size(mesh.nodal_coords, 2)
    n_elems = sum(size(mesh.element_conns[k], 2) for k in keys(mesh.element_conns))
    _carina_logf(0, :setup, "Mesh:    %d nodes, %d elements", n_nodes, n_elems)

    q_type, q_order = _parse_quadrature(dict)
    V       = FEC.FunctionSpace(mesh, FEC.H1Field, FEC.Lagrange;
                                q_degree=q_order, q_type=q_type)
    asm_cpu = FEC.SparseMatrixAssembler(FEC.VectorFunction(V, :displ); use_condensed=true)

    dbcs = _parse_dirichlet_bcs(dict)
    nbcs = _parse_neumann_bcs(dict)
    bfs  = _parse_body_forces(dict)

    t0, tf, dt, times = _parse_times(dict)
    raw_oi = get(dict, "output interval", nothing)
    output_interval = raw_oi === nothing ? dt : Float64(raw_oi)
    num_stops = round(Int, (tf - t0) / output_interval) + 1
    controller = TimeController(t0, tf, output_interval, t0, t0, num_stops, 0)
    _carina_logf(0, :setup, "Time:    [%.2e, %.2e], Δt = %.2e, %d steps",
                 t0, tf, dt, num_stops - 1)

    p_cpu = FEC.create_parameters(
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

    output_spec = _parse_output_spec(dict)
    is_dynamic  = _is_dynamic_integrator(dict)

    # Build VectorFunction list for PostProcessor (all nodal vars in one call).
    nodal_vars = _build_nodal_vars(V, output_spec, is_dynamic)
    rec_names  = _recovered_nodal_var_names(physics, output_spec)
    pp = FEC.PostProcessor(mesh, output_file, nodal_vars...;
                           extra_nodal_names = rec_names)

    # Register element variable names (stress, F, IVs at QPs) before first write.
    el_names = _element_var_names(asm_cpu, physics, output_spec)
    if !isempty(el_names)
        Exodus.write_names(pp.field_output_db, Exodus.ElementVariable, el_names)
    end

    if device == :rocm
        _carina_log(0, :setup, "Transferring to GPU (ROCm)...")
        asm = FEC.rocm(asm_cpu)
        p   = FEC.rocm(p_cpu)
    elseif device == :cuda
        _carina_log(0, :setup, "Transferring to GPU (CUDA)...")
        asm = FEC.cuda(asm_cpu)
        p   = FEC.cuda(p_cpu)
    else
        asm = asm_cpu
        p   = p_cpu
    end

    _carina_log(0, :setup, "Building integrator and solver...")
    integrator = _parse_integrator(dict, asm, asm_cpu, p_cpu, controller, device)
    _carina_logf(0, :setup, "Solver:  %s", _solver_description(integrator))

    # Evaluate Dirichlet BC values at t=0 so _update_for_assembly! can set
    # constrained DOFs correctly before IC application and initial acceleration.
    # Only CPU parameters needed — GPU field is synced via _update_for_assembly!.
    Base.invokelatest(FEC.update_bc_values!, p_cpu, asm_cpu)

    t0 = controller.initial_time
    _apply_initial_displacement_ics!(integrator, mesh, asm_cpu, p, p_cpu,
                                      _parse_displacement_ics(dict), device, t0)
    _apply_initial_velocity_ics!(integrator, mesh, asm_cpu, p_cpu,
                                  _parse_velocity_ics(dict), t0)
    _compute_initial_acceleration!(integrator, asm_cpu, p_cpu)
    _compute_initial_equilibrium!(integrator, p)

    # Build recovery data for L2 projection (CPU-only)
    recovery_data = _build_recovery_data(output_spec.recovery, asm_cpu, p_cpu)

    _carina_log(0, :setup, "Setup complete ($(format_time(time() - t_setup)))")

    n_steps = controller.num_stops - 1
    sim = SingleDomainSimulation(p, p_cpu, asm_cpu, integrator, pp,
                                  controller, device, output_spec, recovery_data)

    # Write initial state (step 1, t=0).
    write_output!(sim, 1)
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
    (; params, post_processor, controller, device, integrator) = sim
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

        if !is_explicit
            _carina_logf(4, :advance, "[%.2e, %.2e] : Δt = %.2e",
                t_prev, t_stop, controller.control_step)
        end

        # Reset FEC clock to start of this control interval
        params.times.time_current = t_prev

        _subcycle!(sim, t_stop)

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

function _subcycle!(sim, target::Float64)
    (; params, integrator) = sim

    while true
        t  = FEC.current_time(params.times)
        dt = _adjusted_step(t, integrator.time_step, target)
        params.times.Δt = dt

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
    (; params, integrator, device) = sim
    _save_state!(integrator, params)

    while true
        integrator.failed[] = false

        Base.invokelatest(FEC.evolve!, integrator, params)

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
