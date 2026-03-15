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
# On-demand GPU backend loading (avoid paying init cost on CPU-only runs)
# ---------------------------------------------------------------------------

const _AMDGPU_ID = Base.PkgId(
    Base.UUID("21141c5a-9bdb-4563-92ae-f87d6854732e"), "AMDGPU"
)
const _amdgpu_loaded = Ref(false)

function _require_amdgpu!()
    if !_amdgpu_loaded[]
        Base.require(_AMDGPU_ID)
        _amdgpu_loaded[] = true
    end
end

const _CUDA_ID = Base.PkgId(
    Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA"
)
const _cuda_loaded = Ref(false)

function _require_cuda!()
    if !_cuda_loaded[]
        Base.require(_CUDA_ID)
        _cuda_loaded[] = true
    end
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
        # invokelatest ensures evolve! and its entire call tree are compiled at
        # the current world age, where GPU backend methods are visible.
        # Without this, run() is compiled before the GPU package is loaded,
        # so direct calls see only Base fallbacks (world age error).
        if sim.device != :cpu
            Base.invokelatest(evolve!, sim)
        else
            evolve!(sim)
        end
        FEC.close(sim.post_processor)
    else
        error("Simulation type \"$sim_type\" not yet supported. Only \"single\" is implemented.")
    end

    _carina_log(0, :done, "Simulation complete")
    _carina_logf(0, :time, "Total wall time = %.1fs", time() - t_start)
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
    device_str = device_override !== nothing ? lowercase(device_override) :
                 lowercase(get(dict, "device", "cpu"))
    device = if device_str == "rocm"
        :rocm
    elseif device_str == "cuda"
        :cuda
    else
        :cpu
    end
    device == :rocm && _require_amdgpu!()
    device == :cuda && _require_cuda!()
    _carina_log(0, :device, device == :rocm ? "ROCm GPU" :
                             device == :cuda ? "CUDA GPU" : "CPU")

    input_mesh  = _resolve(dict, "input mesh file",  basedir)
    output_file = _resolve(dict, "output mesh file", basedir)
    output_interval = Int(get(dict, "output interval", 1))
    _carina_log(0, :setup, "Mesh:   $input_mesh")
    _carina_log(0, :setup, "Output: $output_file")

    cm, density, props_inputs = _parse_material_section(dict)
    props   = create_solid_mechanics_properties(cm, props_inputs)
    physics = SolidMechanics(cm, density)

    mesh    = FEC.UnstructuredMesh(input_mesh)
    V       = FEC.FunctionSpace(mesh, FEC.H1Field, FEC.Lagrange)
    asm_cpu = FEC.SparseMatrixAssembler(FEC.VectorFunction(V, :displ); use_condensed=true)

    dbcs = _parse_dirichlet_bcs(dict)
    nbcs = _parse_neumann_bcs(dict)

    controller, times = _parse_times(dict)

    p_cpu = FEC.create_parameters(
        mesh, asm_cpu, physics, props;
        dirichlet_bcs = dbcs,
        neumann_bcs   = nbcs,
        times         = times,
    )

    output_spec = _parse_output_spec(dict)
    is_dynamic  = _is_dynamic_integrator(dict)

    # Build VectorFunction list for PostProcessor (all nodal vars in one call).
    nodal_vars = _build_nodal_vars(V, output_spec, is_dynamic)
    pp = FEC.PostProcessor(mesh, output_file, nodal_vars...)

    # Register element variable names (stress, F, IVs) before first write.
    el_names = _element_var_names(asm_cpu, physics, output_spec)
    if !isempty(el_names)
        Exodus.write_names(pp.field_output_db, Exodus.ElementVariable, el_names)
    end

    if device == :rocm
        asm = Base.invokelatest(FEC.rocm, asm_cpu)
        p   = Base.invokelatest(FEC.rocm, p_cpu)
    elseif device == :cuda
        asm = Base.invokelatest(FEC.cuda, asm_cpu)
        p   = Base.invokelatest(FEC.cuda, p_cpu)
    else
        asm = asm_cpu
        p   = p_cpu
    end

    integrator = if device != :cpu
        Base.invokelatest(_parse_integrator, dict, asm, asm_cpu, p_cpu, controller, device)
    else
        _parse_integrator(dict, asm, asm_cpu, p_cpu, controller, device)
    end

    _apply_initial_velocity_ics!(integrator, mesh, asm_cpu, p_cpu,
                                  _parse_velocity_ics(dict))

    n_steps = controller.num_stops - 1
    sim = SingleDomainSimulation(p, p_cpu, asm_cpu, integrator, pp,
                                  controller, output_interval, device, output_spec)

    # Write initial state (step 1, t=0).
    write_output!(sim, 1)
    _carina_logf(0, :stop, "[0/%d,  0%%] : Time = %.4e", n_steps, controller.initial_time)
    _carina_log(0, :output, output_file)

    return sim
end

# ---------------------------------------------------------------------------
# evolve!
# ---------------------------------------------------------------------------

"""
    evolve!(sim::SingleDomainSimulation)

Run the full time loop, writing Exodus output at every `output_interval` stops.
"""
function evolve!(sim::SingleDomainSimulation)
    (; params, post_processor, controller, output_interval, device) = sim
    n_steps = controller.num_stops - 1

    for _ in 1:n_steps
        _advance_controller!(controller)
        t_prev = controller.prev_time
        t_stop = controller.time

        _carina_logf(4, :advance,
            "Control step [%d/%d] : [%.4e, %.4e] : Δt_c = %.4e",
            controller.stop, n_steps, t_prev, t_stop, controller.control_step)

        # Reset FEC clock to start of this control interval
        params.times.time_current = t_prev

        t_wall = time()
        _subcycle!(sim, t_stop)

        t   = controller.time
        pct = round(Int, 100 * controller.stop / n_steps)

        if controller.stop % output_interval == 0
            step  = controller.stop + 1
            write_output!(sim, step)
            u_max = maximum(abs, params.h1_field.data)
            _carina_logf(0, :stop,
                "[%d/%d, %3d%%] : Time = %.4e : |U|_max = %.3e : wall = %.2fs",
                controller.stop, n_steps, pct, t, u_max, time() - t_wall)
            _carina_log(0, :output, post_processor.field_output_db.file_name)
        else
            _carina_logf(0, :stop, "[%d/%d, %3d%%] : Time = %.4e : wall = %.2fs",
                         controller.stop, n_steps, pct, t, time() - t_wall)
        end
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

        _pre_step_hook!(integrator, params)
        _advance_one_step!(sim)

        isapprox(FEC.current_time(params.times), target;
                 rtol=1e-6, atol=1e-12) && break
    end
end

function _adjusted_step(t::Float64, dt::Float64, t_stop::Float64,
                         eps::Float64=0.01)::Float64
    gap    = t_stop - t
    t_next = t + dt
    return t_next >= t_stop - eps * dt ? gap : dt
end

function _advance_one_step!(sim)
    (; params, integrator, device) = sim
    _save_state!(integrator, params)

    while true
        integrator.failed[] = false

        if device != :cpu
            Base.invokelatest(FEC.evolve!, integrator, params)
        else
            FEC.evolve!(integrator, params)
        end

        if !integrator.failed[]
            _increase_step!(integrator, params)
            break
        end

        # Step failed: restore state, reduce step, undo time advance, retry
        _restore_state!(integrator, params)
        _decrease_step!(integrator, params)
        params.times.time_current -= params.times.Δt
        params.times.Δt = integrator.time_step
    end
end

# ---------------------------------------------------------------------------
# Internal parsers
# ---------------------------------------------------------------------------

# Resolve a file path relative to the YAML's directory.
function _resolve(dict, key, basedir)
    val = get(dict, key, nothing)
    val === nothing && error("Missing required YAML key: \"$key\"")
    isabspath(val) ? val : joinpath(basedir, val)
end

# ---- material ----

function _parse_material_section(dict)
    model_dict_top = get(dict, "model", nothing)
    model_dict_top === nothing && error("Missing \"model:\" section in YAML.")

    mat_section = get(model_dict_top, "material", nothing)
    mat_section === nothing && error("Missing \"model: material:\" section in YAML.")

    # Expect:  blocks: { block_name: model_name }  followed by model-specific keys.
    blocks = get(mat_section, "blocks", nothing)
    blocks === nothing && error("Missing \"material: blocks:\" mapping.")
    # Use the first (and for single-domain Phase 1, only) block's model name.
    model_name = first(values(blocks))

    # The model-specific sub-dict (e.g.  neohookean: { elastic modulus: ... })
    model_props = get(mat_section, model_name, nothing)
    model_props === nothing && error(
        "Material block \"$model_name\" listed in blocks but no property dict found."
    )

    return parse_material(model_name, model_props)
end

# ---- time ----

function _parse_times(dict)
    ti_dict = get(dict, "time integrator", nothing)
    ti_dict === nothing && error("Missing \"time integrator:\" section.")
    t0  = Float64(get(ti_dict, "initial time", 0.0))
    tf  = Float64(ti_dict["final time"])
    dt  = Float64(ti_dict["time step"])
    num_stops = round(Int, (tf - t0) / dt) + 1
    controller = TimeController(t0, tf, dt, t0, t0, num_stops, 0)
    # FEC.TimeStepper: used by FEC internals (BC evaluation, time queries).
    # Δt will be overwritten each sub-step during subcycling.
    times = FEC.TimeStepper(t0, tf, round(Int, (tf - t0) / dt))
    return controller, times
end

# ---- adaptive stepping ----

function _parse_adaptive_stepping(ti_dict, dt_nominal)
    has_min = haskey(ti_dict, "minimum time step")
    has_max = haskey(ti_dict, "maximum time step")
    has_dec = haskey(ti_dict, "decrease factor")
    has_inc = haskey(ti_dict, "increase factor")
    has_any = has_min || has_max || has_dec || has_inc
    has_all = has_min && has_max && has_dec && has_inc
    has_any && !has_all &&
        error("Adaptive time stepping requires all four: " *
              "\"minimum time step\", \"maximum time step\", " *
              "\"decrease factor\", \"increase factor\".")
    if has_all
        min_dt = Float64(ti_dict["minimum time step"])
        max_dt = Float64(ti_dict["maximum time step"])
        dec    = Float64(ti_dict["decrease factor"])
        inc    = Float64(ti_dict["increase factor"])
        dec >= 1.0 && error("\"decrease factor\" must be < 1.0")
        inc <= 1.0 && error("\"increase factor\" must be > 1.0")
        min_dt > max_dt && error("\"minimum time step\" > \"maximum time step\"")
    else
        min_dt = max_dt = dt_nominal
        dec = inc = 1.0
    end
    return min_dt, max_dt, dec, inc
end

# ---- integrator ----

function _parse_integrator(dict, asm, asm_cpu, p_cpu, controller, device=:cpu)
    ti_dict  = get(dict, "time integrator", nothing)
    ti_dict === nothing && error("Missing \"time integrator:\" section.")
    type_str = lowercase(get(ti_dict, "type", "quasi static"))
    dt       = controller.control_step

    if type_str in ("quasi static", "quasistatic", "static")
        min_dt, max_dt, dec, inc = _parse_adaptive_stepping(ti_dict, dt)
        sol_dict  = get(dict, "solver", Dict{String,Any}())
        ls_dict   = get(sol_dict, "linear solver", Dict{String,Any}())
        kry_type  = lowercase(get(ls_dict, "type", "direct"))
        use_lbfgs = kry_type == "lbfgs"

        solver = _parse_solver(dict, asm, device; force_direct=!use_lbfgs)

        if use_lbfgs
            lbfgs_m       = Int(get(ls_dict, "history size", 10))
            # Jacobi(K) preconditioner: scales H₀ so first step has units of
            # displacement (m) rather than force (N).  Without it, α=1 overshoots
            # by ~E and all backtracking halvings fail to reduce the residual.
            lbfgs_precond = _compute_stiffness_jacobi_precond(asm_cpu, p_cpu,
                                                               solver.linear_solver.ΔUu)
            return QuasiStaticLBFGSIntegrator(solver, lbfgs_m;
                                               precond=lbfgs_precond,
                                               time_step=dt,
                                               min_time_step=min_dt,
                                               max_time_step=max_dt,
                                               decrease_factor=dec,
                                               increase_factor=inc)
        end
        return QuasiStaticIntegrator(solver, dt, min_dt, max_dt, dec, inc)

    elseif type_str in ("newmark", "newmark-beta", "newmark beta")
        β = Float64(get(ti_dict, "beta",  0.25))
        γ = Float64(get(ti_dict, "gamma", 0.5))
        min_dt, max_dt, dec, inc = _parse_adaptive_stepping(ti_dict, dt)

        sol_dict    = get(dict, "solver", Dict{String,Any}())
        ls_dict     = get(sol_dict, "linear solver", Dict{String,Any}())
        kry_type    = lowercase(get(ls_dict, "type", "minres"))
        kry_itmax   = Int(get(ls_dict, "maximum iterations",
                            Int(get(ti_dict, "krylov iterations", 1000))))
        kry_rtol    = Float64(get(ls_dict, "tolerance",
                            Float64(get(ti_dict, "krylov tolerance", 1e-8))))
        use_direct  = kry_type == "direct"
        use_lbfgs   = kry_type == "lbfgs"
        device != :cpu && use_direct && error(
            "\"linear solver: type: direct\" is CPU-only.")
        kry_method  = kry_type in ("cg", "conjugate gradient") ? :cg : :minres

        precond_dict = get(ls_dict, "preconditioner", Dict{String,Any}())
        precond_type = Symbol(lowercase(get(precond_dict, "type", "none")))

        solver = _parse_solver(dict, asm, device; force_direct=true)

        if use_lbfgs
            lbfgs_m = Int(get(ls_dict, "history size", 10))
            # Jacobi preconditioner for H₀: essential when c_M = 1/(β·Δt²) ≫ 1.
            # Without it, the first step direction has units of force, α=1 overshoots
            # by ~c_M, and all 10 backtracking halves fail to reduce the residual.
            lbfgs_precond = _compute_jacobi_precond(β, dt, asm_cpu, p_cpu,
                                                     solver.linear_solver.ΔUu)
            return NewmarkLBFGSIntegrator(solver, β, γ, lbfgs_m;
                                           precond=lbfgs_precond,
                                           time_step=dt,
                                           min_time_step=min_dt,
                                           max_time_step=max_dt,
                                           decrease_factor=dec,
                                           increase_factor=inc)
        end

        precond = if !use_direct && precond_type == :jacobi
            _compute_jacobi_precond(β, dt, asm_cpu, p_cpu, solver.linear_solver.ΔUu)
        else
            NoPreconditioner()
        end

        return NewmarkIntegrator(solver, β, γ;
                                  use_direct=use_direct,
                                  krylov_method=kry_method,
                                  krylov_itmax=kry_itmax,
                                  krylov_rtol=kry_rtol,
                                  precond=precond,
                                  time_step=dt,
                                  min_time_step=min_dt,
                                  max_time_step=max_dt,
                                  decrease_factor=dec,
                                  increase_factor=inc)

    elseif type_str in ("central difference", "centraldifference", "cd")
        γ = Float64(get(ti_dict, "gamma", get(ti_dict, "γ", 0.5)))
        min_dt, max_dt, dec, inc = _parse_adaptive_stepping(ti_dict, dt)
        CFL_val = Float64(get(ti_dict, "CFL", get(ti_dict, "cfl", 0.0)))

        solver  = _parse_solver(dict, asm, device; force_direct=true)
        m_lumped = _compute_lumped_mass(asm_cpu, p_cpu, solver.linear_solver.ΔUu)

        return CentralDifferenceIntegrator(γ, asm, m_lumped;
                                            time_step=dt,
                                            min_time_step=min_dt,
                                            max_time_step=max_dt,
                                            decrease_factor=dec,
                                            increase_factor=inc,
                                            CFL=CFL_val,
                                            c_p_max=Inf)
    else
        error("Unknown time integrator type: \"$type_str\". " *
              "Supported: \"quasi static\", \"Newmark\", \"central difference\".")
    end
end

# Compute the Jacobi (diagonal) preconditioner for the Newmark effective
# stiffness K + c_M·M using the mass-only approximation d_ii ≈ c_M·M_ii.
#
# The diagonal is computed on CPU (asm_cpu, p_cpu) and then transferred to
# the target device (ΔUu_template may be a GPU array).
function _compute_jacobi_precond(β, Δt, asm_cpu, p_cpu, ΔUu_template)
    c_M   = 1.0 / (β * Δt^2)
    n_cpu = length(ΔUu_template)
    ones_v  = ones(Float64, n_cpu)
    U_zeros = zeros(Float64, n_cpu)

    # M · ones  gives the lumped-mass row sums for each unknown DOF.
    FEC.assemble_matrix_action!(asm_cpu, FEC.mass, U_zeros, ones_v, p_cpu)
    m_diag = copy(FEC.hvp(asm_cpu, ones_v))   # copy before next assembly overwrites

    inv_diag_cpu = 1.0 ./ (c_M .* m_diag)

    # Transfer to device (no-op if already CPU).
    inv_diag = similar(ΔUu_template)
    copyto!(inv_diag, inv_diag_cpu)
    return JacobiPreconditioner(inv_diag)
end

# Compute the Jacobi (diagonal) preconditioner for the quasi-static tangent
# stiffness K(U=0) via K·ones.  Used as H₀ in L-BFGS so that the first step
# has units of displacement rather than force.
function _compute_stiffness_jacobi_precond(asm_cpu, p_cpu, ΔUu_template)
    n_cpu   = length(ΔUu_template)
    ones_v  = ones(Float64, n_cpu)
    U_zeros = zeros(Float64, n_cpu)

    FEC.assemble_matrix_action!(asm_cpu, FEC.stiffness, U_zeros, ones_v, p_cpu)
    k_diag = copy(FEC.hvp(asm_cpu, ones_v))

    inv_diag_cpu = 1.0 ./ max.(abs.(k_diag), eps(Float64))

    inv_diag = similar(ΔUu_template)
    copyto!(inv_diag, inv_diag_cpu)
    return JacobiPreconditioner(inv_diag)
end

# Compute the lumped mass vector (diagonal row sums of the consistent mass
# matrix) via M·ones.  The result is on the same device as ΔUu_template.
function _compute_lumped_mass(asm_cpu, p_cpu, ΔUu_template)
    n_cpu   = length(ΔUu_template)
    ones_v  = ones(Float64, n_cpu)
    U_zeros = zeros(Float64, n_cpu)

    FEC.assemble_matrix_action!(asm_cpu, FEC.mass, U_zeros, ones_v, p_cpu)
    m_cpu = copy(FEC.hvp(asm_cpu, ones_v))

    m_out = similar(ΔUu_template)
    copyto!(m_out, m_cpu)
    return m_out
end

# ---- solver ----

function _parse_solver(dict, asm, device=:cpu; force_direct=false)
    # Solver section is optional; Newton with defaults if absent.
    sol_dict  = get(dict, "solver", Dict{String,Any}())
    max_iters = Int(get(sol_dict, "maximum iterations", 20))
    abs_tol   = Float64(get(sol_dict, "absolute tolerance", 1e-10))
    rel_tol   = Float64(get(sol_dict, "relative tolerance", 1e-14))
    # GPU quasi-static requires iterative solver (direct \ is CPU-only).
    # force_direct overrides this (used by Newmark which never calls linear solve).
    sol_type  = (!force_direct && device != :cpu) ? "iterative" :
                lowercase(get(sol_dict, "type", "direct"))

    # Per-iteration logging callback for FEC.NewtonSolver (quasi-static path).
    newton_cb = (iter, norm_ΔUu, norm_R, rel_R, converged) -> begin
        status = _status_str(converged)
        _carina_logf(8, :solve, "Iter [%d] |R| = %.3e : |r| = %.3e : %s",
                     iter, norm_R, rel_R, status)
    end

    # FEC.NewtonSolver takes (max_iters, abs_increment_tol, abs_residual_tol, rel_residual_tol, linear_solver, timer, log_callback)
    if sol_type == "direct"
        linear = FEC.DirectLinearSolver(asm)
        return FEC.NewtonSolver(max_iters, abs_tol, abs_tol, rel_tol, linear, linear.timer, newton_cb)
    elseif sol_type in ("iterative", "krylov", "gmres")
        linear = FEC.IterativeLinearSolver(asm, :gmres)
        return FEC.NewtonSolver(max_iters, abs_tol, abs_tol, rel_tol, linear, linear.timer, newton_cb)
    else
        error("Unknown solver type \"$sol_type\". Supported: \"direct\", \"iterative\".")
    end
end

# ---- Dirichlet BCs ----

function _parse_dirichlet_bcs(dict)
    bc_section = get(dict, "boundary conditions", nothing)
    bc_section === nothing && return FEC.DirichletBC[]
    entries = get(bc_section, "Dirichlet", FEC.DirichletBC[])
    entries isa Vector || error("\"Dirichlet:\" must be a list.")

    dbcs = FEC.DirichletBC[]
    for entry in entries
        var_sym  = _component_to_symbol(entry["component"])
        func     = _make_function(entry["function"])
        # Accept either sideset or nodeset
        if haskey(entry, "sideset")
            push!(dbcs, FEC.DirichletBC(var_sym, func;
                sideset_name = Symbol(entry["sideset"])))
        elseif haskey(entry, "nodeset")
            push!(dbcs, FEC.DirichletBC(var_sym, func;
                nodeset_name = Symbol(entry["nodeset"])))
        else
            error("Dirichlet BC entry must specify \"sideset\" or \"nodeset\".")
        end
    end
    return dbcs
end

# ---- Neumann BCs ----

function _parse_neumann_bcs(dict)
    bc_section = get(dict, "boundary conditions", nothing)
    bc_section === nothing && return FEC.NeumannBC[]
    entries = get(bc_section, "Neumann", FEC.NeumannBC[])
    entries isa Vector || error("\"Neumann:\" must be a list.")

    nbcs = FEC.NeumannBC[]
    for entry in entries
        var_sym  = _component_to_symbol(entry["component"])
        func     = _make_function(entry["function"])
        sset     = Symbol(entry["sideset"])
        push!(nbcs, FEC.NeumannBC(var_sym, func, sset))
    end
    return nbcs
end

# ---- Helpers ----

# Map "x" / "y" / "z" → :displ_x / :displ_y / :displ_z
function _component_to_symbol(comp::String)
    c = lowercase(strip(comp))
    c == "x" && return :displ_x
    c == "y" && return :displ_y
    c == "z" && return :displ_z
    error("Unknown component \"$comp\". Expected x, y, or z.")
end

# Turn a YAML function string into a Julia (coords, t) -> value closure.
# Supported variables in the expression: t, x, y, z (node coordinates).
#
# @eval creates the method at the current world, but FEC's generic
# _update_bc_values! is JIT-compiled for the specific closure type the first
# time it is called (i.e. at the same world or later), so no invokelatest is
# needed.  Avoiding invokelatest is essential for GPU kernels, which cannot
# emit calls to Julia runtime functions like jl_f_invokelatest.
function _make_function(expr_str::String)
    body = Meta.parse(expr_str)
    # Multi-statement strings like "a=8000; expr" parse as Expr(:toplevel,...),
    # which is invalid inside a function body; convert to Expr(:block,...).
    if body isa Expr && body.head === :toplevel
        body = Expr(:block, body.args...)
    end
    return @eval (coords, t) -> begin
        x = coords[1]; y = coords[2]; z = coords[3]
        $body
    end
end

# ---- initial conditions ----

function _parse_velocity_ics(dict)
    ic_dict = get(dict, "initial conditions", nothing)
    ic_dict === nothing && return Any[]
    vel_ics = get(ic_dict, "velocity", Any[])
    vel_ics isa Vector || error("\"initial conditions: velocity:\" must be a list.")
    return vel_ics
end

# Apply initial velocity ICs to the Newmark velocity vector V.
# Each entry: {node set: <name>, component: x|y|z, function: <expr>}
function _apply_initial_velocity_ics!(integrator::NewmarkIntegrator, mesh, asm_cpu, p_cpu, vel_ics)
    isempty(vel_ics) && return
    dof = asm_cpu.dof                # always CPU dof manager for index arithmetic
    X   = p_cpu.h1_coords.data       # flat, node-major: [x₁,y₁,z₁, x₂,y₂,z₂, ...]

    # Inverse map: full_dof_idx -> index in unknown_dofs (0 = constrained DOF)
    n_unk   = length(dof.unknown_dofs)
    inv_map = zeros(Int, length(dof))
    for (i, fd) in enumerate(dof.unknown_dofs)
        inv_map[fd] = i
    end

    # Build on CPU first, then copy to integrator.V (which may be a GPU array).
    V_init = zeros(Float64, n_unk)
    for entry in vel_ics
        var_sym  = _component_to_symbol(entry["component"])
        func     = _make_function(entry["function"])
        nset_sym = Symbol(entry["node set"])
        bk       = FEC.BCBookKeeping(mesh, dof, var_sym; nset_name=nset_sym)
        for (full_dof, node) in zip(bk.dofs, bk.nodes)
            unk_idx = inv_map[full_dof]
            unk_idx == 0 && continue   # skip constrained DOFs
            coords = @view X[(node-1)*3+1 : (node-1)*3+3]
            V_init[unk_idx] = Base.invokelatest(func, coords, 0.0)
        end
    end
    Base.invokelatest(copyto!, integrator.V, V_init)
end

# NewmarkLBFGSIntegrator — identical logic to NewmarkIntegrator.
function _apply_initial_velocity_ics!(integrator::NewmarkLBFGSIntegrator, mesh, asm_cpu, p_cpu, vel_ics)
    isempty(vel_ics) && return
    dof = asm_cpu.dof
    X   = p_cpu.h1_coords.data

    n_unk   = length(dof.unknown_dofs)
    inv_map = zeros(Int, length(dof))
    for (i, fd) in enumerate(dof.unknown_dofs)
        inv_map[fd] = i
    end

    V_init = zeros(Float64, n_unk)
    for entry in vel_ics
        var_sym  = _component_to_symbol(entry["component"])
        func     = _make_function(entry["function"])
        nset_sym = Symbol(entry["node set"])
        bk       = FEC.BCBookKeeping(mesh, dof, var_sym; nset_name=nset_sym)
        for (full_dof, node) in zip(bk.dofs, bk.nodes)
            unk_idx = inv_map[full_dof]
            unk_idx == 0 && continue
            coords = @view X[(node-1)*3+1 : (node-1)*3+3]
            V_init[unk_idx] = Base.invokelatest(func, coords, 0.0)
        end
    end
    Base.invokelatest(copyto!, integrator.V, V_init)
end

# CentralDifferenceIntegrator — same logic as Newmark, copies into integrator.V.
function _apply_initial_velocity_ics!(integrator::CentralDifferenceIntegrator, mesh, asm_cpu, p_cpu, vel_ics)
    isempty(vel_ics) && return
    dof = asm_cpu.dof
    X   = p_cpu.h1_coords.data

    n_unk   = length(dof.unknown_dofs)
    inv_map = zeros(Int, length(dof))
    for (i, fd) in enumerate(dof.unknown_dofs)
        inv_map[fd] = i
    end

    V_init = zeros(Float64, n_unk)
    for entry in vel_ics
        var_sym  = _component_to_symbol(entry["component"])
        func     = _make_function(entry["function"])
        nset_sym = Symbol(entry["node set"])
        bk       = FEC.BCBookKeeping(mesh, dof, var_sym; nset_name=nset_sym)
        for (full_dof, node) in zip(bk.dofs, bk.nodes)
            unk_idx = inv_map[full_dof]
            unk_idx == 0 && continue
            coords = @view X[(node-1)*3+1 : (node-1)*3+3]
            V_init[unk_idx] = Base.invokelatest(func, coords, 0.0)
        end
    end
    Base.invokelatest(copyto!, integrator.V, V_init)
end

# No-op for integrators that do not support initial velocity ICs.
function _apply_initial_velocity_ics!(integrator, mesh, asm_cpu, p_cpu, vel_ics)
    isempty(vel_ics) || @warn "Initial velocity ICs ignored for non-Newmark integrator."
end
