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
# On-demand AMDGPU loading (avoid paying the init cost on CPU-only runs)
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

# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

"""
    Carina.run(yaml_file)

Load `yaml_file`, create a simulation, run it, and close the output file.
"""
function run(yaml_file::String)
    t_start = time()
    _carina_log(0, :carina, "BEGIN SIMULATION")
    _carina_log(0, :setup,  "Reading from $yaml_file")

    dict = YAML.load_file(yaml_file; dicttype=Dict{String,Any})
    sim_type = lowercase(get(dict, "type", "single"))
    if sim_type == "single"
        sim = create_simulation(dict, dirname(abspath(yaml_file)))
        evolve!(sim)
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
    create_simulation(dict, basedir="") -> SingleDomainSimulation

Parse a YAML dict (already loaded) and return a fully initialised simulation.
`basedir` is used to resolve relative file paths inside the YAML.
"""
function create_simulation(dict::Dict{String,Any}, basedir::String="")
    # --- device ---
    device_str = lowercase(get(dict, "device", "cpu"))
    use_gpu = device_str == "rocm"
    use_gpu && _require_amdgpu!()
    _carina_log(0, :device, use_gpu ? "ROCm GPU" : "CPU")

    # --- files ---
    input_mesh  = _resolve(dict, "input mesh file",  basedir)
    output_file = _resolve(dict, "output mesh file", basedir)
    output_interval = Int(get(dict, "output interval", 1))
    _carina_log(0, :setup, "Mesh:   $input_mesh")
    _carina_log(0, :setup, "Output: $output_file")

    # --- material / physics ---
    cm, density, props_inputs = _parse_material_section(dict)
    props   = create_solid_mechanics_properties(cm, props_inputs)
    physics = SolidMechanics(cm, density)

    # --- mesh + function space + CPU assembler ---
    mesh    = FEC.UnstructuredMesh(input_mesh)
    V       = FEC.FunctionSpace(mesh, FEC.H1Field, FEC.Lagrange)
    u       = FEC.VectorFunction(V, :displ)
    asm_cpu = FEC.SparseMatrixAssembler(u; use_condensed=true)

    # --- boundary conditions ---
    dbcs = _parse_dirichlet_bcs(dict)
    nbcs = _parse_neumann_bcs(dict)

    # --- time ---
    times, n_steps = _parse_times(dict)

    # --- FEC parameters (always built on CPU) ---
    p_cpu = FEC.create_parameters(
        mesh, asm_cpu, physics, props;
        dirichlet_bcs = dbcs,
        neumann_bcs   = nbcs,
        times         = times,
    )

    # --- post-processor: step 0 written before GPU adaptation ---
    pp = FEC.PostProcessor(mesh, output_file, u)
    FEC.write_times(pp, 1, 0.0)
    FEC.write_field(pp, 1, ("displ_x", "displ_y", "displ_z"), p_cpu.h1_field)
    _carina_logf(0, :stop,   "[0/%d,  0%%] : Time = %.4e", n_steps, FEC.current_time(times))
    _carina_log(0,  :output, output_file)

    # --- GPU adaptation (after step-0 I/O so Exodus sees CPU arrays) ---
    if use_gpu
        asm = Base.invokelatest(FEC.rocm, asm_cpu)
        p   = Base.invokelatest(FEC.rocm, p_cpu)
    else
        asm = asm_cpu
        p   = p_cpu
    end

    # --- integrator (forces iterative linear solver when use_gpu) ---
    # Use invokelatest when on GPU: AMDGPU was loaded via Base.require, so its
    # methods (e.g. similar(::ROCArray)) are in a newer world than Carina's
    # compilation world; invokelatest lets the whole call chain see them.
    integrator = if use_gpu
        Base.invokelatest(_parse_integrator, dict, asm, use_gpu)
    else
        _parse_integrator(dict, asm, use_gpu)
    end

    # --- initial conditions ---
    # asm_cpu passed explicitly so IC evaluation always uses CPU dof/coords,
    # then copyto! transfers the result to integrator.V (CPU or GPU array).
    _apply_initial_velocity_ics!(integrator, mesh, asm_cpu, p_cpu,
                                  _parse_velocity_ics(dict))

    return SingleDomainSimulation(p, integrator, pp, n_steps, output_interval, use_gpu)
end

# ---------------------------------------------------------------------------
# evolve!
# ---------------------------------------------------------------------------

"""
    evolve!(sim::SingleDomainSimulation)

Run the full time loop, writing Exodus output at every `output_interval` steps.
"""
function evolve!(sim::SingleDomainSimulation)
    (; params, integrator, post_processor, n_steps, output_interval, use_gpu) = sim

    for n in 1:n_steps
        t_prev = use_gpu ? Base.invokelatest(FEC.current_time, params.times) :
                           FEC.current_time(params.times)
        Δt = use_gpu ? Base.invokelatest(FEC.time_step, params.times) :
                       FEC.time_step(params.times)
        _carina_logf(4, :advance, "Time = [%.4e, %.4e] : Δt = %.4e",
                     t_prev, t_prev + Δt, Δt)

        t_step = time()

        # When running on GPU, FEC methods dispatching on ROCArray types are
        # defined in a newer Julia world (AMDGPU was loaded via Base.require).
        # invokelatest lets the entire call chain see those methods.
        if use_gpu
            Base.invokelatest(FEC.evolve!, integrator, params)
        else
            FEC.evolve!(integrator, params)
        end

        t = use_gpu ? Base.invokelatest(FEC.current_time, params.times) :
                      FEC.current_time(params.times)
        wall = time() - t_step
        pct  = round(Int, 100 * n / n_steps)

        if n % output_interval == 0
            # PostProcessor step index starts at 2 (step 1 = initial state).
            # Adapt to CPU; when on GPU, Adapt.adapt(ROCArray→Array) dispatches
            # methods from the AMDGPU world, so invokelatest is required.
            step   = n + 1
            h1_cpu = use_gpu ? Base.invokelatest(Adapt.adapt, Array, params.h1_field) :
                                Adapt.adapt(Array, params.h1_field)
            FEC.write_times(post_processor, step, t)
            FEC.write_field(post_processor, step,
                ("displ_x", "displ_y", "displ_z"), h1_cpu)
            u_max = maximum(abs, h1_cpu.data)
            _carina_logf(0, :stop,   "[%d/%d, %3d%%] : Time = %.4e : |U|_max = %.3e : wall = %.2fs",
                         n, n_steps, pct, t, u_max, wall)
            _carina_log(0,  :output, post_processor.field_output_db.file_name)
        else
            _carina_logf(0, :stop, "[%d/%d, %3d%%] : Time = %.4e : wall = %.2fs",
                         n, n_steps, pct, t, wall)
        end
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
    t0 = Float64(get(ti_dict, "initial time", 0.0))
    tf = Float64(ti_dict["final time"])
    dt = Float64(ti_dict["time step"])
    n_steps = round(Int, (tf - t0) / dt)
    times   = FEC.TimeStepper(t0, tf, n_steps)
    return times, n_steps
end

# ---- integrator ----

function _parse_integrator(dict, asm, use_gpu=false)
    ti_dict = get(dict, "time integrator", nothing)
    ti_dict === nothing && error("Missing \"time integrator:\" section.")
    type_str = lowercase(get(ti_dict, "type", "quasi static"))

    if type_str in ("quasi static", "quasistatic", "static")
        solver = _parse_solver(dict, asm, use_gpu)
        return FEC.QuasiStaticIntegrator(solver)
    elseif type_str in ("newmark", "newmark-beta", "newmark beta")
        β            = Float64(get(ti_dict, "beta",  0.25))
        γ            = Float64(get(ti_dict, "gamma", 0.5))
        kry_itmax    = Int(get(ti_dict, "krylov iterations", 1000))
        kry_rtol     = Float64(get(ti_dict, "krylov tolerance", 1e-8))
        # Newmark uses matrix-free MINRES and never calls linear_solver.solve!.
        # Always use DirectLinearSolver so no sparse K is materialized (on GPU
        # this avoids an OOM from IterativeLinearSolver's eager stiffness build).
        solver = _parse_solver(dict, asm, use_gpu; force_direct=true)
        return NewmarkIntegrator(solver, β, γ;
                                  krylov_itmax=kry_itmax, krylov_rtol=kry_rtol)
    else
        error("Unknown time integrator type: \"$type_str\". " *
              "Supported: \"quasi static\", \"Newmark\".")
    end
end

# ---- solver ----

function _parse_solver(dict, asm, use_gpu=false; force_direct=false)
    # Solver section is optional; Newton with defaults if absent.
    sol_dict  = get(dict, "solver", Dict{String,Any}())
    max_iters = Int(get(sol_dict, "maximum iterations", 20))
    abs_tol   = Float64(get(sol_dict, "absolute tolerance", 1e-10))
    rel_tol   = Float64(get(sol_dict, "relative tolerance", 1e-14))
    # GPU quasi-static requires iterative solver (direct \ is CPU-only).
    # force_direct overrides this (used by Newmark which never calls linear solve).
    sol_type  = (!force_direct && use_gpu) ? "iterative" :
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

# No-op for non-Newmark integrators.
function _apply_initial_velocity_ics!(integrator, mesh, asm_cpu, p_cpu, vel_ics)
    isempty(vel_ics) || @warn "Initial velocity ICs ignored for non-Newmark integrator."
end
