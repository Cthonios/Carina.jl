# YAML-driven simulation factory and time loop.
#
# Entry point: Carina.run(yaml_file)
#
# create_simulation(dict)  reads a parsed YAML dict and builds a
#   SingleDomainSimulation ready to evolve.
#
# evolve!(sim)  runs the full time loop and writes Exodus output.

import FiniteElementContainers as FEC
import ConstitutiveModels as CM
import YAML

# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

"""
    Carina.run(yaml_file)

Load `yaml_file`, create a simulation, run it, and close the output file.
"""
function run(yaml_file::String)
    dict = YAML.load_file(yaml_file; dicttype=Dict{String,Any})
    sim_type = lowercase(get(dict, "type", "single"))
    if sim_type == "single"
        sim = create_simulation(dict, dirname(abspath(yaml_file)))
        evolve!(sim)
        FEC.close(sim.post_processor)
        @info "Done. Output written to $(sim.post_processor.field_output_db.file_name)"
    else
        error("Simulation type \"$sim_type\" not yet supported. Only \"single\" is implemented.")
    end
    return nothing
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
    # --- files ---
    input_mesh  = _resolve(dict, "input mesh file",  basedir)
    output_file = _resolve(dict, "output mesh file", basedir)
    output_interval = Int(get(dict, "output interval", 1))

    # --- material / physics ---
    cm, density, props_inputs = _parse_material_section(dict)
    props   = create_solid_mechanics_properties(cm, props_inputs)
    physics = SolidMechanics(cm, density)

    # --- mesh + function space + assembler ---
    mesh = FEC.UnstructuredMesh(input_mesh)
    V    = FEC.FunctionSpace(mesh, FEC.H1Field, FEC.Lagrange)
    u    = FEC.VectorFunction(V, :displ)
    asm  = FEC.SparseMatrixAssembler(u; use_condensed=true)

    # --- boundary conditions ---
    dbcs = _parse_dirichlet_bcs(dict)
    nbcs = _parse_neumann_bcs(dict)

    # --- time ---
    times, n_steps = _parse_times(dict)

    # --- FEC parameters ---
    p = FEC.create_parameters(
        mesh, asm, physics, props;
        dirichlet_bcs = dbcs,
        neumann_bcs   = nbcs,
        times         = times,
    )

    # --- integrator ---
    integrator = _parse_integrator(dict, asm)

    # --- post-processor (write step 0) ---
    pp = FEC.PostProcessor(mesh, output_file, u)
    FEC.write_times(pp, 1, 0.0)
    FEC.write_field(pp, 1, ("displ_x", "displ_y", "displ_z"), p.h1_field)

    return SingleDomainSimulation(p, integrator, pp, n_steps, output_interval)
end

# ---------------------------------------------------------------------------
# evolve!
# ---------------------------------------------------------------------------

"""
    evolve!(sim::SingleDomainSimulation)

Run the full time loop, writing Exodus output at every `output_interval` steps.
"""
function evolve!(sim::SingleDomainSimulation)
    (; params, integrator, post_processor, n_steps, output_interval) = sim
    for n in 1:n_steps
        FEC.evolve!(integrator, params)
        t = FEC.current_time(params.times)
        if n % output_interval == 0
            # PostProcessor step index starts at 2 (step 1 = initial state)
            step = n + 1
            FEC.write_times(post_processor, step, t)
            FEC.write_field(post_processor, step,
                ("displ_x", "displ_y", "displ_z"), params.h1_field)
        end
        @info "Step $n  t=$(round(t, digits=6))  " *
              "max|u_z|=$(round(maximum(abs, params.h1_field[3,:]), sigdigits=4))"
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

function _parse_integrator(dict, asm)
    ti_dict = get(dict, "time integrator", nothing)
    ti_dict === nothing && error("Missing \"time integrator:\" section.")
    type_str = lowercase(get(ti_dict, "type", "quasi static"))

    solver = _parse_solver(dict, asm)

    if type_str in ("quasi static", "quasistatic", "static")
        return FEC.QuasiStaticIntegrator(solver)
    elseif type_str in ("newmark", "newmark-beta", "newmark beta")
        β = Float64(get(ti_dict, "beta",  0.25))
        γ = Float64(get(ti_dict, "gamma", 0.5))
        return NewmarkIntegrator(solver, β, γ)
    else
        error("Unknown time integrator type: \"$type_str\". " *
              "Supported: \"quasi static\", \"Newmark\".")
    end
end

# ---- solver ----

function _parse_solver(dict, asm)
    # Solver section is optional; Newton with defaults if absent.
    sol_dict = get(dict, "solver", Dict{String,Any}())
    max_iters  = Int(get(sol_dict, "maximum iterations", 20))
    abs_tol    = Float64(get(sol_dict, "absolute tolerance", 1e-10))
    # FEC.NewtonSolver takes (max_iters, abs_increment_tol, abs_residual_tol, linear_solver, timer)
    linear = FEC.DirectLinearSolver(asm)
    return FEC.NewtonSolver(max_iters, abs_tol, abs_tol, linear, linear.timer)
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
# @eval creates a method in a newer Julia world than FEC's compiled code, so
# FEC cannot call it directly.  We wrap it in a plain closure (whose method
# body was compiled at Carina load time) that forwards the call via
# Base.invokelatest, which bypasses the world-age check.
function _make_function(expr_str::String)
    body  = Meta.parse(expr_str)
    inner = @eval (coords, t) -> begin
        x = coords[1]; y = coords[2]; z = coords[3]
        $body
    end
    # The outer closure is a value created at runtime; its method body was
    # compiled when Carina loaded, so FEC can call it without world-age error.
    return (coords, t) -> Base.invokelatest(inner, coords, t)
end
