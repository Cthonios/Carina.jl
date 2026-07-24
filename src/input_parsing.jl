# YAML parsing and factory helpers for simulation construction.
#
# All functions that translate YAML dict entries into Julia objects
# (integrators, solvers, BCs, materials, etc.) live here.

import FiniteElementContainers as FEC
import ConstitutiveModels as CM
import Krylov
using StaticArrays
using LinearAlgebra

# ---------------------------------------------------------------------------
# Key validation
# ---------------------------------------------------------------------------
#
# Each input section has a set of known keys. Unknown keys are flagged with
# a suggestion (Levenshtein distance) to catch typos early.

# Levenshtein distance for fuzzy matching
function _levenshtein(a::AbstractString, b::AbstractString)
    la, lb = length(a), length(b)
    la == 0 && return lb
    lb == 0 && return la
    prev = collect(0:lb)
    curr = similar(prev)
    for i in 1:la
        curr[1] = i
        for j in 1:lb
            cost = a[i] == b[j] ? 0 : 1
            curr[j+1] = min(prev[j+1] + 1, curr[j] + 1, prev[j] + cost)
        end
        prev, curr = curr, prev
    end
    return prev[lb+1]
end

function _suggest(key::String, known::Set{String}; max_dist::Int=5)
    best_dist = max_dist + 1
    best_key  = ""
    for k in known
        d = _levenshtein(lowercase(key), lowercase(k))
        if d < best_dist
            best_dist = d
            best_key = k
        end
    end
    best_dist <= max_dist ? best_key : ""
end

"""
Check that all keys in `dict` are in `known_keys`. Warn for unknown keys
with a "did you mean?" suggestion. `section` is used in the message.
"""
function _validate_keys(dict::AbstractDict, known_keys::Set{String}, section::String)
    for key in keys(dict)
        key isa String || continue
        if key ∉ known_keys
            suggestion = _suggest(key, known_keys)
            msg = "Unknown key \"$key\" in $section."
            if !isempty(suggestion)
                msg *= " Did you mean \"$suggestion\"?"
            end
            _carina_log(0, :warning, msg)
        end
    end
end

"""
Case-insensitive variant of `_validate_keys`, for sections whose sub-keys are
themselves looked up case-insensitively (see `_get_ci`).
"""
function _validate_keys_ci(dict::AbstractDict, known_keys::Set{String}, section::String)
    for key in keys(dict)
        key isa String || continue
        if lowercase(strip(key)) ∉ known_keys
            suggestion = _suggest(key, known_keys)
            msg = "Unknown key \"$key\" in $section."
            if !isempty(suggestion)
                msg *= " Did you mean \"$suggestion\"?"
            end
            _carina_log(0, :warning, msg)
        end
    end
end

"""
Case-insensitive sub-key lookup.

The input file is parsed with `dicttype=Dict{String,Any}` and no case folding,
so a plain `get(section, "dirichlet", default)` silently misses a user's
`Dirichlet:`.  For boundary conditions that failure mode is particularly bad:
the run proceeds with no constraints at all rather than reporting an error.
"""
function _get_ci(dict::AbstractDict, key::String, default=nothing)
    haskey(dict, key) && return dict[key]
    lk = lowercase(key)
    for (k, v) in dict
        k isa String && lowercase(strip(k)) == lk && return v
    end
    return default
end

"""
Error unless every key in `required` is present in `dict`.

Used for entries whose fields are read with `entry["..."]`.  Without this the
missing key surfaces as a bare `KeyError` raised several frames deep inside
FEC, which tells the user nothing about which input entry is at fault.
"""
function _require_keys(dict::AbstractDict, required, section::String)
    for key in required
        haskey(dict, key) || error(
            "$section is missing required key \"$key\". Need: $(join(required, ", ")).")
    end
end

"""
Error unless `name` is one of `valid`, with a "did you mean" suggestion.

`valid` is a collection of names read from the mesh (element blocks, node sets,
side sets).  A mistyped name is otherwise either silently ignored — a material
assigned to a block that does not exist still runs, on every block — or raised
as a `KeyError` from deep inside FEC with no indication of which input entry
produced it.
"""
function _check_mesh_name(name::AbstractString, valid, kind::String, section::String)
    name_str = String(name)
    name_str in valid && return name_str
    known = Set{String}(String(v) for v in valid)
    suggestion = _suggest(name_str, known)
    msg = "$section refers to $kind \"$name_str\", which is not in the mesh."
    if !isempty(suggestion)
        msg *= " Did you mean \"$suggestion\"?"
    end
    msg *= " Available: $(join(sort(collect(known)), ", "))."
    error(msg)
end

# --- Known keys per section ---

const _TOPLEVEL_KEYS = Set([
    "type", "device", "input mesh file", "output mesh file", "output interval",
    "output", "model", "time integrator", "boundary conditions", "body forces",
    "initial conditions", "solver", "quadrature",
])

const _TIME_INTEGRATOR_KEYS = Set([
    "type", "initial time", "final time", "time step",
    "minimum time step", "maximum time step", "decrease factor", "increase factor",
    "initial equilibrium",
    "beta", "gamma", "β", "γ", "alpha",
    "CFL", "cfl", "stable time step interval",
])

const _SOLVER_KEYS = Set([
    "type", "minimum iterations", "maximum iterations",
    "absolute tolerance", "relative tolerance", "termination",
    "use line search", "line search backtrack factor",
    "line search decrease factor", "line search maximum iterations",
    "linear solver", "preconditioner",
    "orthogonality tolerance", "restart interval",
])

const _TERMINATION_TEST_KEYS = Set([
    "type", "tolerance", "combo", "tests",
    "window", "threshold", "value",
])

const _LINEAR_SOLVER_KEYS = Set([
    "type", "maximum iterations", "tolerance", "history size",
    "preconditioner", "assembled",
])

const _BC_SECTION_KEYS = Set(["dirichlet", "neumann"])

const _MODEL_KEYS = Set(["type", "material"])

# Physics declared by `model.type`.  Only solid mechanics is implemented;
# the key is accepted (and required to name a supported physics) so that a
# future thermal or coupled model does not silently run as solid mechanics.
const _MODEL_TYPES = Set(["solid mechanics", "solidmechanics", "mechanics"])

const _QUADRATURE_KEYS = Set(["type", "order"])

const _IC_SECTION_KEYS = Set(["displacement", "velocity", "traveling wave"])

const _DBC_ENTRY_KEYS = Set(["side set", "node set", "component", "function"])
const _NBC_ENTRY_KEYS = Set(["side set", "node set", "component", "function"])
const _BF_ENTRY_KEYS  = Set(["block", "component", "function"])
const _IC_ENTRY_KEYS  = Set(["node set", "component", "function"])
const _TW_IC_ENTRY_KEYS = Set([
    "node set", "component", "displacement", "direction", "wave speed",
])

const _OUTPUT_KEYS = Set([
    "velocity", "acceleration", "stress", "deformation gradient",
    "internal variables", "recovery",
])

# ---------------------------------------------------------------------------
# Internal parsers
# ---------------------------------------------------------------------------

# Resolve a file path relative to the input file's directory.
function _resolve(dict, key, basedir)
    val = get(dict, key, nothing)
    val === nothing && error("Missing required input key: \"$key\"")
    path = val::String
    isabspath(path) ? path : joinpath(basedir, path)
end

# Any → Float64 with concrete-typed isa-branches.  YAML.jl returns Int64
# for unquoted integers and Float64 for floats; this dispatch keeps each
# arm inferrable.
@inline function _f64(x)
    x isa Float64 && return x
    x isa Int64   && return Float64(x)
    return Float64(x::Real)
end

# ---- quadrature ----

function _parse_quadrature(dict)
    q_section = get(dict, "quadrature", nothing)
    if q_section === nothing
        return RFE.GaussLegendre, 2
    end
    _validate_keys(q_section, _QUADRATURE_KEYS, "quadrature")
    type_str = lowercase(strip(get(q_section, "type", "gauss legendre")))
    order    = Int(get(q_section, "order", 2))
    if type_str in ("gauss legendre", "gl")
        return RFE.GaussLegendre, order
    elseif type_str in ("gauss lobatto legendre", "gll")
        return RFE.GaussLobattoLegendre, order
    else
        error("Unknown quadrature.type = \"$type_str\". " *
              "Supported: \"gauss legendre\", \"gauss lobatto legendre\".")
    end
end

# ---- material ----

function _parse_material_section(dict)
    model_dict_top = get(dict, "model", nothing)
    model_dict_top === nothing && error("Missing [model] section in input.")

    _validate_keys(model_dict_top, _MODEL_KEYS, "model")

    # `model.type` selects the physics.  It is currently informational — there
    # is only one physics — but it is checked rather than ignored so that
    # `type: thermal` reports "not supported" instead of quietly running a
    # solid-mechanics simulation.
    if haskey(model_dict_top, "type")
        model_type = lowercase(strip(String(model_dict_top["type"])))
        model_type in _MODEL_TYPES || error(
            "Unknown model.type = \"$model_type\". " *
            "Supported: \"solid mechanics\". " *
            "Thermal and coupled physics are not implemented.")
    end

    mat_section = get(model_dict_top, "material", nothing)
    mat_section === nothing && error("Missing [model.material] section in input.")

    # Expect:  blocks: { block_name: model_name }  followed by model-specific keys.
    blocks = get(mat_section, "blocks", nothing)
    blocks === nothing && error("Missing [model.material.blocks] mapping.")
    blocks_dict = blocks::Dict{String,Any}
    isempty(blocks_dict) && error("[model.material.blocks] is empty; assign a material to a block.")

    # One material for the whole mesh.  `blocks` is a mapping for forward
    # compatibility, but only a single entry can be honoured: the constitutive
    # model built here is applied to every element block. Taking `first` of a
    # multi-entry mapping would pick an arbitrary one — `Dict` iteration order
    # is hash order, not file order — and silently apply it everywhere.
    length(blocks_dict) == 1 || error(
        "[model.material.blocks] lists $(length(blocks_dict)) blocks " *
        "($(join(sort(collect(keys(blocks_dict))), ", "))), but Carina supports a " *
        "single material per simulation. The material is applied to the whole mesh; " *
        "per-block materials are not implemented.")

    pair = first(blocks_dict)
    block_name = pair.first
    model_name = pair.second::String

    # Everything in `material` other than `blocks` must be a property dict named
    # after a supported constitutive model.  The legal key set is the model-name
    # list, matched case-insensitively to agree with the `_get_ci` lookup below.
    _validate_keys_ci(mat_section, Set{String}(["blocks", _MODEL_NAMES...]), "model.material")

    # The model-specific sub-dict (e.g.  neohookean: { elastic_modulus: ... })
    model_props = _get_ci(mat_section, model_name, nothing)
    if model_props === nothing
        present = sort(String[k for k in keys(mat_section) if k != "blocks"])
        error("Material model \"$model_name\" is assigned to block \"$block_name\" in " *
              "[model.material.blocks], but [model.material] has no \"$model_name\" " *
              "property dict. Property dicts present: $(join(present, ", ")).")
    end

    cm, density, props_inputs = parse_material(model_name, model_props::Dict{String,Any})
    return block_name, cm, density, props_inputs
end

# ---- time ----

# True if the integrator declared in `dict` never assembles a global matrix
# (currently only central difference).  Used to opt into FEC's matrix-free
# assembler mode at construction time, before the integrator object exists.
function _integrator_is_matrix_free(dict)
    ti_dict = get(dict, "time integrator", nothing)
    ti_dict === nothing && return false
    type_str = lowercase(strip(get(ti_dict, "type", "")))
    return type_str in ("central difference", "centraldifference", "cd")
end

function _parse_times(dict)
    ti_dict = get(dict, "time integrator", nothing)
    ti_dict === nothing && error("Missing `time integrator` section.")
    _validate_keys(ti_dict, _TIME_INTEGRATOR_KEYS, "time integrator")
    t0  = Float64(get(ti_dict, "initial time", 0.0))
    tf  = Float64(ti_dict["final time"])
    dt  = Float64(ti_dict["time step"])
    # FEC.TimeStepper: used by FEC internals (BC evaluation, time queries).
    # Δt will be overwritten each sub-step during subcycling.
    times = FEC.TimeStepper(t0, tf, round(Int, (tf - t0) / dt))
    return t0, tf, dt, times
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
              "minimum time step, maximum time step, " *
              "decrease factor, increase factor.")
    if has_all
        min_dt = Float64(ti_dict["minimum time step"])
        max_dt = Float64(ti_dict["maximum time step"])
        dec    = Float64(ti_dict["decrease factor"])
        inc    = Float64(ti_dict["increase factor"])
        dec >= 1.0 && error("decrease factor must be < 1.0")
        inc <= 1.0 && error("increase factor must be > 1.0")
        min_dt > max_dt && error("minimum time step > maximum time step")
    else
        min_dt = max_dt = dt_nominal
        dec = inc = 1.0
    end
    return min_dt, max_dt, dec, inc
end

# ---- integrator ----

function _parse_integrator(dict, asm, asm_cpu, p_cpu, controller, backend=KA.CPU())
    ti_dict  = get(dict, "time integrator", nothing)
    ti_dict === nothing && error("Missing `time integrator` section.")
    type_str = lowercase(strip(get(ti_dict, "type", "quasi static")))
    dt       = Float64(ti_dict["time step"])

    # Get a template vector from a DirectLinearSolver built on the device assembler.
    # asm is already on the correct device (CPU, ROCm, or CUDA), so its ΔUu
    # is in the right memory space and can be used as a template for allocations.
    fec_ls   = @carina_timed "  DirectLinearSolver (template)" FEC.DirectLinearSolver(asm)
    template = fec_ls.ΔUu

    if type_str in ("quasi static", "quasistatic", "static")
        sol_dict, ls_dict = _read_solver_dicts(dict)
        min_dt, max_dt, dec, inc = _parse_adaptive_stepping(ti_dict, dt)
        init_eq = Bool(get(ti_dict, "initial equilibrium", false))

        make_precond = () -> _compute_stiffness_jacobi_precond(asm_cpu, p_cpu, template)
        make_amg     = () -> _compute_amg_precond(asm_cpu, p_cpu)
        ls = _parse_linear_solver(ls_dict, template, backend, make_precond, make_amg)
        ns = _parse_nonlinear_solver(sol_dict, ls; template=template, make_precond=make_precond)
        _parse_and_store_termination!(sol_dict)
        return QuasiStaticIntegrator(ns, asm, template;
                                      time_step=dt,
                                      min_time_step=min_dt,
                                      max_time_step=max_dt,
                                      decrease_factor=dec,
                                      increase_factor=inc,
                                      initial_equilibrium=init_eq)

    elseif type_str in ("newmark", "newmark-beta", "newmark-beta")
        sol_dict, ls_dict = _read_solver_dicts(dict)
        α_hht = Float64(get(ti_dict, "alpha", 0.0))
        β = α_hht != 0.0 ? (1.0 - α_hht)^2 / 4.0 : Float64(get(ti_dict, "beta",
                                                                get(ti_dict, "β", 0.25)))
        γ = α_hht != 0.0 ? (1.0 - 2.0*α_hht) / 2.0 : Float64(get(ti_dict, "gamma",
                                                                get(ti_dict, "γ", 0.5)))
        min_dt, max_dt, dec, inc = _parse_adaptive_stepping(ti_dict, dt)

        make_precond = () -> _compute_jacobi_precond(β, dt, asm_cpu, p_cpu, template)
        make_amg     = () -> _compute_amg_precond(asm_cpu, p_cpu)
        ls = @carina_timed "  Linear solver (builds precond #1)" _parse_linear_solver(
                 ls_dict, template, backend, make_precond, make_amg)
        ns = @carina_timed "  Nonlinear solver (builds precond #2)" _parse_nonlinear_solver(
                 sol_dict, ls; template=template, make_precond=make_precond)
        _parse_and_store_termination!(sol_dict)
        return @carina_timed "  NewmarkIntegrator ctor" NewmarkIntegrator(ns, asm, β, γ, template;
                                  α_hht=α_hht,
                                  time_step=dt,
                                  min_time_step=min_dt,
                                  max_time_step=max_dt,
                                  decrease_factor=dec,
                                  increase_factor=inc)

    elseif type_str in ("central difference", "centraldifference", "cd")
        γ = Float64(get(ti_dict, "gamma", get(ti_dict, "γ", 0.5)))
        min_dt, max_dt, dec, inc = _parse_adaptive_stepping(ti_dict, dt)
        CFL_val = Float64(get(ti_dict, "CFL", get(ti_dict, "cfl", 0.0)))
        stable_dt_interval = Int(get(ti_dict, "stable time step interval", 0))

        m_lumped = _compute_lumped_mass(asm_cpu, p_cpu, template)

        ig = CentralDifferenceIntegrator(γ, asm, m_lumped;
                                          time_step=dt,
                                          min_time_step=min_dt,
                                          max_time_step=max_dt,
                                          decrease_factor=dec,
                                          increase_factor=inc,
                                          CFL=CFL_val,
                                          stable_dt_interval=stable_dt_interval)

        # Compute initial stable time step estimate
        if CFL_val > 0.0
            stable_dt = _compute_stable_dt(asm_cpu, p_cpu, CFL_val)
            if stable_dt < ig.time_step
                _carina_logf(0, :warning,
                    "Δt = %.2e exceeds stable Δt = %.2e — using stable step.",
                    ig.time_step, stable_dt)
            end
            ig.time_step = min(stable_dt, ig.time_step)
            _carina_logf(0, :setup, "Stable Δt = %.2e (CFL = %.2f)", stable_dt, CFL_val)
        end

        return ig
    else
        error("Unknown time_integrator.type = \"$type_str\". " *
              "Supported: \"quasi_static\", \"newmark\", \"central_difference\".")
    end
end

function _to_device(cpu_vec, template)
    out = similar(template)
    copyto!(out, cpu_vec)
    return out
end

function _compute_jacobi_precond(β, Δt, asm_cpu, p_cpu, ΔUu_template)
    c_M = 1.0 / (β * Δt^2)
    n = length(ΔUu_template)
    U_zeros = zeros(Float64, n)
    # diag(K_eff) ≈ diag(K) + c_M · m_lumped.  Using the partition-of-unity
    # lumped mass as the M-side contribution is a valid SPD Jacobi preconditioner
    # (same scaling class as diag(M)) and avoids the row-sum-of-reduced-M bug
    # that under-counts at boundary-adjacent free DOFs.
    @carina_timed "    assemble_stiffness!" FEC.assemble_stiffness!(asm_cpu, FEC.stiffness, U_zeros, p_cpu)
    K = FEC.stiffness(asm_cpu)
    k_diag = @carina_timed "    extract diag(K)" [K[i, i] for i in 1:size(K, 1)]
    @carina_timed "    assemble_lumped_mass!" FEC.assemble_lumped_mass!(asm_cpu, FEC.lumped_mass, U_zeros, p_cpu)
    m_diag = copy(FEC.lumped_mass(asm_cpu))
    eff_diag = k_diag .+ c_M .* m_diag
    inv_diag_cpu = 1.0 ./ max.(abs.(eff_diag), eps(Float64))
    return JacobiPreconditioner(_to_device(inv_diag_cpu, ΔUu_template))
end

# AMG preconditioner: record the free-DOF index set.  The near-nullspace
# (rigid-body modes) and the hierarchy are built lazily at each assembled
# solve from the CURRENT configuration (see _rigid_body_modes /
# _update_amg_precond_assembled!), so nothing coordinate-dependent is frozen
# here.
function _compute_amg_precond(asm_cpu, p_cpu)
    return AMGPreconditioner(collect(asm_cpu.dof.unknown_dofs))
end

# Jacobi preconditioner for quasi-static: diag(K(U=0))⁻¹.
# Assembles the full stiffness matrix on CPU to extract the true diagonal,
# then transfers to the target device.
function _compute_stiffness_jacobi_precond(asm_cpu, p_cpu, ΔUu_template)
    n = length(ΔUu_template)
    U_zeros = zeros(Float64, n)
    @carina_timed "    assemble_stiffness!" FEC.assemble_stiffness!(asm_cpu, FEC.stiffness, U_zeros, p_cpu)
    K = FEC.stiffness(asm_cpu)
    k_diag = @carina_timed "    extract diag(K)" [K[i, i] for i in 1:size(K, 1)]
    inv_diag_cpu = 1.0 ./ max.(abs.(k_diag), eps(Float64))
    return JacobiPreconditioner(_to_device(inv_diag_cpu, ΔUu_template))
end

# Partition-of-unity row-sum lumped mass via per-element scatter.
# Each free DOF receives ρ · N_a · JxW summed over connected QPs — preserves
# partition of unity even at free DOFs adjacent to constrained ones, unlike
# the legacy `M_red * 1_free` approach.
function _compute_lumped_mass(asm_cpu, p_cpu, ΔUu_template)
    n = length(ΔUu_template)
    U_zeros = zeros(Float64, n)
    FEC.assemble_lumped_mass!(asm_cpu, FEC.lumped_mass, U_zeros, p_cpu)
    m_cpu = copy(FEC.lumped_mass(asm_cpu))
    return _to_device(m_cpu, ΔUu_template)
end

# Build L2 projection recovery data (CPU-only).
function _build_recovery_data(recovery::Symbol, asm_cpu, p_cpu)
    recovery == :none && return NoRecovery()

    if recovery == :lumped
        # Compute geometric lumped "mass" per node: vol_i = Σ_e Σ_q N_i(ξ_q) w_q |J|
        # This is the volume tributary to each node (no density).
        # Assembled directly from the element loop on CPU.
        fspace = FEC.function_space(asm_cpu.dof)
        n_nodes = size(p_cpu.coords, 2)
        vol = zeros(Float64, n_nodes)
        conns = fspace.elem_conns

        for (b, ref_fe) in enumerate(fspace.ref_fes)
            nelem = conns.nelems[b]
            coffset = conns.offsets[b]
            for e in 1:nelem
                conn = FEC.connectivity(ref_fe, conns.data, e, coffset)
                x_el = FEC._element_level_fields_flat(p_cpu.coords, ref_fe, conn)
                nnpe = RFE.num_cell_dofs(ref_fe)
                for q in 1:RFE.num_cell_quadrature_points(ref_fe)
                    interps = FEC._cell_interpolants(ref_fe, q)
                    cell = FEC.map_interpolants(interps, x_el)
                    N = RFE.cell_shape_function_value(ref_fe, q)
                    for i in 1:nnpe
                        vol[conn[i]] += N[i] * cell.JxW
                    end
                end
            end
        end

        inv_vol = zeros(Float64, n_nodes)
        for i in 1:n_nodes
            if vol[i] > 0.0
                inv_vol[i] = 1.0 / vol[i]
            end
        end
        _carina_log(0, :setup, "L2 lumped recovery initialized")
        return LumpedRecovery(inv_vol)

    elseif recovery == :consistent
        # Assemble the scalar consistent mass matrix M_ij = Σ_e Σ_q N_i N_j |J| w_q
        # (geometry-only / density-free, same basis as the lumped volume above)
        # and cache its Cholesky factor.  L2 projection then solves
        # σ_nodal = M⁻¹ b with b_i = Σ_e Σ_q N_i σ(ξ_q) |J| w_q (assembled in io.jl).
        fspace = FEC.function_space(asm_cpu.dof)
        n_nodes = size(p_cpu.coords, 2)
        conns = fspace.elem_conns
        rows = Int[]; cols = Int[]; vals = Float64[]

        for (b, ref_fe) in enumerate(fspace.ref_fes)
            nelem = conns.nelems[b]
            coffset = conns.offsets[b]
            for e in 1:nelem
                conn = FEC.connectivity(ref_fe, conns.data, e, coffset)
                x_el = FEC._element_level_fields_flat(p_cpu.coords, ref_fe, conn)
                nnpe = RFE.num_cell_dofs(ref_fe)
                for q in 1:RFE.num_cell_quadrature_points(ref_fe)
                    interps = FEC._cell_interpolants(ref_fe, q)
                    cell = FEC.map_interpolants(interps, x_el)
                    N = RFE.cell_shape_function_value(ref_fe, q)
                    JxW = cell.JxW
                    for i in 1:nnpe, j in 1:nnpe
                        push!(rows, conn[i]); push!(cols, conn[j])
                        push!(vals, N[i] * N[j] * JxW)
                    end
                end
            end
        end

        M = SparseArrays.sparse(rows, cols, vals, n_nodes, n_nodes)
        M_factor = cholesky(Symmetric(M))
        _carina_log(0, :setup, "L2 consistent recovery initialized")
        return ConsistentRecovery(M_factor)
    else
        return NoRecovery()
    end
end

# Stable time step estimate for explicit dynamics (GPU-native).
# Uses FEC's quadrature assembly to compute per-element characteristic length
# on the device, then takes minimum over all blocks.
function _compute_stable_dt(asm, p, CFL)
    fspace = FEC.function_space(asm.dof)

    # Pre-allocate per-block storage for element char lengths (nq × nelem)
    char_len_storage = []
    for (b, ref_fe) in enumerate(fspace.ref_fes)
        nquad = RFE.num_cell_quadrature_points(ref_fe)
        nelem = FEC.num_elements(fspace, b)
        push!(char_len_storage, zeros(Float64, nquad, nelem))
    end
    char_len_storage = NamedTuple{keys(fspace.ref_fes)}(char_len_storage)

    # Assemble per-element char lengths on device
    U_zeros = zeros(Float64, length(asm.dof.unknown_dofs))
    FEC.assemble_quadrature_quantity!(
        char_len_storage, nothing, asm.dof,
        element_char_length,
        U_zeros, p
    )

    # Min-reduction over all blocks
    stable_dt = Inf
    for (b, (block_physics, block_storage, props)) in enumerate(zip(
        values(p.physics), values(char_len_storage), values(p.properties),
    ))
        # Density lives in the property vector, not on the physics object --
        # `SolidMechanics` no longer carries a `density` field.  CM's interface
        # mandates that the first property of every model is the Lagrangian-frame
        # density (ConstitutiveModels/src/Interface.jl), which is also what the
        # mass-matrix kernels in physics.jl read.  `CM.density` itself is not
        # usable here: it takes the full (props, Z_old, Z_new, Δt, ∇u, θ) call
        # signature, none of which is meaningful at setup time.
        ρ   = props[1]
        M   = CM.p_wave_modulus(block_physics.constitutive_model, props)
        c_p = sqrt(M / ρ)
        h_min = minimum(block_storage)   # GPU-native reduction if on device
        block_dt = CFL * h_min / c_p
        stable_dt = min(stable_dt, block_dt)
    end
    return stable_dt
end

# ---- solver ----

# Accepted spellings of each nonlinear solver.  Shared by `_read_solver_dicts`
# (which gates on them before anything is built) and `_parse_nonlinear_solver`
# (which dispatches on them), so the two cannot drift apart and let a value
# through one gate only to have the other reject it -- or, worse, quietly treat
# it as Newton.  `hessian minimizer` is a Norma-compatibility alias for Newton.
const _NEWTON_TYPES = ("newton", "newton raphson", "newton-raphson", "hessian minimizer")
const _NLCG_TYPES   = ("nonlinear cg", "nlcg", "conjugate gradient")
const _SD_TYPES     = ("steepest descent", "gradient descent", "sd")
const _SOLVER_TYPES = (_NEWTON_TYPES..., _NLCG_TYPES..., _SD_TYPES...)

const _SOLVER_TYPE_HELP =
    "Supported: \"newton\" (aliases \"newton raphson\", \"newton-raphson\", " *
    "\"hessian minimizer\"), \"nonlinear cg\" (aliases \"nlcg\", " *
    "\"conjugate gradient\"), \"steepest descent\" (aliases \"gradient descent\", " *
    "\"sd\"). L-BFGS is a linear solver: set solver.linear solver.type = \"lbfgs\"."

# Validate and return the solver and linear-solver sub-dicts.
# Errors if the required two-level structure is absent.
function _read_solver_dicts(dict)
    haskey(dict, "solver") ||
        error("Missing required [solver] section.")
    sol_dict = dict["solver"]
    _validate_keys(sol_dict, _SOLVER_KEYS, "solver")

    haskey(sol_dict, "type") ||
        error("Missing required solver.type. " * _SOLVER_TYPE_HELP)
    nl_type = lowercase(strip(sol_dict["type"]))
    nl_type in _SOLVER_TYPES ||
        error("Unknown solver.type = \"$(sol_dict["type"])\". " * _SOLVER_TYPE_HELP)

    if nl_type in (_NLCG_TYPES..., _SD_TYPES...)
        # Matrix-free solvers; linear solver section is optional
        ls_dict = get(sol_dict, "linear solver", Dict{String,Any}("type" => "none"))
    else
        haskey(sol_dict, "linear solver") ||
            error("Missing required [solver.linear_solver] section.")
        ls_dict = sol_dict["linear solver"]
        haskey(ls_dict, "type") ||
            error("Missing required solver.linear_solver.type. " *
                  "Supported values: \"direct\", \"iterative\", \"cg\", \"minres\", \"lbfgs\", \"none\".")
    end

    return sol_dict, ls_dict
end

# --------------------------------------------------------------------------- #
# Termination criteria parsing
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Termination criteria parsing
#
# New syntax (preferred):
#   termination:
#     converge when any:          # or "converge when all:"
#       - absolute residual: 1.0e-06
#       - relative residual: 1.0e-10
#     fail when any:              # or "fail when all:"
#       - maximum iterations: 16
#
# Nested groups via any:/all: inside a list:
#   converge when all:
#     - minimum iterations: 0
#     - any:
#         - absolute residual: 1.0e-08
#         - relative residual: 1.0e-12
#
# Legacy syntax (still supported):
#   termination:
#     - type: combo
#       combo: or
#       tests:
#         - type: absolute residual
#           tolerance: 1.0e-06
#     - type: maximum iterations
#       value: 16
# --------------------------------------------------------------------------- #

# Map from test name → constructor taking a single numeric value.
const _TERMINATION_TEST_MAP = Dict{String,Any}(
    "absolute residual"  => v -> AbsResidualTest(Float64(v)),
    "abs_residual"       => v -> AbsResidualTest(Float64(v)),
    "abs_residual"       => v -> AbsResidualTest(Float64(v)),
    "relative residual"  => v -> RelResidualTest(Float64(v)),
    "rel_residual"       => v -> RelResidualTest(Float64(v)),
    "rel_residual"       => v -> RelResidualTest(Float64(v)),
    "absolute update"    => v -> AbsUpdateTest(Float64(v)),
    "abs_update"         => v -> AbsUpdateTest(Float64(v)),
    "abs_update"         => v -> AbsUpdateTest(Float64(v)),
    "relative update"    => v -> RelUpdateTest(Float64(v)),
    "rel_update"         => v -> RelUpdateTest(Float64(v)),
    "rel_update"         => v -> RelUpdateTest(Float64(v)),
    "maximum iterations" => v -> MaxIterationsTest(Int(v)),
    "max iterations"     => v -> MaxIterationsTest(Int(v)),
    "minimum iterations" => v -> MinIterationsTest(Int(v)),
    "min iterations"     => v -> MinIterationsTest(Int(v)),
    "finite value"       => _ -> FiniteValueTest(),
    "nan check"          => _ -> FiniteValueTest(),
    "divergence"         => v -> DivergenceTest(Float64(v)),
    "stagnation"         => v -> StagnationTest(; window=Int(v)),
)

"""
Parse a single item from a termination test list (new compact syntax).
Each item is a single-key Dict, e.g. `{"absolute residual": 1.0e-06}`,
or a nested group `{"any": [...]}` / `{"all": [...]}`.
"""
function _parse_termination_item(entry::Dict)
    length(entry) == 1 || error(
        "Each termination test entry must have exactly one key, got: $(keys(entry))")
    key, val = first(entry)
    lk = lowercase(key)

    # Nested group: any: [...] or all: [...]
    if lk == "any"
        val isa Vector || error("\"any:\" must contain a list.")
        return ComboOrTest(AbstractStatusTest[_parse_termination_item(e) for e in val])
    elseif lk == "all"
        val isa Vector || error("\"all:\" must contain a list.")
        return ComboAndTest(AbstractStatusTest[_parse_termination_item(e) for e in val])
    end

    # Named test: look up constructor
    haskey(_TERMINATION_TEST_MAP, lk) || error("Unknown termination test \"$key\".")
    return _TERMINATION_TEST_MAP[lk](val)
end

"""
Parse a `converge when any/all:` or `fail when any/all:` list into a
composite status test.
"""
function _parse_when_block(items::Vector, operator::String)
    tests = AbstractStatusTest[_parse_termination_item(e) for e in items]
    return operator == "and" ? ComboAndTest(tests) : ComboOrTest(tests)
end

"""
Parse a single termination test entry (legacy syntax).
Returns an AbstractStatusTest.
"""
function _parse_termination_test(entry::Dict)
    test_type = lowercase(strip(get(entry, "type", "")))

    if test_type in ("absolute residual", "abs_residual", "abs_residual")
        tol = Float64(entry["tolerance"])
        return AbsResidualTest(tol)

    elseif test_type in ("relative residual", "rel_residual", "rel_residual")
        tol = Float64(entry["tolerance"])
        return RelResidualTest(tol)

    elseif test_type in ("absolute update", "abs_update", "abs_update")
        tol = Float64(entry["tolerance"])
        return AbsUpdateTest(tol)

    elseif test_type in ("relative update", "rel_update", "rel_update")
        tol = Float64(entry["tolerance"])
        return RelUpdateTest(tol)

    elseif test_type in ("max iterations", "maximum iterations")
        return MaxIterationsTest(Int(entry["value"]))

    elseif test_type in ("min iterations", "minimum iterations")
        return MinIterationsTest(Int(entry["value"]))

    elseif test_type in ("finite value", "nan check")
        return FiniteValueTest()

    elseif test_type in ("divergence",)
        threshold = Float64(get(entry, "threshold", 1e6))
        return DivergenceTest(threshold)

    elseif test_type in ("stagnation",)
        window = Int(get(entry, "window", 5))
        tol    = Float64(get(entry, "tolerance", 0.95))
        return StagnationTest(; window, tol)

    elseif test_type in ("combo", "combination")
        combo_type = lowercase(strip(get(entry, "combo", "or")))
        # Anything other than "and"/"or" used to be read as "or", quietly
        # inverting the meaning of a test group.
        combo_type in ("and", "or") || error(
            "Unknown combo = \"$combo_type\" in termination test. Expected \"and\" or \"or\".")
        subtests = AbstractStatusTest[_parse_termination_test(t)
                                      for t in entry["tests"]]
        return combo_type == "and" ? ComboAndTest(subtests) : ComboOrTest(subtests)

    else
        error("Unknown termination test type \"$test_type\".")
    end
end

# Keys recognized as the new converge/fail syntax.
const _WHEN_KEYS = Dict(
    "converge when any" => ("converge", "or"),
    "converge when all" => ("converge", "and"),
    "fail when any"     => ("fail",     "or"),
    "fail when all"     => ("fail",     "and"),
)

"""
Parse the termination: block from a solver dict.
Returns an AbstractStatusTest.

Supports three formats:
1. New syntax — dict with `converge when any/all:` and `fail when any/all:` keys.
2. Legacy syntax — list of typed test entries with `combo` nesting.
3. Flat keys — `absolute tolerance` / `relative tolerance` (oldest format).
"""
function _parse_termination(sol_dict)
    if !haskey(sol_dict, "termination")
        # Oldest legacy: flat keys
        abs_tol = Float64(get(sol_dict, "absolute tolerance", 1e-10))
        rel_tol = Float64(get(sol_dict, "relative tolerance", 1e-14))
        return ComboOrTest(AbstractStatusTest[
            AbsResidualTest(abs_tol),
            RelResidualTest(rel_tol),
            FiniteValueTest(),
        ])
    end

    entries = sol_dict["termination"]

    # --- New syntax: termination is a Dict with converge/fail keys ----------
    if entries isa Dict
        converge_tests = AbstractStatusTest[]
        fail_tests     = AbstractStatusTest[]
        for (key, val) in entries
            lk = lowercase(key)
            haskey(_WHEN_KEYS, lk) || error(
                "Unknown termination key \"$key\". " *
                "Expected one of: $(join(keys(_WHEN_KEYS), ", ")).")
            role, operator = _WHEN_KEYS[lk]
            val isa Vector || error("\"$key:\" must contain a list.")
            target = role == "converge" ? converge_tests : fail_tests
            push!(target, _parse_when_block(val, operator))
        end
        # Deterministic order: converge groups, then fail groups, then FiniteValue
        top_tests = AbstractStatusTest[converge_tests; fail_tests; FiniteValueTest()]
        return ComboOrTest(top_tests)
    end

    # --- Legacy syntax: termination is a list of typed entries ---------------
    entries isa Vector || error("\"termination:\" must be a mapping or a list.")
    tests = AbstractStatusTest[_parse_termination_test(e) for e in entries]
    push!(tests, FiniteValueTest())
    return ComboOrTest(tests)
end

# --------------------------------------------------------------------------- #

function _parse_linear_solver(ls_dict, template, backend, make_precond::Function,
                               make_amg_precond::Function = () -> error(
                                   "amg preconditioner not available for this integrator."))
    _validate_keys(ls_dict, _LINEAR_SOLVER_KEYS, "linear solver")
    ls_type = lowercase(ls_dict["type"])
    T  = eltype(template)
    n  = length(template)
    S  = typeof(template)

    if ls_type == "direct"
        !(backend isa KA.CPU) && error(
            "solver.linear_solver.type = \"direct\" is CPU-only.")
        return DirectLinearSolver()

    elseif ls_type in ("iterative", "krylov", "minres", "cg", "conjugate gradient")
        # All solid mechanics stiffness matrices are SPD → always use CG.
        itmax     = Int(get(ls_dict, "maximum iterations", 1000))
        rtol      = Float64(get(ls_dict, "tolerance", 1e-8))
        assembled = backend isa KA.CPU

        precond_dict = get(ls_dict, "preconditioner", Dict{String,Any}())
        precond_type = lowercase(strip(get(precond_dict, "type", "none")))
        precond = if precond_type == "jacobi"
            make_precond()
        elseif precond_type in ("ic", "incomplete cholesky", "ildl", "incomplete ldlt")
            ICPreconditioner()
        elseif precond_type in ("chebyshev", "chebyshev polynomial")
            degree = Int(get(precond_dict, "degree", 5))
            mk_s() = (v = similar(template); fill!(v, zero(T)); v)
            ChebyshevPreconditioner(degree, Ref(0.0), Ref(0.0), mk_s(), mk_s(), mk_s())
        elseif precond_type in ("amg", "algebraic multigrid", "multigrid")
            backend isa KA.CPU || error(
                "preconditioner.type = \"amg\" requires the CPU assembled path " *
                "(GPU AMG not yet implemented).")
            make_amg_precond()
        elseif precond_type == "none"
            NoPreconditioner()
        else
            # Do not fall through to NoPreconditioner.  An unpreconditioned CG
            # solve still converges, just far more slowly, so a typo here used
            # to surface as a mysterious performance problem rather than an
            # input error.
            error("Unknown preconditioner.type = \"$precond_type\". " *
                  "Supported: \"jacobi\", \"ic\" (aliases \"incomplete cholesky\", " *
                  "\"ildl\", \"incomplete ldlt\"), \"chebyshev\", " *
                  "\"amg\" (aliases \"algebraic multigrid\", \"multigrid\"), \"none\".")
        end

        workspace = Krylov.CgWorkspace(n, n, S)
        ones_v  = (v = similar(template); fill!(v, one(T)); v)
        scratch = (v = similar(template); fill!(v, zero(T)); v)

        return KrylovLinearSolver(itmax, rtol, assembled, precond,
                                   workspace, ones_v, scratch)

    elseif ls_type == "lbfgs"
        m     = Int(get(ls_dict, "history size", 10))
        precond = make_precond()
        mk()  = (v = similar(template); fill!(v, zero(T)); v)

        S_buf = [mk() for _ in 1:m]
        Y_buf = [mk() for _ in 1:m]
        ρ         = zeros(Float64, m)
        alpha_buf = zeros(Float64, m)
        R_old, d, q, M_d, M_dU = mk(), mk(), mk(), mk(), mk()

        return LBFGSLinearSolver(m, precond, S_buf, Y_buf, ρ, alpha_buf, 0, 0,
                                  R_old, d, q, M_d, M_dU)

    elseif ls_type == "none"
        return NoLinearSolver()

    else
        error("Unknown solver.linear_solver.type = \"$ls_type\". " *
              "Supported values: \"direct\", \"iterative\", \"cg\", \"minres\", \"lbfgs\", \"none\".")
    end
end

function _parse_nonlinear_solver(sol_dict, ls::AbstractLinearSolver;
                                  template=nothing, make_precond=nothing)
    solver_type = lowercase(strip(get(sol_dict, "type", "newton")))
    T = template !== nothing ? eltype(template) : Float64
    mk() = template !== nothing ?
        (v = similar(template); fill!(v, zero(T)); v) : Float64[]
    # Parse termination tree and extract iteration limits from it (or fall back to flat keys)
    term_tree = _parse_termination(sol_dict)
    tree_max = _extract_max_iters(term_tree)
    min_iters = Int(get(sol_dict, "minimum iterations", 0))
    max_iters = tree_max > 0 ? tree_max : Int(get(sol_dict, "maximum iterations", 20))
    abs_tol   = Float64(get(sol_dict, "absolute tolerance", 1e-10))
    rel_tol   = Float64(get(sol_dict, "relative tolerance", 1e-14))
    # Line search parameters.
    # Default ON: implicit Newton (Newmark / quasi-static) at large Δt or load
    # increments can take element-inverting full steps from a far predictor;
    # Armijo backtracking on ½‖R‖² guards against that.  Costs ~one extra
    # residual evaluation per iteration when the full step is already good
    # (α = 1 accepted immediately).  Flows only to NewtonSolver — NLCG /
    # steepest-descent carry their own line search.
    use_ls     = Bool(get(sol_dict, "use line search", true))
    ls_back    = Float64(get(sol_dict, "line search backtrack factor", 0.5))
    ls_dec     = Float64(get(sol_dict, "line search decrease factor", 1e-4))
    ls_max     = Int(get(sol_dict, "line search maximum iterations", 10))

    if solver_type in _NLCG_TYPES
        orth_tol = Float64(get(sol_dict, "orthogonality tolerance", 0.5))
        restart  = Int(get(sol_dict, "restart interval", 0))
        pc_dict  = get(sol_dict, "preconditioner", nothing)
        precond  = if pc_dict !== nothing && make_precond !== nothing
            make_precond()
        else
            NoPreconditioner()
        end
        return NLCGSolver(min_iters, max_iters, abs_tol, abs_tol, rel_tol,
                          ls_back, ls_dec, ls_max,
                          orth_tol, restart, precond,
                          mk(), mk(), mk(), mk())

    elseif solver_type in _SD_TYPES
        pc_dict  = get(sol_dict, "preconditioner", nothing)
        precond  = if pc_dict !== nothing && make_precond !== nothing
            make_precond()
        else
            NoPreconditioner()
        end
        return SteepestDescentSolver(min_iters, max_iters, abs_tol, abs_tol, rel_tol,
                                      ls_back, ls_dec, ls_max,
                                      precond, mk(), mk())

    elseif solver_type in _NEWTON_TYPES
        return NewtonSolver(min_iters, max_iters, abs_tol, abs_tol, rel_tol, ls,
                            use_ls, ls_back, ls_dec, ls_max)
    else
        # Do not fall through to Newton.  Newton is the right default when
        # `type` is absent, but an unrecognised value means the user asked for
        # something else -- and `lbfgs`, the most likely mistake here, is a
        # `linear solver` type, not a nonlinear one.
        error("Unknown solver.type = \"$solver_type\". " * _SOLVER_TYPE_HELP)
    end
end

# After parsing the nonlinear solver, parse and store termination criteria.
function _parse_and_store_termination!(sol_dict)
    _nonlinear_status_test[] = _parse_termination(sol_dict)
end

# Extract max_iters from the termination tree (for the solver loop bound).
function _extract_max_iters(test::MaxIterationsTest)
    return test.max_iters
end
function _extract_max_iters(test::Union{ComboOrTest, ComboAndTest})
    for sub in test.tests
        v = _extract_max_iters(sub)
        v > 0 && return v
    end
    return 0
end
_extract_max_iters(::AbstractStatusTest) = 0

# ---- Dirichlet BCs ----

function _parse_dirichlet_bcs(dict)
    bc_section = get(dict, "boundary conditions", nothing)
    bc_section === nothing && return FEC.DirichletBC[]
    # Validated here rather than in _parse_neumann_bcs so the warning is
    # emitted once; both parsers see the same section.
    _validate_keys_ci(bc_section, _BC_SECTION_KEYS, "boundary conditions")
    entries = _get_ci(bc_section, "dirichlet", FEC.DirichletBC[])
    entries isa Vector || error("[[boundary_conditions.dirichlet]] must be a list.")

    dbcs = FEC.DirichletBC[]
    for (i, entry) in enumerate(entries)
        _validate_keys(entry, _DBC_ENTRY_KEYS, "Dirichlet BC entry $i")
        _require_keys(entry, ("component", "function"), "Dirichlet BC entry $i")
        var_sym  = _component_to_string(entry["component"])
        func     = _make_function(entry["function"])
        # Accept either "side set" or "node set"
        if haskey(entry, "side set")
            push!(dbcs, FEC.DirichletBC(var_sym, func;
                sideset_name = entry["side set"]))
        elseif haskey(entry, "node set")
            push!(dbcs, FEC.DirichletBC(var_sym, func;
                nodeset_name = entry["node set"]))
        else
            error("Dirichlet BC entry must specify side_set or node_set.")
        end
    end
    return dbcs
end

# ---- Neumann BCs ----

function _parse_neumann_bcs(dict)
    bc_section = get(dict, "boundary conditions", nothing)
    bc_section === nothing && return FEC.NeumannBC[], Dict{String,Any}[]
    entries = _get_ci(bc_section, "neumann", nothing)
    entries === nothing && return FEC.NeumannBC[], Dict{String,Any}[]
    entries isa Vector || error("[[boundary_conditions.neumann]] must be a list.")

    nbcs = FEC.NeumannBC[]
    point_load_entries = Dict{String,Any}[]  # deferred: need mesh/dof for PointLoad creation

    for (i, entry) in enumerate(entries)
        _validate_keys(entry, _NBC_ENTRY_KEYS, "Neumann BC entry $i")
        _require_keys(entry, ("component", "function"), "Neumann BC entry $i")

        if haskey(entry, "side set")
            # Surface traction: integrate over side set (FEC handles this).
            # FEC's Neumann convention adds f_val to the residual R (which is
            # F_int − F_ext), so a positive user traction must be negated.
            var_sym  = _component_to_string(entry["component"])
            comp_idx = var_sym === :displ_x ? 1 : var_sym === :displ_y ? 2 : 3
            scalar   = _make_function(entry["function"])
            func = let idx = comp_idx, f = scalar
                (coords, t) -> begin
                    v = -f(coords, t)
                    SVector{3, Float64}(idx == 1 ? v : 0.0,
                                        idx == 2 ? v : 0.0,
                                        idx == 3 ? v : 0.0)
                end
            end
            sset = entry["side set"]
            push!(nbcs, FEC.NeumannBC(var_sym, func, sset))

        elseif haskey(entry, "node set")
            # Point load: apply directly at nodes (deferred to after mesh is built)
            push!(point_load_entries, entry)

        else
            error("Neumann BC entry $i must specify side_set or node_set.")
        end
    end
    return nbcs, point_load_entries
end

# ---------------------------------------------------------------------------
# Mesh-entity name validation
# ---------------------------------------------------------------------------
#
# Every input key that names an element block, node set, or side set is checked
# against the mesh once, as soon as the mesh is available.  Two failure modes
# motivate this:
#
#   * A material assigned to a block that does not exist ran silently.  The
#     block name is only used for the startup log line -- the constitutive model
#     is applied to the whole mesh -- so `blocks: {cubeTypo: neohookean}`
#     produced a correct-looking run with a wrong label.
#   * A mistyped node set or side set reached FEC and raised a bare
#     `KeyError: key "ssz_" not found` from inside `DirichletBCContainer`,
#     naming neither the section nor the entry it came from.
function _validate_mesh_names(dict, mesh, material_block::AbstractString)
    blocks    = collect(keys(mesh.element_conns))
    node_sets = collect(keys(mesh.nodeset_nodes))
    side_sets = collect(keys(mesh.sideset_elems))

    _check_mesh_name(material_block, blocks, "element block", "[model.material.blocks]")

    # Boundary conditions.  Read through `_get_ci` so the casing accepted by
    # `_parse_dirichlet_bcs` is accepted here too.
    bc_section = get(dict, "boundary conditions", nothing)
    if bc_section !== nothing
        for (kind, key) in (("Dirichlet", "dirichlet"), ("Neumann", "neumann"))
            entries = _get_ci(bc_section, key, nothing)
            entries isa Vector || continue
            for (i, entry) in enumerate(entries)
                entry isa AbstractDict || continue
                haskey(entry, "side set") &&
                    _check_mesh_name(entry["side set"], side_sets, "side set",
                                     "$kind BC entry $i")
                haskey(entry, "node set") &&
                    _check_mesh_name(entry["node set"], node_sets, "node set",
                                     "$kind BC entry $i")
            end
        end
    end

    # Body forces: `block` is optional and defaults to the whole mesh.
    bf_section = get(dict, "body forces", nothing)
    if bf_section !== nothing
        entries = bf_section isa Vector ? bf_section : [bf_section]
        for (i, entry) in enumerate(entries)
            entry isa AbstractDict || continue
            block = get(entry, "block", "all")
            String(block) == "all" ||
                _check_mesh_name(block, blocks, "element block", "body force entry $i")
        end
    end

    # Initial conditions: every entry type is node-set based.
    ic_dict = get(dict, "initial conditions", nothing)
    if ic_dict !== nothing
        for kind in ("displacement", "velocity", "traveling wave")
            entries = get(ic_dict, kind, nothing)
            entries isa Vector || continue
            for (i, entry) in enumerate(entries)
                entry isa AbstractDict || continue
                haskey(entry, "node set") &&
                    _check_mesh_name(entry["node set"], node_sets, "node set",
                                     "$kind IC entry $i")
            end
        end
    end

    return nothing
end

function _build_point_loads(entries, mesh, dof)
    isempty(entries) && return PointLoad[]

    inv_map = zeros(Int, length(dof))
    for (i, fd) in enumerate(dof.unknown_dofs)
        inv_map[fd] = i
    end

    loads = PointLoad[]
    for entry in entries
        var_sym  = _component_to_string(entry["component"])
        func     = _make_function(entry["function"])
        nset_sym = entry["node set"]
        bk = FEC.BCBookKeeping(mesh, dof, var_sym; nset_name=nset_sym)
        for (full_dof, node) in zip(bk.dofs, bk.nodes)
            unk_idx = inv_map[full_dof]
            push!(loads, PointLoad(unk_idx, node, func))
        end
    end
    return loads
end

# ---- Body Forces ----

function _parse_body_forces(dict)
    bf_section = get(dict, "body forces", nothing)
    bf_section === nothing && return FEC.Source[]
    entries = bf_section isa Vector ? bf_section : [bf_section]

    bfs = FEC.Source[]
    for (i, entry) in enumerate(entries)
        _validate_keys(entry, _BF_ENTRY_KEYS, "body force entry $i")
        _require_keys(entry, ("component", "function"), "body force entry $i")
        var_sym  = _component_to_string(entry["component"])
        comp_idx = var_sym === :displ_x ? 1 : var_sym === :displ_y ? 2 : 3
        scalar   = _make_function(entry["function"])
        func = let idx = comp_idx, f = scalar
            (coords, t) -> begin
                v = f(coords, t)
                SVector{3, Float64}(idx == 1 ? v : 0.0,
                                    idx == 2 ? v : 0.0,
                                    idx == 3 ? v : 0.0)
            end
        end
        block = get(entry, "block", "all")
        push!(bfs, FEC.Source(var_sym, func, block))
    end
    return bfs
end

# ---- Helpers ----

function _integrator_name(::QuasiStaticIntegrator) "Quasi-static" end
function _integrator_name(::NewmarkIntegrator)      "Newmark" end
function _integrator_name(::CentralDifferenceIntegrator) "Central difference" end

function _solver_description(ig)
    ig_name = _integrator_name(ig)
    ns = nonlinear_solver(ig)
    if ns isa NewtonSolver
        ls = ns.linear_solver
        ls_name = if ls isa DirectLinearSolver
            "direct"
        elseif ls isa KrylovLinearSolver
            "CG"
        elseif ls isa LBFGSLinearSolver
            "L-BFGS"
        else
            string(typeof(ls))
        end
        return "$ig_name, Newton + $ls_name"
    elseif ns isa NLCGSolver
        return "$ig_name, nonlinear CG"
    elseif ns isa SteepestDescentSolver
        return "$ig_name, steepest descent (energy-based)"
    else
        return "$ig_name"
    end
end

function _solver_description(::CentralDifferenceIntegrator)
    return "Central difference (explicit)"
end

# Map "x" / "y" / "z" → :displ_x / :displ_y / :displ_z
function _component_to_string(comp::String)
    c = lowercase(strip(comp))
    c == "x" && return "displ_x"
    c == "y" && return "displ_y"
    c == "z" && return "displ_z"
    error("Unknown component \"$comp\". Expected x, y, or z.")
end

# Variable namespace used by every YAML expression Carina builds.
# Convention (matches FEC.Expressions.differentiate): time is the LAST
# variable, so the time-derivative index is `num_vars == 4`.
const _CARINA_EXPR_VARS = ["x", "y", "z", "t"]

# Inline `name=value;` bindings into a final expression so FEC's parser
# (which does not understand assignment) sees only the substituted form.
#
# Example
# -------
# `"a=1.0e-3; tc=2.5e-4; a*exp(-(t-tc)^2/...)"` becomes
# `"(1.0e-3)*exp(-(t-(2.5e-4))^2/...)"`.
#
# Word-boundary regex keeps `tc` from matching the leading `t` of e.g.
# `tc*t`.  Each binding's RHS is itself expanded against earlier bindings
# so `tau=tc/2` is legal.
function _inline_expr_bindings(expr_str::String)
    parts = strip.(split(expr_str, ';'))
    isempty(parts) && return expr_str
    length(parts) == 1 && return String(parts[1])
    bindings = Pair{String,String}[]
    for p in @view parts[1:end-1]
        isempty(p) && continue
        eq = findfirst('=', p)
        eq === nothing && error("Expected `name=value` in expression binding fragment: \"$p\"")
        name  = strip(p[1:prevind(p, eq)])
        value = strip(p[nextind(p, eq):end])
        # Expand earlier bindings inside this value.
        for (prev_name, prev_value) in bindings
            value = replace(value, Regex("\\b" * prev_name * "\\b") => "(" * prev_value * ")")
        end
        push!(bindings, String(name) => String(value))
    end
    expr = String(parts[end])
    for (name, value) in bindings
        expr = replace(expr, Regex("\\b" * name * "\\b") => "(" * value * ")")
    end
    return expr
end

# Turn a YAML expression string into an isbits scalar function over
# function over (x, y, z, t).  FEC's flat-form ScalarExpressionFunction
# satisfies both KA kernel-argument requirements (isbits) and juliac
# trim-mode constraints (no @eval, no runtime-defined methods), so a
# single value type replaces the per-expression @eval closures used
# previously — and the world-age `Base.invokelatest` shielding the
# downstream call sites is no longer needed.
function _make_function(expr_str::String)
    return FEC.Expressions.ScalarExpressionFunction{Float64}(
        _inline_expr_bindings(expr_str), _CARINA_EXPR_VARS
    )
end

# ---- initial conditions ----

# Validate the keys of the `initial conditions` section itself.
#
# Called from `create_simulation` rather than from any one of the three IC
# parsers: all three read the same section, so validating inside them would
# either warn three times or make the warning depend on which parser happens to
# run first.  A misspelled sub-key — `velocities:` for `velocity:` — used to
# yield an empty list, so the simulation started from rest and looked like a
# physics result rather than an input error.
function _validate_ic_section(dict)
    ic_dict = get(dict, "initial conditions", nothing)
    ic_dict === nothing && return nothing
    _validate_keys(ic_dict, _IC_SECTION_KEYS, "initial conditions")
    return nothing
end

# Shared entry validation for the `displacement` and `velocity` IC lists.
# Both are consumed by `_apply_initial_*_ics!`, which indexes `entry[...]`
# directly; without this a typo surfaces as a bare KeyError from initialization.
function _validate_ic_entries(entries, kind::String)
    for (i, entry) in enumerate(entries)
        entry isa AbstractDict || error(
            "initial conditions.$kind entry $i must be a mapping with keys: " *
            "node set, component, function.")
        _validate_keys(entry, _IC_ENTRY_KEYS, "$kind IC entry $i")
        _require_keys(entry, ("node set", "component", "function"), "$kind IC entry $i")
    end
    return entries
end

function _parse_displacement_ics(dict)
    ic_dict = get(dict, "initial conditions", nothing)
    ic_dict === nothing && return Any[]
    disp_ics = get(ic_dict, "displacement", Any[])
    disp_ics isa Vector || error("initial conditions.displacement must be a list.")
    return _validate_ic_entries(disp_ics, "displacement")
end

function _parse_velocity_ics(dict)
    ic_dict = get(dict, "initial conditions", nothing)
    ic_dict === nothing && return Any[]
    vel_ics = get(ic_dict, "velocity", Any[])
    vel_ics isa Vector || error("initial conditions.velocity must be a list.")
    return _validate_ic_entries(vel_ics, "velocity")
end

# Traveling-wave IC: the user supplies the initial displacement profile
# u₀(x, y, z), a propagation axis ("x" / "y" / "z"), and a wave speed c.
# The kinematic relation `u(x, t) = f(s − c t)` (with `s` the coordinate
# along the propagation axis) implies `v(x, 0) = −c · ∂u₀/∂s`; Carina
# derives that velocity field symbolically via `FEC.Expressions.differentiate`
# so the user never writes a derivative by hand.  `wave_speed` is signed —
# the sign picks the direction of travel along the axis.
function _parse_traveling_wave_ics(dict)
    ic_dict = get(dict, "initial conditions", nothing)
    ic_dict === nothing && return Any[]
    tw_ics = get(ic_dict, "traveling wave", Any[])
    tw_ics isa Vector || error("initial_conditions.traveling_wave must be a list.")
    for (i, entry) in enumerate(tw_ics)
        _validate_keys(entry, _TW_IC_ENTRY_KEYS, "traveling_wave IC entry $i")
        for key in ("node set", "component", "displacement", "direction", "wave speed")
            haskey(entry, key) || error(
                "traveling_wave IC entry $i missing required key \"$key\". " *
                "Need: node_set, component, displacement, direction, wave_speed."
            )
        end
        dir = lowercase(strip(String(entry["direction"])))
        dir in ("x", "y", "z") || error(
            "traveling_wave IC entry $i: direction \"$dir\" must be x, y, or z."
        )
    end
    return tw_ics
end

# Map a direction label ("x"/"y"/"z") to its variable index in the
# Carina expression namespace `["x", "y", "z", "t"]`.
function _direction_to_idx(dir::String)
    d = lowercase(strip(dir))
    d == "x" && return 1
    d == "y" && return 2
    d == "z" && return 3
    error("Unknown direction \"$dir\". Expected x, y, or z.")
end
