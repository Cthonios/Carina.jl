# YAML parsing and factory helpers for simulation construction.
#
# All functions that translate YAML dict entries into Julia objects
# (integrators, solvers, BCs, materials, etc.) live here.

import FiniteElementContainers as FEC
import ConstitutiveModels as CM
import Krylov
import YAML
using StaticArrays
using LinearAlgebra

# ---------------------------------------------------------------------------
# YAML key validation
# ---------------------------------------------------------------------------
#
# Each YAML section has a set of known keys. Unknown keys are flagged with
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

const _DBC_ENTRY_KEYS = Set(["side set", "node set", "component", "function"])
const _NBC_ENTRY_KEYS = Set(["side set", "node set", "component", "function"])
const _BF_ENTRY_KEYS  = Set(["block", "component", "function"])
const _IC_ENTRY_KEYS  = Set(["node set", "component", "function"])

const _OUTPUT_KEYS = Set([
    "velocity", "acceleration", "stress", "deformation gradient",
    "internal variables", "recovery",
])

# ---------------------------------------------------------------------------
# Internal parsers
# ---------------------------------------------------------------------------

# Resolve a file path relative to the YAML's directory.
function _resolve(dict, key, basedir)
    val = get(dict, key, nothing)
    val === nothing && error("Missing required YAML key: \"$key\"")
    isabspath(val) ? val : joinpath(basedir, val)
end

# ---- quadrature ----

function _parse_quadrature(dict)
    q_section = get(dict, "quadrature", nothing)
    if q_section === nothing
        return RFE.GaussLegendre, 2
    end
    type_str = lowercase(get(q_section, "type", "gauss legendre"))
    order    = Int(get(q_section, "order", 2))
    if type_str in ("gauss legendre", "gl")
        return RFE.GaussLegendre, order
    elseif type_str in ("gauss lobatto legendre", "gll")
        return RFE.GaussLobattoLegendre, order
    else
        error("Unknown \"quadrature: type: $type_str\". " *
              "Supported: \"gauss legendre\", \"gauss lobatto legendre\".")
    end
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
    dt       = Float64(ti_dict["time step"])

    # Get a template vector from a DirectLinearSolver built on the device assembler.
    # asm is already on the correct device (CPU, ROCm, or CUDA), so its ΔUu
    # is in the right memory space and can be used as a template for allocations.
    fec_ls   = FEC.DirectLinearSolver(asm)
    template = fec_ls.ΔUu

    if type_str in ("quasi static", "quasistatic", "static")
        sol_dict, ls_dict = _read_solver_dicts(dict)
        min_dt, max_dt, dec, inc = _parse_adaptive_stepping(ti_dict, dt)
        init_eq = Bool(get(ti_dict, "initial equilibrium", false))

        make_precond = () -> _compute_stiffness_jacobi_precond(asm_cpu, p_cpu, template)
        ls = _parse_linear_solver(ls_dict, template, device, make_precond)
        ns = _parse_nonlinear_solver(sol_dict, ls; template=template, make_precond=make_precond)
        _parse_and_store_termination!(sol_dict)
        return QuasiStaticIntegrator(ns, asm, template;
                                      time_step=dt,
                                      min_time_step=min_dt,
                                      max_time_step=max_dt,
                                      decrease_factor=dec,
                                      increase_factor=inc,
                                      initial_equilibrium=init_eq)

    elseif type_str in ("newmark", "newmark-beta", "newmark beta")
        sol_dict, ls_dict = _read_solver_dicts(dict)
        α_hht = Float64(get(ti_dict, "alpha", 0.0))
        β = α_hht != 0.0 ? (1.0 - α_hht)^2 / 4.0 : Float64(get(ti_dict, "beta",  0.25))
        γ = α_hht != 0.0 ? (1.0 - 2.0*α_hht) / 2.0 : Float64(get(ti_dict, "gamma", 0.5))
        min_dt, max_dt, dec, inc = _parse_adaptive_stepping(ti_dict, dt)

        make_precond = () -> _compute_jacobi_precond(β, dt, asm_cpu, p_cpu, template)
        ls = _parse_linear_solver(ls_dict, template, device, make_precond)
        ns = _parse_nonlinear_solver(sol_dict, ls; template=template, make_precond=make_precond)
        _parse_and_store_termination!(sol_dict)
        return NewmarkIntegrator(ns, asm, β, γ, template;
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
        error("Unknown time integrator type: \"$type_str\". " *
              "Supported: \"quasi static\", \"Newmark\", \"central difference\".")
    end
end

# Compute the Jacobi (diagonal) preconditioner for the Newmark effective
# stiffness K + c_M·M using the mass-only approximation d_ii ≈ c_M·M_ii.
#
# The diagonal is computed on CPU (asm_cpu, p_cpu) and then transferred to
# the target device (ΔUu_template may be a GPU array).
# Assemble diag(A·ones) on CPU via matrix-action, copy result to target device.
# Common boilerplate for Jacobi preconditioner and lumped mass computation.
function _assemble_diag_cpu(action, asm_cpu, p_cpu, template)
    n    = length(template)
    ones_v  = ones(Float64, n)
    U_zeros = zeros(Float64, n)
    FEC.assemble_matrix_action!(asm_cpu, action, U_zeros, ones_v, p_cpu)
    return copy(FEC.hvp(asm_cpu, ones_v))
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
    # diag(K_eff) = diag(K) + c_M · diag(M)
    FEC.assemble_stiffness!(asm_cpu, FEC.stiffness, U_zeros, p_cpu)
    K = FEC.stiffness(asm_cpu)
    k_diag = [K[i, i] for i in 1:size(K, 1)]
    m_diag = _assemble_diag_cpu(FEC.mass, asm_cpu, p_cpu, ΔUu_template)
    eff_diag = k_diag .+ c_M .* m_diag
    inv_diag_cpu = 1.0 ./ max.(abs.(eff_diag), eps(Float64))
    return JacobiPreconditioner(_to_device(inv_diag_cpu, ΔUu_template))
end

# Jacobi preconditioner for quasi-static: diag(K(U=0))⁻¹.
# Assembles the full stiffness matrix on CPU to extract the true diagonal,
# then transfers to the target device.
function _compute_stiffness_jacobi_precond(asm_cpu, p_cpu, ΔUu_template)
    n = length(ΔUu_template)
    U_zeros = zeros(Float64, n)
    FEC.assemble_stiffness!(asm_cpu, FEC.stiffness, U_zeros, p_cpu)
    K = FEC.stiffness(asm_cpu)
    k_diag = [K[i, i] for i in 1:size(K, 1)]
    inv_diag_cpu = 1.0 ./ max.(abs.(k_diag), eps(Float64))
    return JacobiPreconditioner(_to_device(inv_diag_cpu, ΔUu_template))
end

# Lumped mass vector (diagonal row sums of consistent mass matrix).
function _compute_lumped_mass(asm_cpu, p_cpu, ΔUu_template)
    m_cpu = _assemble_diag_cpu(FEC.mass, asm_cpu, p_cpu, ΔUu_template)
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
        # TODO: assemble and factor scalar mass matrix
        _carina_log(0, :warning, "Consistent L2 recovery not yet implemented, falling back to lumped")
        return _build_recovery_data(:lumped, asm_cpu, p_cpu)
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
        ρ   = block_physics.density
        M   = CM.p_wave_modulus(block_physics.constitutive_model, props)
        c_p = sqrt(M / ρ)
        h_min = minimum(block_storage)   # GPU-native reduction if on device
        block_dt = CFL * h_min / c_p
        stable_dt = min(stable_dt, block_dt)
    end
    return stable_dt
end

# ---- solver ----

# Validate and return the solver and linear-solver sub-dicts.
# Errors if the required two-level structure is absent.
function _read_solver_dicts(dict)
    haskey(dict, "solver") ||
        error("Missing required \"solver:\" section.")
    sol_dict = dict["solver"]
    _validate_keys(sol_dict, _SOLVER_KEYS, "solver")

    haskey(sol_dict, "type") ||
        error("Missing required \"solver: type:\". " *
              "Supported values: \"newton\", \"hessian minimizer\".")
    nl_type = lowercase(sol_dict["type"])
    nl_type in ("newton", "hessian minimizer", "nonlinear cg", "nlcg", "conjugate gradient",
                 "steepest descent", "gradient descent", "sd") ||
        error("Unknown \"solver: type: $(sol_dict["type"])\". " *
              "Supported: \"newton\", \"nonlinear cg\", \"steepest descent\".")

    if nl_type in ("nonlinear cg", "nlcg", "conjugate gradient",
                    "steepest descent", "gradient descent", "sd")
        # Matrix-free solvers; linear solver section is optional
        ls_dict = get(sol_dict, "linear solver", Dict{String,Any}("type" => "none"))
    else
        haskey(sol_dict, "linear solver") ||
            error("Missing required \"solver: linear solver:\" section.")
        ls_dict = sol_dict["linear solver"]
        haskey(ls_dict, "type") ||
            error("Missing required \"solver: linear solver: type:\". " *
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

# Map from YAML test name → constructor taking a single numeric value.
const _TERMINATION_TEST_MAP = Dict{String,Any}(
    "absolute residual"  => v -> AbsResidualTest(Float64(v)),
    "abs residual"       => v -> AbsResidualTest(Float64(v)),
    "abs_residual"       => v -> AbsResidualTest(Float64(v)),
    "relative residual"  => v -> RelResidualTest(Float64(v)),
    "rel residual"       => v -> RelResidualTest(Float64(v)),
    "rel_residual"       => v -> RelResidualTest(Float64(v)),
    "absolute update"    => v -> AbsUpdateTest(Float64(v)),
    "abs update"         => v -> AbsUpdateTest(Float64(v)),
    "abs_update"         => v -> AbsUpdateTest(Float64(v)),
    "relative update"    => v -> RelUpdateTest(Float64(v)),
    "rel update"         => v -> RelUpdateTest(Float64(v)),
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
Parse a single termination test entry from YAML (legacy syntax).
Returns an AbstractStatusTest.
"""
function _parse_termination_test(entry::Dict)
    test_type = lowercase(get(entry, "type", ""))

    if test_type in ("absolute residual", "abs residual", "abs_residual")
        tol = Float64(entry["tolerance"])
        return AbsResidualTest(tol)

    elseif test_type in ("relative residual", "rel residual", "rel_residual")
        tol = Float64(entry["tolerance"])
        return RelResidualTest(tol)

    elseif test_type in ("absolute update", "abs update", "abs_update")
        tol = Float64(entry["tolerance"])
        return AbsUpdateTest(tol)

    elseif test_type in ("relative update", "rel update", "rel_update")
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
        combo_type = lowercase(get(entry, "combo", "or"))
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

function _parse_linear_solver(ls_dict, template, device, make_precond::Function)
    _validate_keys(ls_dict, _LINEAR_SOLVER_KEYS, "linear solver")
    ls_type = lowercase(ls_dict["type"])
    T  = eltype(template)
    n  = length(template)
    S  = typeof(template)

    if ls_type == "direct"
        device != :cpu && error(
            "\"solver: linear solver: type: direct\" is CPU-only.")
        return DirectLinearSolver()

    elseif ls_type in ("iterative", "krylov", "minres", "cg", "conjugate gradient")
        # All solid mechanics stiffness matrices are SPD → always use CG.
        itmax     = Int(get(ls_dict, "maximum iterations", 1000))
        rtol      = Float64(get(ls_dict, "tolerance", 1e-8))
        assembled = (device == :cpu)

        precond_dict = get(ls_dict, "preconditioner", Dict{String,Any}())
        precond_type = lowercase(get(precond_dict, "type", "none"))
        precond = if precond_type == "jacobi"
            make_precond()
        elseif precond_type in ("ic", "incomplete cholesky", "ildl", "incomplete ldlt")
            ICPreconditioner()
        elseif precond_type in ("chebyshev", "chebyshev polynomial")
            degree = Int(get(precond_dict, "degree", 5))
            mk_s() = (v = similar(template); fill!(v, zero(T)); v)
            ChebyshevPreconditioner(degree, Ref(0.0), Ref(0.0), mk_s(), mk_s(), mk_s())
        else
            NoPreconditioner()
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
        error("Unknown \"solver: linear solver: type: $ls_type\". " *
              "Supported values: \"direct\", \"iterative\", \"cg\", \"minres\", \"lbfgs\", \"none\".")
    end
end

function _parse_nonlinear_solver(sol_dict, ls::AbstractLinearSolver;
                                  template=nothing, make_precond=nothing)
    solver_type = lowercase(get(sol_dict, "type", "newton"))
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
    # Line search parameters
    use_ls     = Bool(get(sol_dict, "use line search", false))
    ls_back    = Float64(get(sol_dict, "line search backtrack factor", 0.5))
    ls_dec     = Float64(get(sol_dict, "line search decrease factor", 1e-4))
    ls_max     = Int(get(sol_dict, "line search maximum iterations", 10))

    if solver_type in ("nonlinear cg", "nlcg", "conjugate gradient")
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

    elseif solver_type in ("steepest descent", "gradient descent", "sd")
        pc_dict  = get(sol_dict, "preconditioner", nothing)
        precond  = if pc_dict !== nothing && make_precond !== nothing
            make_precond()
        else
            NoPreconditioner()
        end
        return SteepestDescentSolver(min_iters, max_iters, abs_tol, abs_tol, rel_tol,
                                      ls_back, ls_dec, ls_max,
                                      precond, mk(), mk())
    else
        return NewtonSolver(min_iters, max_iters, abs_tol, abs_tol, rel_tol, ls,
                            use_ls, ls_back, ls_dec, ls_max)
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
    entries = get(bc_section, "Dirichlet", FEC.DirichletBC[])
    entries isa Vector || error("\"Dirichlet:\" must be a list.")

    dbcs = FEC.DirichletBC[]
    for (i, entry) in enumerate(entries)
        _validate_keys(entry, _DBC_ENTRY_KEYS, "Dirichlet BC entry $i")
        var_sym  = _component_to_symbol(entry["component"])
        func     = _make_function(entry["function"])
        # Accept either "side set" or "node set"
        if haskey(entry, "side set")
            push!(dbcs, FEC.DirichletBC(var_sym, func;
                sideset_name = Symbol(entry["side set"])))
        elseif haskey(entry, "node set")
            push!(dbcs, FEC.DirichletBC(var_sym, func;
                nodeset_name = Symbol(entry["node set"])))
        else
            error("Dirichlet BC entry must specify \"side set\" or \"node set\".")
        end
    end
    return dbcs
end

# ---- Neumann BCs ----

function _parse_neumann_bcs(dict)
    bc_section = get(dict, "boundary conditions", nothing)
    bc_section === nothing && return FEC.NeumannBC[], Dict{String,Any}[]
    entries = get(bc_section, "Neumann", nothing)
    entries === nothing && return FEC.NeumannBC[], Dict{String,Any}[]
    entries isa Vector || error("\"Neumann:\" must be a list.")

    nbcs = FEC.NeumannBC[]
    point_load_entries = Dict{String,Any}[]  # deferred: need mesh/dof for PointLoad creation

    for (i, entry) in enumerate(entries)
        _validate_keys(entry, _NBC_ENTRY_KEYS, "Neumann BC entry $i")

        if haskey(entry, "side set")
            # Surface traction: integrate over side set (FEC handles this).
            # FEC's Neumann convention adds f_val to the residual R (which is
            # F_int − F_ext), so a positive user traction must be negated.
            var_sym  = _component_to_symbol(entry["component"])
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
            sset = Symbol(entry["side set"])
            push!(nbcs, FEC.NeumannBC(var_sym, func, sset))

        elseif haskey(entry, "node set")
            # Point load: apply directly at nodes (deferred to after mesh is built)
            push!(point_load_entries, entry)

        else
            error("Neumann BC entry $i must specify \"side set\" or \"node set\".")
        end
    end
    return nbcs, point_load_entries
end

function _build_point_loads(entries, mesh, dof)
    isempty(entries) && return PointLoad[]

    inv_map = zeros(Int, length(dof))
    for (i, fd) in enumerate(dof.unknown_dofs)
        inv_map[fd] = i
    end

    loads = PointLoad[]
    for entry in entries
        var_sym  = _component_to_symbol(entry["component"])
        func     = _make_function(entry["function"])
        nset_sym = Symbol(entry["node set"])
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
        var_sym  = _component_to_symbol(entry["component"])
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
        block = Symbol(get(entry, "block", "all"))
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
# Uses @eval to create a compiled anonymous function that is isbits
# (GPU-compatible as a kernel argument). The function is defined at the
# current top-level scope, so callers in the time loop must use
# Base.invokelatest to see it.
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

function _parse_displacement_ics(dict)
    ic_dict = get(dict, "initial conditions", nothing)
    ic_dict === nothing && return Any[]
    disp_ics = get(ic_dict, "displacement", Any[])
    disp_ics isa Vector || error("\"initial conditions: displacement:\" must be a list.")
    return disp_ics
end

function _parse_velocity_ics(dict)
    ic_dict = get(dict, "initial conditions", nothing)
    ic_dict === nothing && return Any[]
    vel_ics = get(ic_dict, "velocity", Any[])
    vel_ics isa Vector || error("\"initial conditions: velocity:\" must be a list.")
    return vel_ics
end
