# Unified termination criteria for nonlinear and linear solvers.
#
# Design follows NOX's StatusTest pattern (Trilinos):
#   - Abstract base type with check() returning a status enum
#   - Concrete tests for residual, update, iteration, stagnation, divergence
#   - AND/OR composite for complex criteria trees
#   - Push-based ModelFlag for constitutive model failure signaling
#
# Unique to Carina: same framework used for both nonlinear (Newton, NLCG, SD,
# L-BFGS) and linear (CG) solvers.

# --------------------------------------------------------------------------- #
# Status enum
# --------------------------------------------------------------------------- #

@enum SolverStatus begin
    Unconverged = 0
    Converged   = 1
    Failed      = -1
end

# --------------------------------------------------------------------------- #
# Solver info — snapshot of solver state passed to status tests
# --------------------------------------------------------------------------- #

"""
    SolverInfo

Snapshot of solver state for status test evaluation.  Populated by the
solver loop and passed to `check()`.  All fields are optional (set to
NaN or 0 if not applicable).
"""
struct SolverInfo
    iteration     ::Int       # current iteration index (1-based)
    norm_R        ::Float64   # current residual norm ||R||
    norm_R_init   ::Float64   # initial residual norm ||R₀|| (iter 0)
    norm_R_prev   ::Float64   # residual norm at previous iteration
    norm_step     ::Float64   # update norm ||ΔU|| or ||Δx||
    norm_solution ::Float64   # current solution norm ||U|| or ||x||
end

# --------------------------------------------------------------------------- #
# Abstract base type
# --------------------------------------------------------------------------- #

abstract type AbstractStatusTest end

"""
    check(test, info) -> SolverStatus

Evaluate the status test against the current solver state.
Returns `Converged`, `Unconverged`, or `Failed`.
"""
function check end

# --------------------------------------------------------------------------- #
# Convergence tests
# --------------------------------------------------------------------------- #

"""Converged when ||R|| < tol."""
struct AbsResidualTest <: AbstractStatusTest
    tol::Float64
end
check(t::AbsResidualTest, info::SolverInfo) =
    info.norm_R < t.tol ? Converged : Unconverged

"""Converged when ||R|| / ||R₀|| < tol."""
struct RelResidualTest <: AbstractStatusTest
    tol::Float64
end
check(t::RelResidualTest, info::SolverInfo) =
    (info.norm_R_init > 0 && info.norm_R / info.norm_R_init < t.tol) ? Converged : Unconverged

"""Converged when ||ΔU|| < tol."""
struct AbsUpdateTest <: AbstractStatusTest
    tol::Float64
end
check(t::AbsUpdateTest, info::SolverInfo) =
    info.norm_step < t.tol ? Converged : Unconverged

"""Converged when ||ΔU|| / ||U|| < tol."""
struct RelUpdateTest <: AbstractStatusTest
    tol::Float64
end
check(t::RelUpdateTest, info::SolverInfo) =
    (info.norm_solution > 0 && info.norm_step / info.norm_solution < t.tol) ? Converged : Unconverged

# --------------------------------------------------------------------------- #
# Failure tests
# --------------------------------------------------------------------------- #

"""Failed when iteration count reaches max."""
struct MaxIterationsTest <: AbstractStatusTest
    max_iters::Int
end
check(t::MaxIterationsTest, info::SolverInfo) =
    info.iteration >= t.max_iters ? Failed : Unconverged

"""Failed when ||R|| contains NaN or Inf."""
struct FiniteValueTest <: AbstractStatusTest end
check(::FiniteValueTest, info::SolverInfo) =
    isfinite(info.norm_R) ? Unconverged : Failed

"""Failed when ||R_k|| > threshold * ||R₀|| (divergence)."""
struct DivergenceTest <: AbstractStatusTest
    threshold::Float64   # e.g., 1e6 — fail if residual grows by this factor
end
check(t::DivergenceTest, info::SolverInfo) =
    (info.norm_R_init > 0 && info.norm_R > t.threshold * info.norm_R_init) ? Failed : Unconverged

"""
Failed when ||R|| hasn't decreased by more than `tol` fraction over
the last `window` iterations (stagnation detection).
"""
mutable struct StagnationTest <: AbstractStatusTest
    window   ::Int
    tol      ::Float64       # minimum required reduction per window (e.g., 0.95)
    history  ::Vector{Float64}
    count    ::Int           # consecutive stagnation count
end
StagnationTest(; window::Int=5, tol::Float64=0.95) =
    StagnationTest(window, tol, Float64[], 0)

function check(t::StagnationTest, info::SolverInfo)
    push!(t.history, info.norm_R)
    length(t.history) <= t.window && return Unconverged
    old = t.history[end - t.window]
    ratio = info.norm_R / max(old, eps(Float64))
    if ratio > t.tol
        t.count += 1
    else
        t.count = 0
    end
    return t.count >= t.window ? Failed : Unconverged
end

function reset!(t::StagnationTest)
    empty!(t.history)
    t.count = 0
end

"""
Push-based failure flag.  Constitutive models or other external code
can set `flag.status = Failed` and `flag.message = "..."` to signal
failure to the solver.  The solver's status test tree includes this
flag and checks it each iteration.
"""
mutable struct ModelFlagTest <: AbstractStatusTest
    status  ::SolverStatus
    message ::String
end
ModelFlagTest() = ModelFlagTest(Unconverged, "")

check(t::ModelFlagTest, ::SolverInfo) = t.status

function reset!(t::ModelFlagTest)
    t.status  = Unconverged
    t.message = ""
end

# --------------------------------------------------------------------------- #
# Composite tests (AND / OR)
# --------------------------------------------------------------------------- #

"""
AND composite: Converged only when ALL sub-tests return Converged.
If any sub-test returns Failed, the composite returns Failed.
All sub-tests are always evaluated (no short-circuit) for diagnostics.
"""
struct ComboAndTest <: AbstractStatusTest
    tests::Vector{AbstractStatusTest}
end

function check(t::ComboAndTest, info::SolverInfo)
    all_converged = true
    for sub in t.tests
        s = check(sub, info)
        s == Failed && return Failed
        s != Converged && (all_converged = false)
    end
    return all_converged ? Converged : Unconverged
end

"""
OR composite: Converged if ANY sub-test returns Converged.
Failed if ANY sub-test returns Failed and none return Converged.
All sub-tests are always evaluated (no short-circuit) for diagnostics.
"""
struct ComboOrTest <: AbstractStatusTest
    tests::Vector{AbstractStatusTest}
end

function check(t::ComboOrTest, info::SolverInfo)
    any_converged = false
    any_failed    = false
    for sub in t.tests
        s = check(sub, info)
        s == Converged && (any_converged = true)
        s == Failed    && (any_failed = true)
    end
    any_converged && return Converged
    any_failed    && return Failed
    return Unconverged
end

# --------------------------------------------------------------------------- #
# Reset for mutable tests (stagnation, model flag)
# --------------------------------------------------------------------------- #

reset!(::AbstractStatusTest) = nothing  # no-op default

function reset!(t::ComboAndTest)
    for sub in t.tests; reset!(sub); end
end
function reset!(t::ComboOrTest)
    for sub in t.tests; reset!(sub); end
end

# --------------------------------------------------------------------------- #
# Default convergence criterion (backward-compatible)
#
# Builds the standard test tree from solver parameters:
#   OR(
#     AND(AbsResidual, RelResidual),   # convergence
#     MaxIterations,                    # failure
#     FiniteValue                       # failure
#   )
# --------------------------------------------------------------------------- #

function default_nonlinear_status_test(;
    abs_tol::Float64  = 1e-10,
    rel_tol::Float64  = 1e-14,
    max_iters::Int    = 20,
)
    convergence = ComboAndTest([
        AbsResidualTest(abs_tol),
        RelResidualTest(rel_tol),
    ])
    return ComboOrTest([
        convergence,
        MaxIterationsTest(max_iters),
        FiniteValueTest(),
    ])
end

function default_linear_status_test(;
    rtol::Float64  = 1e-8,
    max_iters::Int = 1000,
)
    return ComboOrTest([
        RelResidualTest(rtol),
        MaxIterationsTest(max_iters),
    ])
end
