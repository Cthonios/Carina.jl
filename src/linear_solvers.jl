# Linear solver implementations for Carina.
#
# Contains all linear solve dispatch, preconditioner helpers, Krylov setup,
# and L-BFGS two-loop recursion + helpers.
#
# Depends on types from solvers.jl and integrator types from integrators.jl.

using LinearAlgebra
import Krylov
import LinearOperators: LinearOperator
import IterativeSolvers
import LimitedLDLFactorizations: lldl

# --------------------------------------------------------------------------- #
# Helper: preconditioner updates
# --------------------------------------------------------------------------- #

# Assembled path: update Jacobi diagonal directly from sparse K_eff matrix.
function _update_jacobi_precond_assembled!(precond::JacobiPreconditioner, K_eff)
    d = diag(K_eff)
    @. precond.inv_diag = 1.0 / max(abs(d), eps(Float64))
    return nothing
end
_update_jacobi_precond_assembled!(::NoPreconditioner, _) = nothing
_update_jacobi_precond_assembled!(::ICPreconditioner, _) = nothing  # IC built in _linear_solve!
_update_jacobi_precond_assembled!(::ChebyshevPreconditioner, _) = nothing  # bounds below

# Compute (K + c_M·M)·v via matrix-free actions, storing result in asm storage.
function _apply_eff_stiffness!(asm, U, v, c_M, p, scratch)
    FEC.assemble_matrix_free_action!(asm, FEC.stiffness_action, U, v, p)
    copyto!(scratch, asm.stiffness_action_storage.data)
    FEC.assemble_matrix_free_action!(asm, FEC.mass_action, U, v, p)
    @. asm.stiffness_action_storage.data = scratch + c_M * asm.stiffness_action_storage.data
end

# Matrix-free Jacobi preconditioner: diag(K + c_M·M) via diagonal extraction.
function _update_jacobi_precond_eff!(precond::JacobiPreconditioner, asm, U, ones_v, c_M, p, scratch)
    FEC.assemble_diagonal!(asm, FEC.stiffness, U, p)
    copyto!(scratch, FEC.diagonal(asm))
    FEC.assemble_diagonal!(asm, FEC.mass, U, p)
    d_mass = FEC.diagonal(asm)
    @. precond.inv_diag = 1.0 / max(abs(scratch + c_M * d_mass), eps(Float64))
    return nothing
end
_update_jacobi_precond_eff!(::Preconditioner, args...) = nothing

# GPU matrix-free displacement Jacobian: y = (K + c_M·M)·v
function _eff_stiffness_matvec!(y, v, asm, U, c_M, p, scratch)
    _apply_eff_stiffness!(asm, U, v, c_M, p, scratch)
    copyto!(y, FEC.hvp(asm, v))
    return y
end

# QS matrix-free Jacobi: true diag(K) via diagonal extraction kernel.
function _update_jacobi_precond_qs!(precond::JacobiPreconditioner, asm, U, ones_v, p)
    FEC.assemble_diagonal!(asm, FEC.stiffness, U, p)
    d = FEC.diagonal(asm)
    @. precond.inv_diag = 1.0 / max(abs(d), eps(Float64))
    return nothing
end
_update_jacobi_precond_qs!(::Preconditioner, args...) = nothing

# Assembled Chebyshev: bounds are estimated inside _build_precond_op (which
# constructs the symmetrically-scaled operator S = D⁻¹/²AD⁻¹/²), so the
# setup_jacobian! update is a no-op on the assembled path.
_update_chebyshev_precond_assembled!(::ChebyshevPreconditioner, _) = nothing
_update_chebyshev_precond_assembled!(::Preconditioner, _) = nothing

# QS matrix-free path: estimate λ_max of D⁻¹K via power method.
# D⁻¹/² is stored in work3 for use by _chebyshev_precond_op.
function _update_chebyshev_precond_qs!(precond::ChebyshevPreconditioner, asm, U, p)
    n = length(U)
    FEC.assemble_diagonal!(asm, FEC.stiffness, U, p)
    d = FEC.diagonal(asm)
    inv_sqrt_d = precond.work3
    @. inv_sqrt_d = 1.0 / sqrt(max(abs(d), eps(Float64)))
    inv_diag = similar(d)
    @. inv_diag = 1.0 / max(abs(d), eps(Float64))
    matvec! = (y, v) -> _stiffness_matvec_qs!(y, v, asm, U, p)
    _estimate_lambda_max!(precond, matvec!, inv_diag, n)
    return nothing
end
_update_chebyshev_precond_qs!(::Preconditioner, args...) = nothing

# Newmark matrix-free path: estimate λ_max of D⁻¹K_eff.
function _update_chebyshev_precond_eff!(precond::ChebyshevPreconditioner, asm, U, c_M, p, scratch)
    n = length(U)
    FEC.assemble_diagonal!(asm, FEC.stiffness, U, p)
    d_k = copy(FEC.diagonal(asm))
    FEC.assemble_diagonal!(asm, FEC.mass, U, p)
    d_m = FEC.diagonal(asm)
    inv_sqrt_d = precond.work3
    @. inv_sqrt_d = 1.0 / sqrt(max(abs(d_k + c_M * d_m), eps(Float64)))
    inv_diag = similar(d_k)
    @. inv_diag = 1.0 / max(abs(d_k + c_M * d_m), eps(Float64))
    matvec! = (y, v) -> _eff_stiffness_matvec!(y, v, asm, U, c_M, p, scratch)
    _estimate_lambda_max!(precond, matvec!, inv_diag, n)
    return nothing
end
_update_chebyshev_precond_eff!(::Preconditioner, args...) = nothing

# Jacobi preconditioner as LinearOperator (shared by Krylov paths).
function _jacobi_precond_op(precond::JacobiPreconditioner, n)
    LinearOperator(Float64, n, n, true, true,
        (y, v) -> (@. y = precond.inv_diag * v; y))
end
_jacobi_precond_op(::NoPreconditioner, n) = nothing

# QS K·v via stiffness_action.
function _stiffness_matvec_qs!(y, v, asm, U, p)
    FEC.assemble_matrix_free_action!(asm, FEC.stiffness_action, U, v, p)
    copyto!(y, FEC.hvp(asm, v))
    return y
end

# --------------------------------------------------------------------------- #
# Chebyshev preconditioner: 4th-kind with optimal weights
#
# Based on Ifpack2's implementation of Chebyshev polynomials of the 4th kind
# with optimal coefficients (arxiv 2202.08830).
#
# Key advantages over standard Chebyshev:
#   - Only needs λ_max (no λ_min or eigenvalue ratio)
#   - Baked-in Jacobi scaling (D⁻¹ applied at each iteration)
#   - SPD by construction (optimal weights guarantee positive definiteness)
#   - k matvecs per application (not 2k like the squared polynomial)
#
# λ_max estimated via power method on D⁻¹A (10 iterations, cheap).
# --------------------------------------------------------------------------- #

# Optimal weights for 4th-kind Chebyshev, degrees 1–16 (arxiv 2202.08830).
const _CHEBYSHEV_OPT_WEIGHTS = (
    [1.12500000000000],
    [1.02387287570313, 1.26408905371085],
    [1.00842544782028, 1.08867839208730, 1.33753125909618],
    [1.00391310427285, 1.04035811188593, 1.14863498546254, 1.38268869241000],
    [1.00212930146164, 1.02173711549260, 1.07872433192603, 1.19810065292663,
     1.41322542791682],
    [1.00128517255940, 1.01304293035233, 1.04678215124113, 1.11616489419675,
     1.23829020218444, 1.43524297106744],
    [1.00083464397912, 1.00843949430122, 1.03008707768713, 1.07408384092003,
     1.15036186707366, 1.27116474046139, 1.45186658649364],
    [1.00057246631197, 1.00577427662415, 1.02050187922941, 1.05019803444565,
     1.10115572984941, 1.18086042806856, 1.29838585382576, 1.46486073151099],
)

# Estimate λ_max of D⁻¹A via power method (spectral radius).
function _estimate_lambda_max!(precond::ChebyshevPreconditioner, matvec!, inv_diag, n;
                                power_iters::Int=10, boost::Float64=1.1)
    x = precond.work1
    y = precond.work2
    fill!(x, 1.0 / sqrt(Float64(n)))

    for _ in 1:power_iters
        matvec!(y, x)           # y = A·x
        @. x = inv_diag * y     # x = D⁻¹·A·x
        nrm = sqrt(dot(x, x))
        nrm < 1e-14 && break
        @. x = x / nrm
    end

    # Final Rayleigh quotient: λ = xᵀ(D⁻¹A)x
    matvec!(y, x)
    @. y = inv_diag * y
    lmax = dot(x, y)
    precond.lambda_max[] = lmax * boost

    return nothing
end

_estimate_lambda_max!(::Preconditioner, args...; kwargs...) = nothing

# Apply 4th-kind Chebyshev preconditioner with optimal weights.
#
# Computes y = M·b where M ≈ A⁻¹, using the recurrence:
#   Z₁  = (4/3)·σ·D⁻¹·b,  X₄₁ = Z₁,  y = β₁·Z₁
#   For i = 1..k-1:
#     γ = (2i-1)/(2i+3), ρ = (8i+4)/(2i+3)·σ
#     Zᵢ₊₁ = ρ·D⁻¹·(b − A·X₄ᵢ) + γ·Zᵢ
#     X₄ᵢ₊₁ = X₄ᵢ + Zᵢ₊₁
#     y += βᵢ₊₁·Zᵢ₊₁
# where σ = 1/(λ_max·boost) and βᵢ are the optimal weights.
#
# Cost: k matvecs per application.
function _apply_chebyshev_precond!(y, b, precond::ChebyshevPreconditioner,
                                    matvec!, inv_diag)
    k    = precond.degree
    lmax = precond.lambda_max[]
    σ    = 1.0 / lmax

    betas = k <= length(_CHEBYSHEV_OPT_WEIGHTS) ? _CHEBYSHEV_OPT_WEIGHTS[k] : ones(k)

    z  = precond.work1   # current update step
    x4 = precond.work2   # raw 4th-kind iterate
    w  = precond.work3   # scratch for matvec output

    # Iteration 0 (zero initial guess)
    @. z  = (4.0/3.0 * σ) * inv_diag * b
    copyto!(x4, z)
    @. y  = betas[1] * z

    for i in 1:k-1
        γ = (2.0*i - 1.0) / (2.0*i + 3.0)
        ρ = (8.0*i + 4.0) / (2.0*i + 3.0) * σ
        matvec!(w, x4)                          # w = A·x4
        @. z  = ρ * inv_diag * (b - w) + γ * z  # new Z
        @. x4 = x4 + z                          # advance x4
        @. y  = y + betas[i+1] * z              # weighted accumulation
    end
    return y
end

# Wrap Chebyshev preconditioner as a LinearOperator for Krylov.jl.
# The 4th-kind Chebyshev has baked-in Jacobi (D⁻¹) so no external scaling
# is needed — just pass inv_diag from the Jacobi preconditioner.
function _chebyshev_precond_op(precond::ChebyshevPreconditioner, n, raw_matvec!;
                                inv_sqrt_d=nothing)
    # inv_sqrt_d is D⁻¹/² from setup; we need D⁻¹ = (D⁻¹/²)²
    if inv_sqrt_d !== nothing
        inv_d = copy(inv_sqrt_d)
        @. inv_d = inv_sqrt_d * inv_sqrt_d   # D⁻¹ = (D⁻¹/²)²
    else
        inv_d = ones(n)  # fallback: identity scaling
    end
    return LinearOperator(Float64, n, n, true, true,
        (y, v) -> _apply_chebyshev_precond!(y, v, precond, raw_matvec!, inv_d))
end

# --------------------------------------------------------------------------- #
# Two-loop L-BFGS recursion (unchanged)
# --------------------------------------------------------------------------- #
#
# Computes d = H_k · R_eff (L-BFGS descent direction).
# Convention: ∇Φ = −R_eff, so d = H·R_eff = −H·∇Φ is the descent direction.
# Ring buffer: newest history at slot `head` (1-indexed), `hfill` valid entries.
#
# Initial Hessian H₀ priority:
#   1. hfill > 0: Barzilai-Borwein γ₀ = (s·y)/(y·y) from last history pair.
#   2. precond is JacobiPreconditioner: H₀ = diag(inv_diag), giving correct
#      dimensional scaling on the first step (critical when c_M ≫ 1).
#   3. Fallback: H₀ = I.
function _lbfgs_two_loop!(d, q, R_eff, S, Y, ρ, alpha, head, hfill, m, precond)
    copyto!(q, R_eff)

    # First loop: newest → oldest (i = 1 → newest at S[head], i = hfill → oldest)
    for i in 1:hfill
        idx = mod1(head - i + 1, m)
        alpha[i] = ρ[idx] * dot(S[idx], q)
        @. q = q - alpha[i] * Y[idx]
    end

    # Apply initial Hessian H₀.
    if hfill > 0
        # Barzilai-Borwein scaling from most recent history pair.
        sy = dot(S[head], Y[head])
        yy = dot(Y[head], Y[head])
        γ₀ = (sy > 0.0 && yy > 0.0) ? sy / yy : 1.0
        @. d = γ₀ * q
    elseif !(precond isa NoPreconditioner)
        # Jacobi (diagonal) preconditioner: H₀ = diag(inv_diag).
        # Essential on the first step when c_M = 1/(β·Δt²) ≫ 1.
        @. d = precond.inv_diag * q
    else
        copyto!(d, q)   # H₀ = I (fallback)
    end

    # Second loop: oldest → newest (i = hfill → oldest, i = 1 → newest)
    for i in hfill:-1:1
        idx = mod1(head - i + 1, m)
        β = ρ[idx] * dot(Y[idx], d)
        @. d = d + (alpha[i] - β) * S[idx]
    end

    return d
end

# --------------------------------------------------------------------------- #
# Krylov setup: _setup_linear_ops
# Dispatch on BOTH integrator and linear solver types.
# --------------------------------------------------------------------------- #

_setup_linear_ops(ig, ::DirectLinearSolver, p)  = nothing
_setup_linear_ops(ig, ::LBFGSLinearSolver,  p)  = nothing
_setup_linear_ops(ig, ::NoLinearSolver,     p)  = nothing

# Generic preconditioner → LinearOperator for matrix-free path.
# matvec! is the system operator A (needed by Chebyshev; ignored by Jacobi).
_mf_precond_op(::NoPreconditioner, n, matvec!)          = nothing
_mf_precond_op(p::JacobiPreconditioner, n, matvec!)     = _jacobi_precond_op(p, n)
_mf_precond_op(p::ChebyshevPreconditioner, n, matvec!)  =
    _chebyshev_precond_op(p, n, matvec!; inv_sqrt_d=p.work3)

function _setup_linear_ops(ig::QuasiStaticIntegrator, ls::KrylovLinearSolver, p)
    U = ig.U; n = length(U)
    ls.assembled && return (nothing, nothing)
    matvec! = (y, v) -> _stiffness_matvec_qs!(y, v, ig.asm, U, p)
    K_op = LinearOperator(Float64, n, n, true, true, matvec!)
    return K_op, _mf_precond_op(ls.precond, n, matvec!)
end

function _setup_linear_ops(ig::NewmarkIntegrator, ls::KrylovLinearSolver, p)
    U = ig.U; n = length(U); c_M = ig.c_M
    ls.assembled && return (nothing, nothing)
    matvec! = (y, v) -> _eff_stiffness_matvec!(y, v, ig.asm, U, c_M, p, ls.scratch)
    K_eff_op = LinearOperator(Float64, n, n, true, true, matvec!)
    return K_eff_op, _mf_precond_op(ls.precond, n, matvec!)
end

# --------------------------------------------------------------------------- #
# Linear solvers: _linear_solve!(ls, ig, p, ops) → (ΔU, t_solve)
# Sign convention: K_eff · ΔU = ig.R_eff  (ig.R_eff is already negated residual)
# --------------------------------------------------------------------------- #

function _linear_solve!(::DirectLinearSolver, ig, p, _ops)
    K  = FEC.stiffness(ig.asm)
    af = _asm_flags
    t  = @elapsed begin
        if af.compute_factorization
            # NOTE: K is SPD in theory, but FEC's assembly produces a slightly
            # asymmetric matrix (~1e-7 relative) due to the AD material tangent
            # path.  Cholesky(Symmetric(K)) reads only one triangle, giving a
            # ~50% solve error.  Use LU until the assembly is exactly symmetric,
            # then switch to cholesky(Symmetric(K)) for ~2× speedup.
            F = lu(K)
            if af.is_linear
                _factorization_cache[] = F
                af.compute_factorization = false
            end
        else
            F = _factorization_cache[]
        end
        ΔU = F \ residual(ig)
    end
    return ΔU, t
end

function _build_precond_op(::NoPreconditioner, K_sparse, n)
    return nothing
end
function _build_precond_op(precond::JacobiPreconditioner, K_sparse, n)
    return _jacobi_precond_op(precond, n)
end
function _build_precond_op(::ICPreconditioner, K_sparse, n)
    # Incomplete LDLᵀ factorization.
    # K_sparse is already Symmetric from the symmetrization in _linear_solve!.
    # α > 0 adds a diagonal shift to guarantee positive definiteness
    # of the factor (at the cost of a weaker preconditioner).
    F_ic = lldl(K_sparse; memory=20, α=0.01)
    return LinearOperator(Float64, n, n, true, true,
        (y, v) -> ldiv!(y, F_ic, v))
end
# Chebyshev-Jacobi on assembled path: polynomial on the symmetrically
# scaled system S = D⁻¹/²AD⁻¹/².  Penalty BCs create eigenvalues ~1e15
# in A but only ~1 in S, making the polynomial effective.
# Preconditioner: M = D⁻¹/² p_k(S)² D⁻¹/²  ≈ A⁻¹, and M is SPD.
function _build_precond_op(precond::ChebyshevPreconditioner, K_sparse, n)
    d = diag(K_sparse)
    inv_diag = similar(d)
    @. inv_diag = 1.0 / max(abs(d), eps(Float64))

    # Estimate λ_max of D⁻¹A via power method
    matvec! = (y, v) -> mul!(y, K_sparse, v)
    _estimate_lambda_max!(precond, matvec!, inv_diag, n)

    # 4th-kind Chebyshev with baked-in Jacobi
    return LinearOperator(Float64, n, n, true, true,
        (y, v) -> _apply_chebyshev_precond!(y, v, precond, matvec!, inv_diag))
end

function _linear_solve!(ls::KrylovLinearSolver, ig::QuasiStaticIntegrator, p, ops)
    U = ig.U; asm = ig.asm; n = length(U)
    K_op, M_op = ops
    R = residual(ig)   # K·ΔU = R_eff (positive, already negated)
    af = _asm_flags
    t_kry = @elapsed begin
        if ls.assembled
            K_raw = FEC.stiffness(asm)
            # FEC assembly produces a slightly asymmetric K (~1e-7 relative)
            # due to the AD material tangent.  CG requires exact symmetry.
            K_sparse = Symmetric((K_raw + K_raw') / 2, :L)
            if af.compute_factorization
                M_op_asm = _build_precond_op(ls.precond, K_sparse, n)
                if af.is_linear
                    _precond_op_cache[] = M_op_asm
                    af.compute_factorization = false
                end
            else
                M_op_asm = _precond_op_cache[]
            end
            if M_op_asm === nothing
                Krylov.krylov_solve!(ls.workspace, K_sparse, R;
                                     atol=0.0, rtol=ls.rtol, itmax=ls.itmax, history=true)
            else
                Krylov.krylov_solve!(ls.workspace, K_sparse, R;
                                     M=M_op_asm, atol=0.0, rtol=ls.rtol, itmax=ls.itmax, history=true)
            end
        else
            if M_op === nothing
                Krylov.krylov_solve!(ls.workspace, K_op, R;
                                     atol=0.0, rtol=ls.rtol, itmax=ls.itmax, history=true)
            else
                Krylov.krylov_solve!(ls.workspace, K_op, R;
                                     M=M_op, atol=0.0, rtol=ls.rtol, itmax=ls.itmax, history=true)
            end
        end
    end
    ΔU  = copy(Krylov.solution(ls.workspace))
    res = ls.workspace.stats.residuals
    r_cg = isempty(res) ? NaN : res[end]
    _carina_logf(8, :solve, "    CG: %d iters : |r|_CG = %.2e : %s",
                 ls.workspace.stats.niter, r_cg,
                 _cg_status_str(ls.workspace.stats.solved))
    return ΔU, t_kry
end

function _linear_solve!(ls::KrylovLinearSolver, ig::NewmarkIntegrator, p, ops)
    asm = ig.asm; n = length(ig.U)
    K_eff_op, M_op_mf = ops
    R = residual(ig)
    af = _asm_flags
    ΔU = similar(ig.U)
    t_kry = @elapsed begin
        try
            if ls.assembled
                K_eff_raw = FEC.stiffness(asm)
                # Symmetrize: FEC assembly is ~1e-7 asymmetric (AD tangent).
                K_eff_sparse = Symmetric((K_eff_raw + K_eff_raw') / 2, :L)
                if ls.precond isa ICPreconditioner
                    if af.compute_factorization
                        F_ic = lldl(K_eff_sparse)
                        if af.is_linear
                            _factorization_cache[] = F_ic
                            af.compute_factorization = false
                        end
                    else
                        F_ic = _factorization_cache[]
                    end
                    ΔU_vec, cg_hist = IterativeSolvers.cg(K_eff_sparse, R;
                        Pl=F_ic, abstol=0.0, reltol=ls.rtol, log=true)
                else
                    ΔU_vec, cg_hist = IterativeSolvers.cg(K_eff_sparse, R;
                        abstol=0.0, reltol=ls.rtol, log=true)
                end
                _carina_logf(8, :solve, "    CG: %d iters : |r|_CG = %.2e : %s",
                    length(cg_hist.data[:resnorm]),
                    cg_hist.data[:resnorm][end],
                    _cg_status_str(cg_hist.isconverged))
                copyto!(ΔU, ΔU_vec)
            else
                if M_op_mf === nothing
                    Krylov.krylov_solve!(ls.workspace, K_eff_op, R;
                        atol=0.0, rtol=ls.rtol, itmax=ls.itmax, history=true)
                else
                    Krylov.krylov_solve!(ls.workspace, K_eff_op, R;
                        M=M_op_mf, atol=0.0, rtol=ls.rtol, itmax=ls.itmax, history=true)
                end
                copyto!(ΔU, Krylov.solution(ls.workspace))
                res = ls.workspace.stats.residuals
                r_cg = isempty(res) ? NaN : res[end]
                _carina_logf(8, :solve, "    CG: %d iters : |r|_CG = %.2e : %s",
                             ls.workspace.stats.niter, r_cg,
                             _cg_status_str(ls.workspace.stats.solved))
            end
        catch
            ig.failed[] = true
        end
    end
    return ΔU, t_kry
end

# --------------------------------------------------------------------------- #
# LBFGS helpers (dispatch on integrator type for Newmark vs QS differences)
# --------------------------------------------------------------------------- #

# ---- _lbfgs_init_M_dU! ----

_lbfgs_init_M_dU!(::QuasiStaticIntegrator, ls) = nothing

function _lbfgs_init_M_dU!(::NewmarkIntegrator, ls)
    fill!(ls.M_dU, zero(eltype(ls.M_dU)))
end

# ---- _lbfgs_precompute_M_d! ----

_lbfgs_precompute_M_d!(::QuasiStaticIntegrator, ls, p) = nothing

function _lbfgs_precompute_M_d!(ig::NewmarkIntegrator, ls, p)
    FEC.assemble_matrix_free_action!(ig.asm, FEC.mass_action, ig.U, ls.d, p)
    copyto!(ls.M_d, FEC.hvp(ig.asm, ls.d))
end

# ---- _lbfgs_update_M_dU! ----

_lbfgs_update_M_dU!(::QuasiStaticIntegrator, ls, step) = nothing

function _lbfgs_update_M_dU!(::NewmarkIntegrator, ls, step)
    @. ls.M_dU += step * ls.M_d
end

# ---- _lbfgs_trial_rhs! ----
# Sets ig.R_eff at trial point U + step*d.

function _lbfgs_trial_rhs!(ig::QuasiStaticIntegrator, ls, step, p)
    U = ig.U; asm = ig.asm
    @. ls.q = U + step * ls.d
    FEC.assemble_vector!(asm, FEC.residual, ls.q, p)
    FEC.assemble_vector_neumann_bc!(asm, ls.q, p)
    FEC.assemble_vector_source!(asm, ls.q, p)
    R_int_trial = FEC.residual(asm)
    @. ig.R_eff = -R_int_trial
    _apply_point_loads!(ig.R_eff, FEC.current_time(p.times))
end

function _lbfgs_trial_rhs!(ig::NewmarkIntegrator, ls, step, p)
    α_hht = ig.α_hht; c_M = ig.c_M
    @. ls.q = ig.U + step * ls.d
    FEC.assemble_vector!(ig.asm, FEC.residual, ls.q, p)
    FEC.assemble_vector_neumann_bc!(ig.asm, ls.q, p)
    FEC.assemble_vector_source!(ig.asm, ls.q, p)
    R_int_trial = FEC.residual(ig.asm)
    @. ig.R_eff = -((1 + α_hht) * R_int_trial + c_M * (ls.M_dU + step * ls.M_d) - α_hht * ig.F_int_n)
    _apply_point_loads!(ig.R_eff, FEC.current_time(p.times))
end
