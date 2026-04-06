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

# Matrix-free Jacobi preconditioner: diag(K + c_M·M) via (K + c_M·M)·ones.
function _update_jacobi_precond_eff!(precond::JacobiPreconditioner, asm, U, ones_v, c_M, p, scratch)
    _apply_eff_stiffness!(asm, U, ones_v, c_M, p, scratch)
    d_eff = FEC.hvp(asm, ones_v)
    @. precond.inv_diag = 1.0 / max(abs(d_eff), eps(Float64))
    return nothing
end
_update_jacobi_precond_eff!(::NoPreconditioner, args...) = nothing

# GPU matrix-free displacement Jacobian: y = (K + c_M·M)·v
function _eff_stiffness_matvec!(y, v, asm, U, c_M, p, scratch)
    _apply_eff_stiffness!(asm, U, v, c_M, p, scratch)
    copyto!(y, FEC.hvp(asm, v))
    return y
end

# QS matrix-free Jacobi: uses stiffness_action only (no mass).
function _update_jacobi_precond_qs!(precond::JacobiPreconditioner, asm, U, ones_v, p)
    FEC.assemble_matrix_free_action!(asm, FEC.stiffness_action, U, ones_v, p)
    d = FEC.hvp(asm, ones_v)
    @. precond.inv_diag = 1.0 / max(abs(d), eps(Float64))
    return nothing
end
_update_jacobi_precond_qs!(::NoPreconditioner, args...) = nothing

# Assembled Chebyshev: bounds are estimated inside _build_precond_op (which
# constructs the symmetrically-scaled operator S = D⁻¹/²AD⁻¹/²), so the
# setup_jacobian! update is a no-op on the assembled path.
_update_chebyshev_precond_assembled!(::ChebyshevPreconditioner, _) = nothing
_update_chebyshev_precond_assembled!(::Preconditioner, _) = nothing

# QS matrix-free path: matvec from stiffness_action.
function _update_chebyshev_precond_qs!(precond::ChebyshevPreconditioner, asm, U, p)
    n = length(U)
    matvec! = (y, v) -> _stiffness_matvec_qs!(y, v, asm, U, p)
    _estimate_spectral_bounds!(precond, matvec!, n)
    return nothing
end
_update_chebyshev_precond_qs!(::Preconditioner, args...) = nothing

# Newmark matrix-free path: matvec from effective stiffness (K + c_M·M).
function _update_chebyshev_precond_eff!(precond::ChebyshevPreconditioner, asm, U, c_M, p, scratch)
    n = length(U)
    matvec! = (y, v) -> _eff_stiffness_matvec!(y, v, asm, U, c_M, p, scratch)
    _estimate_spectral_bounds!(precond, matvec!, n)
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
# Chebyshev preconditioner: spectral bound estimation (Lanczos)
# --------------------------------------------------------------------------- #

# Estimate extremal eigenvalues of SPD operator A via short Lanczos iteration.
# Builds tridiagonal T = tridiag(β, α, β) and returns its extremal eigenvalues.
#
# matvec!(y, x) must compute y = A·x in-place.
function _estimate_spectral_bounds!(precond::ChebyshevPreconditioner, matvec!, n;
                                     lanczos_steps::Int=20)
    kmax = min(lanczos_steps, n)

    # Three vectors needed: v_curr, v_prev, w (matvec output).
    v_curr = precond.work1
    v_prev = precond.work2
    w = similar(v_curr)

    # Deterministic starting vector: uniform 1/√n.
    fill!(v_curr, 1.0 / sqrt(Float64(n)))
    fill!(v_prev, 0.0)

    alphas = Vector{Float64}(undef, kmax)
    betas  = Vector{Float64}(undef, kmax)
    beta_prev = 0.0
    k_actual = kmax

    for j in 1:kmax
        matvec!(w, v_curr)
        alphas[j] = dot(v_curr, w)
        @. w = w - alphas[j] * v_curr - beta_prev * v_prev

        beta_j = sqrt(dot(w, w))
        betas[j] = beta_j

        if beta_j < 1e-14
            k_actual = j
            break
        end

        # Rotate: v_prev ← v_curr, v_curr ← w/β_j
        copyto!(v_prev, v_curr)
        @. v_curr = w / beta_j
        beta_prev = beta_j
    end

    # Eigenvalues of the tridiagonal matrix.
    # β_j connects row j to j+1, so off-diagonal has k-1 entries: β_1..β_{k-1}.
    T = SymTridiagonal(alphas[1:k_actual], betas[1:k_actual-1])
    eigs = eigvals(T)

    # Clamp to positive (A is SPD) and widen by 5% for robustness.
    lmin = max(eigs[1],   1e-14)
    lmax = max(eigs[end], lmin * 1.01)
    precond.lambda_min[] = lmin * 0.95
    precond.lambda_max[] = lmax * 1.05

    return nothing
end

# No-op dispatches for non-Chebyshev preconditioners.
_estimate_spectral_bounds!(::NoPreconditioner, args...; kwargs...)    = nothing
_estimate_spectral_bounds!(::JacobiPreconditioner, args...; kwargs...) = nothing
_estimate_spectral_bounds!(::ICPreconditioner, args...; kwargs...)    = nothing

# --------------------------------------------------------------------------- #
# Chebyshev preconditioner: polynomial application
# --------------------------------------------------------------------------- #

# Apply degree-k Chebyshev polynomial preconditioner: y = p_k(A)²·v (SPD).
#
# The raw polynomial p_k(A) ≈ A⁻¹ is NOT guaranteed SPD — the polynomial
# can go negative on [λ_min, λ_max].  CG requires an SPD preconditioner,
# so we apply the squared form p_k(A)² which is always SPD:
#   ⟨p_k(A)²v, v⟩ = ⟨p_k(A)v, p_k(A)v⟩ = ‖p_k(A)v‖² ≥ 0.
# Cost: 2k matvecs per application (two passes of the degree-k polynomial).
#
# Inner recurrence (single pass, PETSc/Trilinos formulation):
#   x₀ = (1/θ)·b
#   x₁ = x₀ + ρ₁·(2/δ)·(b - A·x₀)           ρ₁ = 1/σ
#   xⱼ = xⱼ₋₁ + ρⱼ·((2/δ)·(b - A·xⱼ₋₁) + ρⱼ₋₁·(xⱼ₋₂ - xⱼ₋₁))
#                                                ρⱼ = 1/(2σ - ρⱼ₋₁)
# where θ = (λ_max+λ_min)/2, δ = (λ_max-λ_min)/2, σ = θ/δ.

# Single-pass p_k(A)·v.  w and xp are scratch vectors.
function _chebyshev_poly_pass!(y, v, theta, delta, sigma, k, matvec!, w, xp)
    @. y = v / theta
    k == 0 && return y

    matvec!(w, y)
    rho_prev = 1.0 / sigma
    copyto!(xp, y)
    @. y = y + rho_prev * (2.0 / delta) * (v - w)

    for j in 2:k
        matvec!(w, y)
        rho = 1.0 / (2.0 * sigma - rho_prev)
        @. w = rho * ((2.0 / delta) * (v - w) + rho_prev * (xp - y))
        copyto!(xp, y)
        @. y = y + w
        rho_prev = rho
    end
    return y
end

function _apply_chebyshev_precond!(y, v, precond::ChebyshevPreconditioner, matvec!)
    k    = precond.degree
    lmin = precond.lambda_min[]
    lmax = precond.lambda_max[]

    theta = (lmax + lmin) / 2.0
    delta = (lmax - lmin) / 2.0
    sigma = theta / delta

    w  = precond.work1
    xp = precond.work2

    # First pass: y = p_k(A)·v
    _chebyshev_poly_pass!(y, v, theta, delta, sigma, k, matvec!, w, xp)

    # Second pass: y = p_k(A)·(p_k(A)·v) = p_k(A)²·v
    # Copy intermediate result to pre-allocated scratch, then apply again.
    t = precond.work3
    copyto!(t, y)
    _chebyshev_poly_pass!(y, t, theta, delta, sigma, k, matvec!, w, xp)

    return y
end

# Wrap Chebyshev preconditioner as a LinearOperator for Krylov.jl.
function _chebyshev_precond_op(precond::ChebyshevPreconditioner, n, matvec!)
    LinearOperator(Float64, n, n, true, true,
        (y, v) -> _apply_chebyshev_precond!(y, v, precond, matvec!))
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
_mf_precond_op(p::ChebyshevPreconditioner, n, matvec!)  = _chebyshev_precond_op(p, n, matvec!)

function _setup_linear_ops(ig::QuasiStaticIntegrator, ls::KrylovLinearSolver, p)
    U = ig.solution; n = length(U)
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
    inv_sqrt_d = similar(d)
    @. inv_sqrt_d = 1.0 / sqrt(max(abs(d), eps(Float64)))

    # Matvec for S = D⁻¹/²AD⁻¹/²: y = D⁻¹/² · A · (D⁻¹/² · v)
    tmp = similar(d)
    matvec_S! = (y, v) -> begin
        @. tmp = inv_sqrt_d * v
        mul!(y, K_sparse, tmp)
        @. y = inv_sqrt_d * y
    end

    # Estimate spectral bounds of S (should be near [~0.5, ~2] for FEM).
    _estimate_spectral_bounds!(precond, matvec_S!, n)

    # M·v = D⁻¹/² · p_k(S)² · D⁻¹/² · v
    # Pre-allocate scratch for the D⁻¹/²·v intermediate (avoid hot-path alloc).
    scaled_input = similar(d)
    return LinearOperator(Float64, n, n, true, true,
        (y, v) -> begin
            @. scaled_input = inv_sqrt_d * v   # D⁻¹/² · v
            _apply_chebyshev_precond!(y, scaled_input, precond, matvec_S!)
            @. y = inv_sqrt_d * y              # D⁻¹/² · result
        end)
end

function _linear_solve!(ls::KrylovLinearSolver, ig::QuasiStaticIntegrator, p, ops)
    U = ig.solution; asm = ig.asm; n = length(U)
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
    U = ig.solution; asm = ig.asm
    @. ls.q = U + step * ls.d
    FEC.assemble_vector!(asm, FEC.residual, ls.q, p)
    FEC.assemble_vector_neumann_bc!(asm, ls.q, p)
    FEC.assemble_vector_source!(asm, ls.q, p)
    R_int_trial = FEC.residual(asm)
    @. ig.R_eff = -R_int_trial
end

function _lbfgs_trial_rhs!(ig::NewmarkIntegrator, ls, step, p)
    α_hht = ig.α_hht; c_M = ig.c_M
    @. ls.q = ig.U + step * ls.d
    FEC.assemble_vector!(ig.asm, FEC.residual, ls.q, p)
    FEC.assemble_vector_neumann_bc!(ig.asm, ls.q, p)
    FEC.assemble_vector_source!(ig.asm, ls.q, p)
    R_int_trial = FEC.residual(ig.asm)
    @. ig.R_eff = -((1 + α_hht) * R_int_trial + c_M * (ls.M_dU + step * ls.M_d) - α_hht * ig.F_int_n)
end
