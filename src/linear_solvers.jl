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

function _setup_linear_ops(ig::QuasiStaticIntegrator, ls::KrylovLinearSolver, p)
    U = ig.solution; n = length(U)
    ls.assembled && return (nothing, nothing)
    K_op = LinearOperator(Float64, n, n, true, true,
        (y, v) -> _stiffness_matvec_qs!(y, v, ig.asm, U, p))
    return K_op, _jacobi_precond_op(ls.precond, n)
end

function _setup_linear_ops(ig::NewmarkIntegrator, ls::KrylovLinearSolver, p)
    U = ig.U; n = length(U); c_M = ig.c_M
    ls.assembled && return (nothing, nothing)
    K_eff_op = LinearOperator(Float64, n, n, true, true,
        (y, v) -> _eff_stiffness_matvec!(y, v, ig.asm, U, c_M, p, ls.scratch))
    return K_eff_op, _jacobi_precond_op(ls.precond, n)
end

# --------------------------------------------------------------------------- #
# Linear solvers: _linear_solve!(ls, ig, p, ops) → (ΔU, t_solve)
# Sign convention: K_eff · ΔU = ig.R_eff  (ig.R_eff is already negated residual)
# --------------------------------------------------------------------------- #

function _linear_solve!(::DirectLinearSolver, ig, p, _ops)
    K  = FEC.stiffness(ig.asm)
    t  = @elapsed begin
        # NOTE: K is SPD in theory, but FEC's assembly produces a slightly
        # asymmetric matrix (~1e-7 relative) due to the AD material tangent
        # path.  Cholesky(Symmetric(K)) reads only one triangle, giving a
        # ~50% solve error.  Use LU until the assembly is exactly symmetric,
        # then switch to cholesky(Symmetric(K)) for ~2× speedup.
        F  = lu(K)
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
    # Incomplete LDLᵀ factorization from the lower triangle of K.
    # α > 0 adds a diagonal shift to guarantee positive definiteness
    # of the factor (at the cost of a weaker preconditioner).
    F_ic = lldl(Symmetric(K_sparse, :L); memory=20, α=0.01)
    return LinearOperator(Float64, n, n, true, true,
        (y, v) -> ldiv!(y, F_ic, v))
end

function _linear_solve!(ls::KrylovLinearSolver, ig::QuasiStaticIntegrator, p, ops)
    U = ig.solution; asm = ig.asm; n = length(U)
    K_op, M_op = ops
    R = residual(ig)   # K·ΔU = R_eff (positive, already negated)
    t_kry = @elapsed begin
        if ls.assembled
            K_sparse = FEC.stiffness(asm)
            M_op_asm = _build_precond_op(ls.precond, K_sparse, n)
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
    ΔU = similar(ig.U)
    t_kry = @elapsed begin
        try
            if ls.assembled
                K_eff_sparse = FEC.stiffness(asm)
                if ls.precond isa ICPreconditioner
                    Ks = Symmetric((K_eff_sparse + K_eff_sparse') / 2)
                    F_ic = lldl(Ks)
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
