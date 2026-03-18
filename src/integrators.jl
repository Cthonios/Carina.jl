# Time integrators for Carina.
#
# Three integrators are provided:
#
#   QuasiStaticIntegrator   — pseudo-time quasi-static Newton (no inertia)
#   NewmarkIntegrator       — implicit Newmark-β (GPU-ready matrix-free + direct LU)
#   CentralDifferenceIntegrator — explicit central difference (GPU-ready)
#
# All integrators follow the same protocol with the TimeController:
#   • time_step       — current (adaptive) integration step
#   • min/max_time_step, decrease/increase_factor — adaptive stepping bounds
#   • failed          — Ref{Bool} set by FEC.evolve! to signal non-convergence
#   • _save_state! / _restore_state! — rollback on step failure
#   • _increase_step! / _decrease_step! — adjust time_step
#   • _pre_step_hook! — called before each sub-step (CFL update, no-op today)
#
# Solver factoring:
#   AbstractLinearSolver    — DirectLinearSolver | KrylovLinearSolver | LBFGSLinearSolver | NoLinearSolver
#   AbstractNonlinearSolver — NewtonSolver{LS}
#   Integrator{NS}          — QuasiStaticIntegrator | NewmarkIntegrator | CentralDifferenceIntegrator
#
# evolve! dispatch selects the implementation by (integrator type) × (linear solver type).

import FiniteElementContainers as FEC
using LinearAlgebra
import Krylov
import LinearOperators: LinearOperator
import IterativeSolvers

# --------------------------------------------------------------------------- #
# Abstract solver types
# --------------------------------------------------------------------------- #

abstract type AbstractLinearSolver end
abstract type AbstractNonlinearSolver end

# --------------------------------------------------------------------------- #
# Concrete linear solver types
# --------------------------------------------------------------------------- #

struct DirectLinearSolver <: AbstractLinearSolver end

mutable struct KrylovLinearSolver{KW, Vec} <: AbstractLinearSolver
    method   ::Symbol         # :minres or :cg
    itmax    ::Int
    rtol     ::Float64
    assembled::Bool           # true = CPU sparse K_eff; false = GPU matrix-free
    precond  ::Preconditioner
    workspace::KW
    ones_v   ::Vec
    scratch  ::Vec
end

mutable struct LBFGSLinearSolver{Vec, PC <: Preconditioner} <: AbstractLinearSolver
    m         ::Int
    precond   ::PC
    S         ::Vector{Vec}
    Y         ::Vector{Vec}
    ρ         ::Vector{Float64}
    alpha_buf ::Vector{Float64}
    head      ::Int
    hist_fill ::Int
    R_eff     ::Vec   # current effective residual
    R_old     ::Vec   # snapshot for y = R_old − R_new
    d         ::Vec   # descent direction
    q         ::Vec   # two-loop work / trial scratch
    M_d       ::Vec   # Newmark: M·d precomputed for line search
    M_dU      ::Vec   # Newmark: M·(U−U_pred) maintained incrementally
    F_int_n   ::Vec   # HHT-α: F_int at t_n
end

struct NoLinearSolver <: AbstractLinearSolver end

# --------------------------------------------------------------------------- #
# Newton nonlinear solver
# --------------------------------------------------------------------------- #

mutable struct NewtonSolver{LS <: AbstractLinearSolver} <: AbstractNonlinearSolver
    max_iters         ::Int
    abs_increment_tol ::Float64
    abs_residual_tol  ::Float64
    rel_residual_tol  ::Float64
    linear_solver     ::LS
end

# --------------------------------------------------------------------------- #
# QuasiStaticIntegrator{NS, Vec}
# --------------------------------------------------------------------------- #

mutable struct QuasiStaticIntegrator{NS <: AbstractNonlinearSolver, Vec}
    nonlinear_solver ::NS
    asm              ::Any
    solution         ::Vec   # free-DOF displacement accumulator
    time_step        ::Float64
    min_time_step    ::Float64
    max_time_step    ::Float64
    decrease_factor  ::Float64
    increase_factor  ::Float64
    failed           ::Base.RefValue{Bool}
    U_save           ::Vec   # rollback snapshot (used by LBFGS path)
    R_eff            ::Vec   # effective residual for Newton solve
end

function QuasiStaticIntegrator(ns::NS, asm, template::Vec;
                                time_step::Float64=0.0,
                                min_time_step::Float64=0.0,
                                max_time_step::Float64=0.0,
                                decrease_factor::Float64=1.0,
                                increase_factor::Float64=1.0) where {NS, Vec}
    T  = eltype(template)
    mk() = (v = similar(template); fill!(v, zero(T)); v)
    solution = mk()
    U_save   = mk()
    R_eff    = mk()
    return QuasiStaticIntegrator(ns, asm, solution, time_step, min_time_step, max_time_step,
                                  decrease_factor, increase_factor, Ref(false), U_save, R_eff)
end

# --------------------------------------------------------------------------- #
# NewmarkIntegrator{NS, Vec}
# --------------------------------------------------------------------------- #

mutable struct NewmarkIntegrator{NS <: AbstractNonlinearSolver, Vec}
    nonlinear_solver ::NS
    asm              ::Any
    β                ::Float64
    γ                ::Float64
    α_hht            ::Float64
    U   ::Vec; V   ::Vec; A   ::Vec
    U_prev::Vec; V_prev::Vec; A_prev::Vec
    U_pred ::Vec
    dU     ::Vec   # U − U_pred (Newton/Krylov paths)
    R_eff  ::Vec   # effective residual (Newton/Krylov paths; unused in LBFGS)
    c_M    ::Float64  # 1/(β·Δt²), updated by predict!
    time_step        ::Float64
    min_time_step    ::Float64
    max_time_step    ::Float64
    decrease_factor  ::Float64
    increase_factor  ::Float64
    failed           ::Base.RefValue{Bool}
    U_save::Vec; V_save::Vec; A_save::Vec
end

function NewmarkIntegrator(ns::NS, asm, β::Float64, γ::Float64, template::Vec;
                            α_hht::Float64=0.0,
                            time_step::Float64=0.0,
                            min_time_step::Float64=0.0,
                            max_time_step::Float64=0.0,
                            decrease_factor::Float64=1.0,
                            increase_factor::Float64=1.0) where {NS, Vec}
    T  = eltype(template)
    mk() = (v = similar(template); fill!(v, zero(T)); v)

    U, V, A             = mk(), mk(), mk()
    U_prev, V_prev, A_prev = mk(), mk(), mk()
    U_pred, dU, R_eff   = mk(), mk(), mk()
    U_save, V_save, A_save = mk(), mk(), mk()

    return NewmarkIntegrator(ns, asm, β, γ, α_hht,
                              U, V, A, U_prev, V_prev, A_prev,
                              U_pred, dU, R_eff,
                              0.0,   # c_M initialised to 0; set by predict!
                              time_step, min_time_step, max_time_step,
                              decrease_factor, increase_factor,
                              Ref(false),
                              U_save, V_save, A_save)
end

# --------------------------------------------------------------------------- #
# CentralDifferenceIntegrator — keep exactly as-is
# --------------------------------------------------------------------------- #

mutable struct CentralDifferenceIntegrator{Asm, Vec}
    γ::Float64
    asm::Asm
    U::Vec; V::Vec; A::Vec
    m_lumped::Vec
    # Adaptive time stepping
    time_step       ::Float64
    min_time_step   ::Float64
    max_time_step   ::Float64
    decrease_factor ::Float64
    increase_factor ::Float64
    # CFL (deferred — CFL=0.0 means disabled)
    CFL             ::Float64
    c_p_max         ::Float64
    failed          ::Base.RefValue{Bool}
    # Rollback state
    U_save::Vec; V_save::Vec; A_save::Vec
end

function CentralDifferenceIntegrator(γ::Float64, asm, m_lumped::Vec;
                                      time_step::Float64=0.0,
                                      min_time_step::Float64=0.0,
                                      max_time_step::Float64=0.0,
                                      decrease_factor::Float64=1.0,
                                      increase_factor::Float64=1.0,
                                      CFL::Float64=0.0,
                                      c_p_max::Float64=Inf) where {Vec}
    T  = eltype(m_lumped)
    mk() = (v = similar(m_lumped); fill!(v, zero(T)); v)
    U, V, A = mk(), mk(), mk()
    U_save, V_save, A_save = mk(), mk(), mk()
    return CentralDifferenceIntegrator(
        γ, asm, U, V, A, m_lumped,
        time_step, min_time_step, max_time_step, decrease_factor, increase_factor,
        CFL, c_p_max,
        Ref(false),
        U_save, V_save, A_save,
    )
end

# --------------------------------------------------------------------------- #
# Helper: preconditioner updates
# --------------------------------------------------------------------------- #

# Refresh Jacobi diagonal from K(U) + c_M·M at the current iterate U.
# Uses matrix-free action kernels (no 24×24 K_el formed).
function _update_jacobi_precond!(precond::JacobiPreconditioner, asm, U, ones_v, c_M, p, scratch)
    FEC.assemble_matrix_free_action!(asm, FEC.stiffness_action, U, ones_v, p)
    copyto!(scratch, asm.stiffness_action_storage.data)
    FEC.assemble_matrix_free_action!(asm, FEC.mass_action, U, ones_v, p)
    @. asm.stiffness_action_storage.data = scratch + c_M * asm.stiffness_action_storage.data
    d_eff = FEC.hvp(asm, ones_v)
    @. precond.inv_diag = 1.0 / max(abs(d_eff), eps(Float64))
    return nothing
end
_update_jacobi_precond!(::NoPreconditioner, args...) = nothing

# Assembled path: update Jacobi diagonal directly from sparse K_eff matrix.
function _update_jacobi_precond_assembled!(precond::JacobiPreconditioner, K_eff)
    d = diag(K_eff)
    @. precond.inv_diag = 1.0 / max(abs(d), eps(Float64))
    return nothing
end
_update_jacobi_precond_assembled!(::NoPreconditioner, _) = nothing

# Acceleration Jacobian Jacobi: diag(c_K·K + M) via matrix-free action.
function _update_jacobi_precond_eff!(precond::JacobiPreconditioner, asm, U, ones_v, c_M, p, scratch)
    FEC.assemble_matrix_free_action!(asm, FEC.stiffness_action, U, ones_v, p)
    copyto!(scratch, asm.stiffness_action_storage.data)
    FEC.assemble_matrix_free_action!(asm, FEC.mass_action, U, ones_v, p)
    @. asm.stiffness_action_storage.data = scratch + c_M * asm.stiffness_action_storage.data
    d_eff = FEC.hvp(asm, ones_v)
    @. precond.inv_diag = 1.0 / max(abs(d_eff), eps(Float64))
    return nothing
end
_update_jacobi_precond_eff!(::NoPreconditioner, args...) = nothing

# GPU matrix-free displacement Jacobian: y = (K + c_M·M)·v  — same as Norma's K_eff
function _eff_stiffness_matvec!(y, v, asm, U, c_M, p, scratch)
    FEC.assemble_matrix_free_action!(asm, FEC.stiffness_action, U, v, p)
    copyto!(scratch, asm.stiffness_action_storage.data)
    FEC.assemble_matrix_free_action!(asm, FEC.mass_action, U, v, p)
    @. asm.stiffness_action_storage.data = scratch + c_M * asm.stiffness_action_storage.data
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

# Matrix-free acceleration Jacobian: y = (c_K·K + M)·v  where c_K = β·Δt²
function _acc_stiffness_matvec!(y, v, asm, U, c_K, p, scratch)
    FEC.assemble_matrix_free_action!(asm, FEC.stiffness_action, U, v, p)
    copyto!(scratch, asm.stiffness_action_storage.data)
    FEC.assemble_matrix_free_action!(asm, FEC.mass_action, U, v, p)
    @. asm.stiffness_action_storage.data =
        c_K * scratch + asm.stiffness_action_storage.data
    copyto!(y, FEC.hvp(asm, v))
    return y
end

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
# predict! / correct! — integrator state transitions (Norma-style)
# --------------------------------------------------------------------------- #

predict!(::QuasiStaticIntegrator, p) = nothing

function predict!(ig::NewmarkIntegrator, p)
    (; β, γ, U, V, A, U_prev, V_prev, A_prev, U_pred, dU) = ig
    Δt    = FEC.time_step(p.times)
    ig.c_M = 1.0 / (β * Δt^2)
    copyto!(U_prev, U); copyto!(V_prev, V); copyto!(A_prev, A)
    @. U = U_prev + Δt * V_prev + Δt^2 * (0.5 - β) * A_prev
    @. V = V_prev + Δt * (1.0 - γ) * A_prev
    copyto!(U_pred, U)
    fill!(dU, zero(eltype(dU)))
    return nothing
end

correct!(::QuasiStaticIntegrator, p) = nothing

function correct!(ig::NewmarkIntegrator, p)
    Δt = FEC.time_step(p.times)
    @. ig.A = ig.c_M * (ig.U - ig.U_pred)
    @. ig.V += Δt * ig.γ * ig.A
    return nothing
end

# --------------------------------------------------------------------------- #
# _displacement — current displacement field used by assembler
# --------------------------------------------------------------------------- #

_displacement(ig::QuasiStaticIntegrator) = ig.solution
_displacement(ig::NewmarkIntegrator)     = ig.U

# --------------------------------------------------------------------------- #
# _eval_rhs! — compute R_eff stored in ig.R_eff; returns true on success
# --------------------------------------------------------------------------- #

function _eval_rhs!(ig::QuasiStaticIntegrator, p)
    U = ig.solution; asm = ig.asm
    FEC.assemble_vector!(asm, FEC.residual, U, p)
    FEC.assemble_vector_neumann_bc!(asm, U, p)
    R = FEC.residual(asm)
    @. ig.R_eff = -R
    return isfinite(sqrt(sum(abs2, ig.R_eff)))
end

function _eval_rhs!(ig::NewmarkIntegrator, p)
    (; asm, U, U_pred, dU, R_eff, c_M) = ig
    @. dU = U - U_pred
    FEC.assemble_vector!(asm, FEC.residual, U, p)
    FEC.assemble_vector_neumann_bc!(asm, U, p)
    FEC.assemble_matrix_free_action!(asm, FEC.mass_action, U, dU, p)
    R_int = FEC.residual(asm)
    M_dU  = FEC.hvp(asm, dU)
    @. R_eff = -(R_int + c_M * M_dU)
    return isfinite(sqrt(sum(abs2, R_eff)))
end

# --------------------------------------------------------------------------- #
# _setup_jacobian! — assemble K_eff (or update precond) at current U.
# Returns true on success, false on exception (sets ig.failed[]).
# --------------------------------------------------------------------------- #

function _setup_jacobian!(ig::QuasiStaticIntegrator{<:NewtonSolver{DirectLinearSolver}}, p)
    FEC.assemble_stiffness!(ig.asm, FEC.stiffness, ig.solution, p)
    return true
end

function _setup_jacobian!(ig::NewmarkIntegrator{<:NewtonSolver{DirectLinearSolver}}, p)
    asm = ig.asm; U = ig.U; c_M = ig.c_M
    FEC.assemble_stiffness!(asm, FEC.stiffness, U, p)
    FEC.assemble_mass!(asm, FEC.mass, U, p)
    @. asm.stiffness_storage += c_M * asm.mass_storage
    return true
end

function _setup_jacobian!(ig::QuasiStaticIntegrator{<:NewtonSolver{<:KrylovLinearSolver}}, p)
    ls = ig.nonlinear_solver.linear_solver; U = ig.solution; asm = ig.asm
    if ls.assembled
        FEC.assemble_stiffness!(asm, FEC.stiffness, U, p)
        _update_jacobi_precond_assembled!(ls.precond, FEC.stiffness(asm))
    else
        _update_jacobi_precond_qs!(ls.precond, asm, U, ls.ones_v, p)
    end
    return true
end

function _setup_jacobian!(ig::NewmarkIntegrator{<:NewtonSolver{<:KrylovLinearSolver}}, p)
    ls = ig.nonlinear_solver.linear_solver; asm = ig.asm; U = ig.U; c_M = ig.c_M
    if ls.assembled
        try
            FEC.assemble_stiffness!(asm, FEC.stiffness, U, p)
            FEC.assemble_mass!(asm, FEC.mass, U, p)
            @. asm.stiffness_storage += c_M * asm.mass_storage
            _update_jacobi_precond_assembled!(ls.precond, FEC.stiffness(asm))
        catch
            ig.failed[] = true
            return false
        end
    else
        _update_jacobi_precond_eff!(ls.precond, asm, U, ls.ones_v, c_M, p, ls.scratch)
    end
    return true
end

# --------------------------------------------------------------------------- #
# _setup_linear_ops — create LinearOperators for Krylov solves (once per step)
# Returns nothing for Direct; (K_op, M_op) tuple for Krylov.
# --------------------------------------------------------------------------- #

_setup_linear_ops(::QuasiStaticIntegrator{<:NewtonSolver{DirectLinearSolver}}, p)  = nothing
_setup_linear_ops(::NewmarkIntegrator{<:NewtonSolver{DirectLinearSolver}}, p)      = nothing

function _setup_linear_ops(ig::QuasiStaticIntegrator{<:NewtonSolver{<:KrylovLinearSolver}}, p)
    ls = ig.nonlinear_solver.linear_solver; U = ig.solution; n = length(U)
    ls.assembled && return (nothing, nothing)
    K_op = LinearOperator(Float64, n, n, true, true,
        (y, v) -> _stiffness_matvec_qs!(y, v, ig.asm, U, p))
    M_op = ls.precond isa NoPreconditioner ? nothing :
        LinearOperator(Float64, n, n, true, true,
            (y, v) -> (@. y = ls.precond.inv_diag * v; y))
    return K_op, M_op
end

function _setup_linear_ops(ig::NewmarkIntegrator{<:NewtonSolver{<:KrylovLinearSolver}}, p)
    ls = ig.nonlinear_solver.linear_solver; U = ig.U; n = length(U); c_M = ig.c_M
    ls.assembled && return (nothing, nothing)
    K_eff_op = LinearOperator(Float64, n, n, true, true,
        (y, v) -> _eff_stiffness_matvec!(y, v, ig.asm, U, c_M, p, ls.scratch))
    M_op = ls.precond isa NoPreconditioner ? nothing :
        LinearOperator(Float64, n, n, true, true,
            (y, v) -> (@. y = ls.precond.inv_diag * v; y))
    return K_eff_op, M_op
end

# --------------------------------------------------------------------------- #
# _solve_and_apply! — solve K_eff·ΔU = R_eff, apply U += ΔU.
# Returns (norm_step, t_solve).  Sets ig.failed[] = true on hard failure.
# --------------------------------------------------------------------------- #

# Direct: QS and Newmark share the same LU solve logic.
function _solve_and_apply!(
    ig::Union{QuasiStaticIntegrator{<:NewtonSolver{DirectLinearSolver}},
              NewmarkIntegrator{<:NewtonSolver{DirectLinearSolver}}}, p, _ops)
    U = _displacement(ig); asm = ig.asm
    t_lu = @elapsed begin
        K  = FEC.stiffness(asm)
        F  = lu(K)
        ΔU = F \ ig.R_eff
    end
    norm_step = sqrt(sum(abs2, ΔU))
    U .+= ΔU
    FEC._update_for_assembly!(p, asm.dof, U)
    return norm_step, t_lu
end

# QS Krylov (both assembled and matrix-free paths).
function _solve_and_apply!(
    ig::QuasiStaticIntegrator{<:NewtonSolver{<:KrylovLinearSolver}}, p, ops)
    ns = ig.nonlinear_solver; ls = ns.linear_solver
    U = ig.solution; asm = ig.asm; n = length(U)
    K_op, M_op = ops
    neg_R = -ig.R_eff
    t_kry = @elapsed begin
        if ls.assembled
            K_sparse = FEC.stiffness(asm)
            if ls.precond isa NoPreconditioner
                Krylov.krylov_solve!(ls.workspace, K_sparse, neg_R;
                                     atol=0.0, rtol=ls.rtol, itmax=ls.itmax)
            else
                M_op_asm = LinearOperator(Float64, n, n, true, true,
                    (y, v) -> (@. y = ls.precond.inv_diag * v; y))
                Krylov.krylov_solve!(ls.workspace, K_sparse, neg_R;
                                     M=M_op_asm, atol=0.0, rtol=ls.rtol, itmax=ls.itmax)
            end
        else
            if M_op === nothing
                Krylov.krylov_solve!(ls.workspace, K_op, neg_R;
                                     atol=0.0, rtol=ls.rtol, itmax=ls.itmax)
            else
                Krylov.krylov_solve!(ls.workspace, K_op, neg_R;
                                     M=M_op, atol=0.0, rtol=ls.rtol, itmax=ls.itmax)
            end
        end
    end
    ΔU        = Krylov.solution(ls.workspace)
    norm_step = sqrt(sum(abs2, ΔU))
    U .+= ΔU
    FEC._update_for_assembly!(p, asm.dof, U)
    kry_iters  = ls.workspace.stats.niter
    kry_solved = ls.workspace.stats.solved
    _carina_logf(8, :solve, "    Krylov: %d iters, solved=%s", kry_iters,
                 kry_solved ? "true" : "STALL")
    return norm_step, t_kry
end

# Newmark Krylov: assembled CPU uses IterativeSolvers.cg (Norma-identical);
# matrix-free GPU uses Krylov.krylov_solve!.
function _solve_and_apply!(
    ig::NewmarkIntegrator{<:NewtonSolver{<:KrylovLinearSolver}}, p, ops)
    ns = ig.nonlinear_solver; ls = ns.linear_solver
    U = ig.U; asm = ig.asm; n = length(U)
    K_eff_op, M_op_mf = ops
    t_kry = @elapsed begin
        try
            if ls.assembled
                K_eff_sparse = FEC.stiffness(asm)
                ΔU_vec, cg_hist = IterativeSolvers.cg(K_eff_sparse, ig.R_eff;
                    abstol=ns.abs_residual_tol, reltol=ls.rtol, log=true)
                _carina_logf(8, :solve, "    CG: %d iters, converged=%s, |r|_CG=%.3e",
                    length(cg_hist.data[:resnorm]), string(cg_hist.isconverged),
                    cg_hist.data[:resnorm][end])
                copyto!(ig.dU, ΔU_vec)
            else
                if M_op_mf === nothing
                    Krylov.krylov_solve!(ls.workspace, K_eff_op, ig.R_eff;
                        atol=0.0, rtol=ls.rtol, itmax=ls.itmax)
                else
                    Krylov.krylov_solve!(ls.workspace, K_eff_op, ig.R_eff;
                        M=M_op_mf, atol=0.0, rtol=ls.rtol, itmax=ls.itmax)
                end
                copyto!(ig.dU, Krylov.solution(ls.workspace))
            end
        catch
            ig.failed[] = true
            return 0.0, 0.0
        end
    end
    norm_step = sqrt(sum(abs2, ig.dU))
    U .+= ig.dU
    FEC._update_for_assembly!(p, asm.dof, U)
    return norm_step, t_kry
end

# --------------------------------------------------------------------------- #
# _mark_failed_on_nonconvergence — QS fails the step; Newmark accepts best U
# --------------------------------------------------------------------------- #

_mark_failed_on_nonconvergence(::QuasiStaticIntegrator) = true
_mark_failed_on_nonconvergence(::NewmarkIntegrator)     = false

# --------------------------------------------------------------------------- #
# _newton_solve! — unified Newton loop for Direct and Krylov integrators
# --------------------------------------------------------------------------- #

function _newton_solve!(ig, p)
    ns  = ig.nonlinear_solver
    asm = ig.asm

    FEC.update_time!(p)
    FEC.update_bc_values!(p)

    predict!(ig, p)

    # Initial residual at predictor.
    ok = _eval_rhs!(ig, p)
    if !ok
        ig.failed[] = true
        return nothing
    end
    _setup_jacobian!(ig, p) || return nothing

    initial_norm = sqrt(sum(abs2, ig.R_eff))
    _carina_logf(8, :solve, "Iter [0] |R| = %.3e : |r| = %.3e : %s",
                 initial_norm, 1.0, _status_str(false))

    # Create linear operators (noop for Direct; LinearOperators for Krylov).
    ops = _setup_linear_ops(ig, p)

    converged = false
    for iter in 1:ns.max_iters
        # Solve K_eff·ΔU = R_eff and apply U += ΔU.
        norm_step, t_solve = _solve_and_apply!(ig, p, ops)
        ig.failed[] && return nothing

        # Evaluate R_eff at updated U.
        t_eval = @elapsed _eval_rhs!(ig, p)
        norm_R = sqrt(sum(abs2, ig.R_eff))
        rel_R  = initial_norm > 0.0 ? norm_R / initial_norm : norm_R

        converged = norm_step < ns.abs_increment_tol ||
                    norm_R   < ns.abs_residual_tol   ||
                    rel_R    < ns.rel_residual_tol
        _carina_logf(8, :solve,
            "Iter [%d] |R| = %.3e : |r| = %.3e : %s  |ΔU|=%.3e : t_eval=%.3fs : t_solve=%.3fs",
            iter, norm_R, rel_R, _status_str(converged), norm_step, t_eval, t_solve)
        converged && break

        # Reassemble Jacobian only when not yet converged.
        _setup_jacobian!(ig, p) || return nothing
    end

    correct!(ig, p)

    if !converged
        if _mark_failed_on_nonconvergence(ig)
            ig.failed[] = true
        else
            _carina_logf(4, :solve,
                "Newton did not converge in %d iterations; accepting best solution",
                ns.max_iters)
        end
    end

    FEC._update_for_assembly!(p, asm.dof, _displacement(ig))
    p.h1_field_old.data .= p.h1_field.data
    return nothing
end

# --------------------------------------------------------------------------- #
# evolve! — QuasiStatic + Newton + Direct
# --------------------------------------------------------------------------- #

FEC.evolve!(ig::QuasiStaticIntegrator{<:NewtonSolver{DirectLinearSolver}}, p) =
    _newton_solve!(ig, p)

# --------------------------------------------------------------------------- #
# evolve! — QuasiStatic + Newton + Krylov
# --------------------------------------------------------------------------- #

FEC.evolve!(ig::QuasiStaticIntegrator{<:NewtonSolver{<:KrylovLinearSolver}}, p) =
    _newton_solve!(ig, p)

# --------------------------------------------------------------------------- #
# evolve! — QuasiStatic + Newton + LBFGS
# --------------------------------------------------------------------------- #

function FEC.evolve!(ig::QuasiStaticIntegrator{<:NewtonSolver{<:LBFGSLinearSolver}}, p)
    ns  = ig.nonlinear_solver
    ls  = ns.linear_solver
    asm = ig.asm
    dof = asm.dof
    U   = ig.solution

    FEC.update_time!(p)
    FEC.update_bc_values!(p)

    # Reset L-BFGS history each load step.
    ls.head      = 0
    ls.hist_fill = 0

    # Initial residual.
    FEC.assemble_vector!(asm, FEC.residual, U, p)
    FEC.assemble_vector_neumann_bc!(asm, U, p)
    R_int0 = FEC.residual(asm)
    if !isfinite(sqrt(sum(abs2, R_int0)))
        ig.failed[] = true
        return nothing
    end
    @. ls.R_eff = -R_int0

    initial_norm = sqrt(sum(abs2, ls.R_eff))
    _carina_logf(8, :solve, "Iter [0] |R| = %.3e : |r| = %.3e : %s",
                 initial_norm, 1.0, _status_str(false))

    converged = false
    for iter in 1:ns.max_iters
        norm_R = sqrt(sum(abs2, ls.R_eff))
        rel_R  = initial_norm > 0.0 ? norm_R / initial_norm : norm_R

        # L-BFGS descent direction  d = H·R_eff
        t_dir = @elapsed begin
            _lbfgs_two_loop!(ls.d, ls.q, ls.R_eff, ls.S, ls.Y, ls.ρ, ls.alpha_buf,
                              ls.head, ls.hist_fill, ls.m, ls.precond)
        end

        copyto!(ls.R_old, ls.R_eff)

        # Backtracking line search on ‖R(U + α·d)‖.
        α        = 1.0
        ls_iters = 0
        t_ls = @elapsed begin
            for lsi in 1:10
                ls_iters = lsi
                @. ls.q = U + α * ls.d
                FEC.assemble_vector!(asm, FEC.residual, ls.q, p)
                FEC.assemble_vector_neumann_bc!(asm, ls.q, p)
                R_int_trial = FEC.residual(asm)
                @. ls.R_eff = -R_int_trial
                norm_R_trial = sqrt(sum(abs2, ls.R_eff))
                if isfinite(norm_R_trial) && norm_R_trial < norm_R
                    break
                end
                α *= 0.5
            end
        end

        @. U = U + α * ls.d

        norm_dU    = α * sqrt(sum(abs2, ls.d))
        new_norm_R = sqrt(sum(abs2, ls.R_eff))
        new_rel_R  = initial_norm > 0.0 ? new_norm_R / initial_norm : new_norm_R

        converged = norm_dU    < ns.abs_increment_tol ||
                    new_norm_R < ns.abs_residual_tol   ||
                    new_rel_R  < ns.rel_residual_tol

        _carina_logf(8, :solve,
            "Iter [%d] |R| = %.3e : |r| = %.3e : %s\n" *
            "                  |ΔU|=%.3e : α=%.2e : LS=%d : t_dir=%.0fms : t_ls=%.0fms",
            iter, new_norm_R, new_rel_R, _status_str(converged),
            norm_dU, α, ls_iters, t_dir * 1e3, t_ls * 1e3)

        # L-BFGS history update: s_k = α·d,  y_k = R_old − R_eff_new
        new_head = mod1(ls.head + 1, ls.m)
        @. ls.S[new_head] = α * ls.d
        @. ls.Y[new_head] = ls.R_old - ls.R_eff
        ys = dot(ls.Y[new_head], ls.S[new_head])
        if ys > 0.0
            ls.ρ[new_head] = 1.0 / ys
            ls.head        = new_head
            ls.hist_fill   = min(ls.hist_fill + 1, ls.m)
        end

        converged && break
    end

    ig.failed[] = !converged

    FEC._update_for_assembly!(p, dof, U)
    p.h1_field_old.data .= p.h1_field.data

    return nothing
end

# --------------------------------------------------------------------------- #
# evolve! — Newmark + Newton + Direct / Krylov  →  unified _newton_solve!
# --------------------------------------------------------------------------- #
#
# Primary unknown: U (displacement).  A = c_M·(U − U_pred) after Newton.
# Residual:  R_eff = −(F_int(U) − F_ext + c_M·M·(U − U_pred))
# Jacobian:  K_eff = K + c_M·M   (Direct: assembled; Krylov CPU: IterativeSolvers.cg)
#
FEC.evolve!(ig::NewmarkIntegrator{<:NewtonSolver{DirectLinearSolver}}, p) =
    _newton_solve!(ig, p)

FEC.evolve!(ig::NewmarkIntegrator{<:NewtonSolver{<:KrylovLinearSolver}}, p) =
    _newton_solve!(ig, p)

# --------------------------------------------------------------------------- #
# evolve! — Newmark + Newton + LBFGS
# --------------------------------------------------------------------------- #

function FEC.evolve!(ig::NewmarkIntegrator{<:NewtonSolver{<:LBFGSLinearSolver}}, p)
    ns  = ig.nonlinear_solver
    ls  = ns.linear_solver
    asm = ig.asm
    dof = asm.dof
    (; β, γ, α_hht, U, V, A, U_prev, V_prev, A_prev, U_pred) = ig

    FEC.update_time!(p)
    FEC.update_bc_values!(p)

    Δt  = FEC.time_step(p.times)
    c_M = 1.0 / (β * Δt^2)

    # ---- Newmark predictor ----
    copyto!(U_prev, U); copyto!(V_prev, V); copyto!(A_prev, A)
    @. U = U_prev + Δt * V_prev + Δt^2 * (0.5 - β) * A_prev
    @. V = V_prev + Δt * (1.0 - γ) * A_prev
    copyto!(U_pred, U)

    # Reset L-BFGS history each time step.
    ls.head      = 0
    ls.hist_fill = 0

    # ---- Initial residual at predictor ----
    # dU = U − U_pred = 0 at start, so the inertia term vanishes.
    FEC.assemble_vector!(asm, FEC.residual, U, p)
    FEC.assemble_vector_neumann_bc!(asm, U, p)
    R_int0 = FEC.residual(asm)
    if !isfinite(sqrt(sum(abs2, R_int0)))
        ig.failed[] = true
        return nothing
    end
    fill!(ls.M_dU, zero(eltype(ls.M_dU)))   # M·(U_pred − U_pred) = 0
    @. ls.R_eff = -((1 + α_hht) * R_int0 - α_hht * ls.F_int_n)

    initial_norm = sqrt(sum(abs2, ls.R_eff))
    _carina_logf(8, :solve, "Iter [0] |R| = %.3e : |r| = %.3e : %s",
                 initial_norm, 1.0, _status_str(false))

    converged = false
    for iter in 1:ns.max_iters
        norm_R = sqrt(sum(abs2, ls.R_eff))
        rel_R  = initial_norm > 0.0 ? norm_R / initial_norm : norm_R

        # ---- L-BFGS descent direction  d = H·R_eff ----
        t_dir = @elapsed begin
            _lbfgs_two_loop!(ls.d, ls.q, ls.R_eff, ls.S, ls.Y, ls.ρ, ls.alpha_buf,
                              ls.head, ls.hist_fill, ls.m, ls.precond)
        end

        # ---- Precompute M·d for incremental line-search evaluations ----
        t_Md = @elapsed begin
            FEC.assemble_matrix_free_action!(asm, FEC.mass_action, U, ls.d, p)
            copyto!(ls.M_d, FEC.hvp(asm, ls.d))
        end

        # Snapshot R_eff before the step (needed for y = R_old − R_eff_new).
        copyto!(ls.R_old, ls.R_eff)

        # ---- Backtracking line search on ‖R_eff‖ ----
        # Trial residual uses precomputed M·d:
        #   M·(U + step·d − U_pred) = M_dU + step·M_d  (no extra mass assembly)
        step     = 1.0
        ls_iters = 0
        t_ls = @elapsed begin
            for lsi in 1:10
                ls_iters = lsi
                @. ls.q = U + step * ls.d    # trial point (q reused as scratch)
                FEC.assemble_vector!(asm, FEC.residual, ls.q, p)
                FEC.assemble_vector_neumann_bc!(asm, ls.q, p)
                R_int_trial = FEC.residual(asm)
                @. ls.R_eff = -((1 + α_hht) * R_int_trial + c_M * (ls.M_dU + step * ls.M_d) - α_hht * ls.F_int_n)
                norm_R_trial = sqrt(sum(abs2, ls.R_eff))
                if isfinite(norm_R_trial) && norm_R_trial < norm_R
                    break
                end
                step *= 0.5
            end
        end

        # Accept step and maintain M_dU = M·(U_new − U_pred) incrementally.
        @. U = U + step * ls.d
        @. ls.M_dU = ls.M_dU + step * ls.M_d

        norm_dU    = step * sqrt(sum(abs2, ls.d))
        new_norm_R = sqrt(sum(abs2, ls.R_eff))
        new_rel_R  = initial_norm > 0.0 ? new_norm_R / initial_norm : new_norm_R

        converged = norm_dU    < ns.abs_increment_tol ||
                    new_norm_R < ns.abs_residual_tol   ||
                    new_rel_R  < ns.rel_residual_tol

        _carina_logf(8, :solve,
            "Iter [%d] |R| = %.3e : |r| = %.3e : %s\n" *
            "                  |ΔU|=%.3e : step=%.2e : LS=%d : t_dir=%.0fms : t_ls=%.0fms",
            iter, new_norm_R, new_rel_R, _status_str(converged),
            norm_dU, step, ls_iters, (t_dir + t_Md) * 1e3, t_ls * 1e3)
        @debug "Newmark L-BFGS" iter new_norm_R new_rel_R norm_dU

        # ---- L-BFGS history update ----
        # s_k = step·d_k,  y_k = g_{k+1} − g_k = R_old − R_eff_new
        new_head = mod1(ls.head + 1, ls.m)
        @. ls.S[new_head] = step * ls.d
        @. ls.Y[new_head] = ls.R_old - ls.R_eff
        ys = dot(ls.Y[new_head], ls.S[new_head])
        if ys > 0.0
            ls.ρ[new_head] = 1.0 / ys
            ls.head        = new_head
            ls.hist_fill   = min(ls.hist_fill + 1, ls.m)
        end

        converged && break
    end

    ig.failed[] = !converged

    # ---- Newmark velocity and acceleration update ----
    @. A = c_M * (U - U_pred)
    @. V = V + Δt * γ * A

    FEC._update_for_assembly!(p, dof, U)
    p.h1_field_old.data .= p.h1_field.data

    # Store F_int(U_{n+1}) for HHT-α residual in the next time step.
    if α_hht != 0.0 && !ig.failed[]
        FEC.assemble_vector!(asm, FEC.residual, U, p)
        copyto!(ls.F_int_n, FEC.residual(asm))
    end

    return nothing
end

# --------------------------------------------------------------------------- #
# evolve! — CentralDifferenceIntegrator (unchanged)
# --------------------------------------------------------------------------- #

function FEC.evolve!(integrator::CentralDifferenceIntegrator, p)
    (; γ, asm, U, V, A, m_lumped) = integrator

    FEC.update_time!(p)
    FEC.update_bc_values!(p)

    Δt  = FEC.time_step(p.times)
    dof = asm.dof

    @. U = U + Δt * V + 0.5 * Δt^2 * A
    @. V = V + (1.0 - γ) * Δt * A

    FEC.assemble_vector!(asm, FEC.residual, U, p)
    FEC.assemble_vector_neumann_bc!(asm, U, p)
    R = FEC.residual(asm)

    # Detect inverted elements or other constitutive failures (e.g. J ≤ 0 → NaN in
    # neo-Hookean energy) that produce non-finite residual entries.
    if !isfinite(sqrt(sum(abs2, R)))
        integrator.failed[] = true
        return nothing
    end

    @. A = -R / m_lumped

    @. V = V + γ * Δt * A

    integrator.failed[] = false

    FEC._update_for_assembly!(p, dof, U)
    p.h1_field_old.data .= p.h1_field.data

    return nothing
end

# --------------------------------------------------------------------------- #
# State save / restore
# --------------------------------------------------------------------------- #

# QuasiStatic Newton (Direct or Krylov): rollback via h1_field_old
_save_state!(ig::QuasiStaticIntegrator, p) = nothing
function _restore_state!(ig::QuasiStaticIntegrator, p)
    copyto!(p.h1_field.data, p.h1_field_old.data)
    FEC._update_for_assembly!(p, ig.asm.dof, ig.solution)
end

# QuasiStatic LBFGS: save/restore U
function _save_state!(ig::QuasiStaticIntegrator{<:NewtonSolver{<:LBFGSLinearSolver}}, p)
    copyto!(ig.U_save, ig.solution)
end
function _restore_state!(ig::QuasiStaticIntegrator{<:NewtonSolver{<:LBFGSLinearSolver}}, p)
    copyto!(ig.solution, ig.U_save)
    FEC._update_for_assembly!(p, ig.asm.dof, ig.solution)
    p.h1_field_old.data .= p.h1_field.data
end

# Newmark: always save U, V, A
function _save_state!(ig::NewmarkIntegrator, p)
    copyto!(ig.U_save, ig.U)
    copyto!(ig.V_save, ig.V)
    copyto!(ig.A_save, ig.A)
end
function _restore_state!(ig::NewmarkIntegrator, p)
    copyto!(ig.U, ig.U_save)
    copyto!(ig.V, ig.V_save)
    copyto!(ig.A, ig.A_save)
    FEC._update_for_assembly!(p, ig.asm.dof, ig.U)
    p.h1_field_old.data .= p.h1_field.data
end

# CentralDifference: unchanged
function _save_state!(ig::CentralDifferenceIntegrator, p)
    copyto!(ig.U_save, ig.U)
    copyto!(ig.V_save, ig.V)
    copyto!(ig.A_save, ig.A)
end
function _restore_state!(ig::CentralDifferenceIntegrator, p)
    copyto!(ig.U, ig.U_save)
    copyto!(ig.V, ig.V_save)
    copyto!(ig.A, ig.A_save)
    FEC._update_for_assembly!(p, ig.asm.dof, ig.U)
    p.h1_field_old.data .= p.h1_field.data
end

# --------------------------------------------------------------------------- #
# Shared adaptive-stepping helpers
# --------------------------------------------------------------------------- #

function _increase_step!(ig, params)
    ig.increase_factor == 1.0 && return
    new_dt = min(ig.time_step * ig.increase_factor, ig.max_time_step)
    new_dt > ig.time_step && (ig.time_step = new_dt)
end

function _decrease_step!(ig, params)
    ig.decrease_factor == 1.0 &&
        error("Step failed but decrease_factor = 1.0 (adaptive stepping disabled). " *
              "Specify minimum/maximum time step and decrease/increase factors.")
    new_dt = ig.time_step * ig.decrease_factor
    new_dt < ig.min_time_step &&
        error("Cannot reduce time step to $(new_dt): below minimum $(ig.min_time_step).")
    ig.time_step = new_dt
    _carina_logf(0, :recover, "Step failed → reducing Δt to %.3e", new_dt)
end

# --- CFL hook (no-op today; add dispatch for CentralDifferenceIntegrator later) ---
_pre_step_hook!(integrator, params) = nothing
