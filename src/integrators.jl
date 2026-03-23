# Time integrators for Carina.
#
# 3-axis separation of concerns:
#   Integrators   — predict!, evaluate!, setup_jacobian!, apply_increment!, residual, correct!
#   Nonlinear     — solve!(ns, ig, p)
#   Linear        — _linear_solve!(ls, ig, p, ops) → (ΔU, t_solve)
#
# A single generic FEC.evolve!(ig, p) works for ALL integrators.
#
# Three integrators:
#   QuasiStaticIntegrator       — pseudo-time quasi-static Newton (no inertia)
#   NewmarkIntegrator           — implicit Newmark-β (GPU-ready matrix-free + direct LU)
#   CentralDifferenceIntegrator — explicit central difference (GPU-ready)
#
# Solver factoring:
#   AbstractLinearSolver    — DirectLinearSolver | KrylovLinearSolver | LBFGSLinearSolver | NoLinearSolver
#   AbstractNonlinearSolver — NewtonSolver{LS} | ExplicitSolver
#   Integrators use nonlinear_solver(ig) to dispatch to the right solver.
#
# Adaptive stepping protocol:
#   • time_step / min/max_time_step / decrease/increase_factor
#   • failed — Ref{Bool} set by FEC.evolve! to signal non-convergence
#   • _save_state! / _restore_state! — rollback on step failure
#   • _increase_step! / _decrease_step! — adjust time_step
#   • _pre_step_hook! — called before each sub-step (CFL update, no-op today)

import FiniteElementContainers as FEC
using LinearAlgebra
import Krylov
import LinearOperators: LinearOperator
import IterativeSolvers
import LimitedLDLFactorizations: lldl

# --------------------------------------------------------------------------- #
# Abstract solver types
# --------------------------------------------------------------------------- #

abstract type AbstractLinearSolver end
abstract type AbstractNonlinearSolver end

struct ExplicitSolver <: AbstractNonlinearSolver end

# --------------------------------------------------------------------------- #
# Concrete linear solver types
# --------------------------------------------------------------------------- #

struct DirectLinearSolver <: AbstractLinearSolver end

mutable struct KrylovLinearSolver{KW, Vec} <: AbstractLinearSolver
    itmax    ::Int
    rtol     ::Float64
    assembled::Bool           # true = CPU sparse K_eff; false = GPU matrix-free
    precond  ::Preconditioner
    workspace::KW
    ones_v   ::Vec
    scratch  ::Vec
end

# LBFGSLinearSolver: ring-buffer and scratch vectors only.
# R_eff and F_int_n removed — they are integrator state, not solver state.
mutable struct LBFGSLinearSolver{Vec, PC <: Preconditioner} <: AbstractLinearSolver
    m         ::Int
    precond   ::PC
    S         ::Vector{Vec}
    Y         ::Vector{Vec}
    ρ         ::Vector{Float64}
    alpha_buf ::Vector{Float64}
    head      ::Int
    hist_fill ::Int
    R_old     ::Vec   # snapshot for y = R_old − R_new
    d         ::Vec   # descent direction
    q         ::Vec   # two-loop work / trial scratch
    M_d       ::Vec   # Newmark: M·d precomputed for line search
    M_dU      ::Vec   # Newmark: M·(U−U_pred) maintained incrementally
end

struct NoLinearSolver <: AbstractLinearSolver end

# --------------------------------------------------------------------------- #
# Newton nonlinear solver
# --------------------------------------------------------------------------- #

mutable struct NewtonSolver{LS <: AbstractLinearSolver} <: AbstractNonlinearSolver
    min_iters         ::Int
    max_iters         ::Int
    abs_increment_tol ::Float64
    abs_residual_tol  ::Float64
    rel_residual_tol  ::Float64
    linear_solver     ::LS
    # Line search parameters
    use_line_search   ::Bool
    ls_backtrack      ::Float64   # step reduction factor (default 0.5)
    ls_decrease       ::Float64   # Armijo sufficient decrease (default 1e-4)
    ls_max_iters      ::Int       # max backtracking steps (default 10)
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
    F_int_n::Vec   # HHT-α: F_int at t_n (zeroed when α_hht=0)
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
    F_int_n             = mk()
    U_save, V_save, A_save = mk(), mk(), mk()

    return NewmarkIntegrator(ns, asm, β, γ, α_hht,
                              U, V, A, U_prev, V_prev, A_prev,
                              U_pred, dU, R_eff, F_int_n,
                              0.0,   # c_M initialised to 0; set by predict!
                              time_step, min_time_step, max_time_step,
                              decrease_factor, increase_factor,
                              Ref(false),
                              U_save, V_save, A_save)
end

# --------------------------------------------------------------------------- #
# CentralDifferenceIntegrator
# --------------------------------------------------------------------------- #

mutable struct CentralDifferenceIntegrator{Asm, Vec}
    γ::Float64
    asm::Asm
    U::Vec; V::Vec; A::Vec
    m_lumped::Vec
    R_eff   ::Vec   # effective residual = -R_int
    # Adaptive time stepping
    time_step       ::Float64
    min_time_step   ::Float64
    max_time_step   ::Float64
    decrease_factor ::Float64
    increase_factor ::Float64
    # CFL stable time step
    CFL                ::Float64
    stable_dt_interval ::Int      # steps between recomputation (0 = init only)
    stable_dt_counter  ::Int      # steps since last recomputation
    failed             ::Base.RefValue{Bool}
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
                                      stable_dt_interval::Int=0) where {Vec}
    T  = eltype(m_lumped)
    mk() = (v = similar(m_lumped); fill!(v, zero(T)); v)
    U, V, A = mk(), mk(), mk()
    R_eff   = mk()
    U_save, V_save, A_save = mk(), mk(), mk()
    return CentralDifferenceIntegrator(
        γ, asm, U, V, A, m_lumped, R_eff,
        time_step, min_time_step, max_time_step, decrease_factor, increase_factor,
        CFL, stable_dt_interval, 0,
        Ref(false),
        U_save, V_save, A_save,
    )
end

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
# Integrator interface: predict!, evaluate!, setup_jacobian!,
#   apply_increment!, residual, correct!, nonlinear_solver, _finalize_step!
# --------------------------------------------------------------------------- #

# ---- predict! ----

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

function predict!(ig::CentralDifferenceIntegrator, p)
    Δt = FEC.time_step(p.times)
    @. ig.U += Δt * ig.V + 0.5 * Δt^2 * ig.A
    @. ig.V += (1.0 - ig.γ) * Δt * ig.A
end

# ---- evaluate! ----
# Assembles R_eff = -(force residual), stored in ig.R_eff. Returns isfinite.

function evaluate!(ig::QuasiStaticIntegrator, p)
    U = ig.solution; asm = ig.asm
    FEC.assemble_vector!(asm, FEC.residual, U, p)
    FEC.assemble_vector_neumann_bc!(asm, U, p)
    R = FEC.residual(asm)
    @. ig.R_eff = -R
    return isfinite(sqrt(sum(abs2, ig.R_eff)))
end

function evaluate!(ig::NewmarkIntegrator, p)
    (; asm, U, U_pred, dU, R_eff, c_M, α_hht, F_int_n) = ig
    @. dU = U - U_pred
    FEC.assemble_vector!(asm, FEC.residual, U, p)
    FEC.assemble_vector_neumann_bc!(asm, U, p)
    FEC.assemble_matrix_free_action!(asm, FEC.mass_action, U, dU, p)
    R_int = FEC.residual(asm)
    M_dU  = FEC.hvp(asm, dU)
    @. R_eff = -((1 + α_hht) * R_int + c_M * M_dU - α_hht * F_int_n)
    return isfinite(sqrt(sum(abs2, R_eff)))
end

function evaluate!(ig::CentralDifferenceIntegrator, p)
    asm = ig.asm; U = ig.U
    FEC.assemble_vector!(asm, FEC.residual, U, p)
    FEC.assemble_vector_neumann_bc!(asm, U, p)
    R_int = FEC.residual(asm)
    @. ig.R_eff = -R_int
    return isfinite(sqrt(sum(abs2, ig.R_eff)))
end

# ---- setup_jacobian! ----
# Assemble K_eff (or update precond) at current U.
# Returns true on success, false on exception (may set ig.failed[]).

function setup_jacobian!(ig::QuasiStaticIntegrator{<:NewtonSolver{DirectLinearSolver}}, p)
    FEC.assemble_stiffness!(ig.asm, FEC.stiffness, ig.solution, p)
    return true
end

function setup_jacobian!(ig::NewmarkIntegrator{<:NewtonSolver{DirectLinearSolver}}, p)
    asm = ig.asm; U = ig.U; c_M = ig.c_M
    FEC.assemble_stiffness!(asm, FEC.stiffness, U, p)
    FEC.assemble_mass!(asm, FEC.mass, U, p)
    @. asm.stiffness_storage += c_M * asm.mass_storage
    return true
end

function setup_jacobian!(ig::QuasiStaticIntegrator{<:NewtonSolver{<:KrylovLinearSolver}}, p)
    ls = ig.nonlinear_solver.linear_solver; U = ig.solution; asm = ig.asm
    if ls.assembled
        FEC.assemble_stiffness!(asm, FEC.stiffness, U, p)
        _update_jacobi_precond_assembled!(ls.precond, FEC.stiffness(asm))
    else
        _update_jacobi_precond_qs!(ls.precond, asm, U, ls.ones_v, p)
    end
    return true
end

function setup_jacobian!(ig::NewmarkIntegrator{<:NewtonSolver{<:KrylovLinearSolver}}, p)
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

# LBFGS and CentralDifference: no Jacobian needed
setup_jacobian!(ig::QuasiStaticIntegrator{<:NewtonSolver{<:LBFGSLinearSolver}}, p) = true
setup_jacobian!(ig::NewmarkIntegrator{<:NewtonSolver{<:LBFGSLinearSolver}}, p)     = true
setup_jacobian!(ig::CentralDifferenceIntegrator, p)                                 = true

# ---- apply_increment! ----

function apply_increment!(ig, ΔU, p)
    U = _displacement(ig)
    U .+= ΔU
    FEC._update_for_assembly!(p, ig.asm.dof, U)
end

# ---- residual accessor ----

residual(ig) = ig.R_eff

# ---- correct! ----

correct!(::QuasiStaticIntegrator, p) = nothing

function correct!(ig::NewmarkIntegrator, p)
    Δt = FEC.time_step(p.times)
    @. ig.A = ig.c_M * (ig.U - ig.U_pred)
    @. ig.V += Δt * ig.γ * ig.A
    return nothing
end

function correct!(ig::CentralDifferenceIntegrator, p)
    Δt = FEC.time_step(p.times)
    @. ig.V += ig.γ * Δt * ig.A
end

# ---- nonlinear_solver ----

nonlinear_solver(ig) = ig.nonlinear_solver
nonlinear_solver(::CentralDifferenceIntegrator) = ExplicitSolver()

# ---- _finalize_step! ----

function _finalize_step!(ig, p)
    FEC._update_for_assembly!(p, ig.asm.dof, _displacement(ig))
    p.field_old.data .= p.field.data
    # Promote converged state: state_old ← state_new
    p.state_old.data .= p.state_new.data
end

function _finalize_step!(ig::NewmarkIntegrator, p)
    FEC._update_for_assembly!(p, ig.asm.dof, ig.U)
    p.field_old.data .= p.field.data
    p.state_old.data .= p.state_new.data
    if ig.α_hht != 0.0 && !ig.failed[]
        FEC.assemble_vector!(ig.asm, FEC.residual, ig.U, p)
        copyto!(ig.F_int_n, FEC.residual(ig.asm))
    end
end

# ---- _mark_failed_on_nonconvergence ----

_mark_failed_on_nonconvergence(::QuasiStaticIntegrator) = true
_mark_failed_on_nonconvergence(::NewmarkIntegrator)     = true

# ---- _displacement ----

_displacement(ig::QuasiStaticIntegrator)        = ig.solution
_displacement(ig::NewmarkIntegrator)            = ig.U
_displacement(ig::CentralDifferenceIntegrator)  = ig.U

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
    _carina_logf(8, :solve, "    CG: %d iters : |r|_CG = %.3e : %s",
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
                _carina_logf(8, :solve, "    CG: %d iters : |r|_CG = %.3e : %s",
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
                _carina_logf(8, :solve, "    CG: %d iters : |r|_CG = %.3e : %s",
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
    R_int_trial = FEC.residual(asm)
    @. ig.R_eff = -R_int_trial
end

function _lbfgs_trial_rhs!(ig::NewmarkIntegrator, ls, step, p)
    α_hht = ig.α_hht; c_M = ig.c_M
    @. ls.q = ig.U + step * ls.d
    FEC.assemble_vector!(ig.asm, FEC.residual, ls.q, p)
    FEC.assemble_vector_neumann_bc!(ig.asm, ls.q, p)
    R_int_trial = FEC.residual(ig.asm)
    @. ig.R_eff = -((1 + α_hht) * R_int_trial + c_M * (ls.M_dU + step * ls.M_d) - α_hht * ig.F_int_n)
end

# --------------------------------------------------------------------------- #
# Backtracking line search
# --------------------------------------------------------------------------- #
#
# Given Newton direction ΔU, find step length α ∈ (0, 1] such that the
# merit function m(α) = 0.5 * ‖R(U + α*ΔU)‖² satisfies the Armijo
# sufficient decrease condition:
#   m(α) ≤ m(0) + c * α * m'(0)
# where c = ls_decrease (typically 1e-4) and m'(0) = -‖R‖² (Newton direction
# is the steepest descent direction for the merit function).
#
# If the Armijo condition is not met, α is reduced by ls_backtrack (default 0.5).
# Returns the accepted α.

function _backtrack_line_search(ns::NewtonSolver, ig, p, ΔU)
    α = 1.0
    merit_0 = sum(abs2, residual(ig))  # ‖R(U)‖² (before step)
    U = _displacement(ig)
    U_save = copy(U)
    state_new_save = copy(p.state_new.data)

    for ls_iter in 1:ns.ls_max_iters
        # Trial point: U + α * ΔU
        @. U = U_save + α * ΔU
        FEC._update_for_assembly!(p, ig.asm.dof, U)
        evaluate!(ig, p)

        merit = sum(abs2, residual(ig))

        # Armijo condition: sufficient decrease in merit
        if merit ≤ (1.0 - 2.0 * ns.ls_decrease * α) * merit_0
            _carina_logf(8, :linesearch, "    LS: α = %.3e : m = %.3e → %.3e : [ACCEPT]",
                         α, merit_0, merit)
            return α
        end

        _carina_logf(8, :linesearch, "    LS: α = %.3e : m = %.3e → %.3e : [REDUCE]",
                     α, merit_0, merit)
        α *= ns.ls_backtrack
    end

    # Max iterations reached — restore original state and accept full step
    copyto!(U, U_save)
    copyto!(p.state_new.data, state_new_save)
    FEC._update_for_assembly!(p, ig.asm.dof, U)
    evaluate!(ig, p)
    _carina_logf(8, :linesearch, "    LS: max iters reached, using α = 1.0")
    return 1.0
end

# --------------------------------------------------------------------------- #
# Nonlinear solvers
# --------------------------------------------------------------------------- #

# ---- Newton (all linear solvers except LBFGS) ----

function solve!(ns::NewtonSolver, ig, p)
    evaluate!(ig, p) || (ig.failed[] = true; return)
    setup_jacobian!(ig, p) || return
    initial_norm = sqrt(sum(abs2, residual(ig)))
    _carina_logf(8, :solve, "Iter [0] |R| = %.3e : |r| = %.3e : %s",
                 initial_norm, 1.0, _status_str(false))
    ops = _setup_linear_ops(ig, ns.linear_solver, p)
    converged = false
    for iter in 1:ns.max_iters
        ΔU, t_solve = _linear_solve!(ns.linear_solver, ig, p, ops)
        ig.failed[] && return

        if ns.use_line_search
            α = _backtrack_line_search(ns, ig, p, ΔU)
            # Line search already applied the step and evaluated the residual
            norm_step = α * sqrt(sum(abs2, ΔU))
            t_eval = 0.0  # already done inside line search
        else
            apply_increment!(ig, ΔU, p)
            t_eval = @elapsed evaluate!(ig, p)
            norm_step = sqrt(sum(abs2, ΔU))
        end

        norm_R    = sqrt(sum(abs2, residual(ig)))
        rel_R     = initial_norm > 0.0 ? norm_R / initial_norm : norm_R
        met_tol   = norm_step < ns.abs_increment_tol ||
                    norm_R   < ns.abs_residual_tol   ||
                    rel_R    < ns.rel_residual_tol
        converged = met_tol && iter > ns.min_iters
        _carina_logf(8, :solve,
            "Iter [%d] |R| = %.3e : |r| = %.3e : |ΔU| = %.3e : t_eval = %.2fs : t_solve = %.2fs : %s",
            iter, norm_R, rel_R, norm_step, t_eval, t_solve, _status_str(converged))
        converged && break
        setup_jacobian!(ig, p) || return
    end
    if !converged
        if _mark_failed_on_nonconvergence(ig)
            ig.failed[] = true
        else
            _carina_logf(4, :solve,
                "Newton did not converge in %d iterations; accepting best solution",
                ns.max_iters)
        end
    end
end

# ---- LBFGS ----

function solve!(ns::NewtonSolver{<:LBFGSLinearSolver}, ig, p)
    ls = ns.linear_solver
    ls.head = 0; ls.hist_fill = 0
    _lbfgs_init_M_dU!(ig, ls)
    evaluate!(ig, p) || (ig.failed[] = true; return)
    initial_norm = sqrt(sum(abs2, residual(ig)))
    _carina_logf(8, :solve, "Iter [0] |R| = %.3e : |r| = %.3e : %s",
                 initial_norm, 1.0, _status_str(false))
    converged = false
    for iter in 1:ns.max_iters
        norm_R = sqrt(sum(abs2, residual(ig)))
        rel_R  = initial_norm > 0.0 ? norm_R / initial_norm : norm_R
        # L-BFGS descent direction
        t_dir = @elapsed _lbfgs_two_loop!(ls.d, ls.q, residual(ig), ls.S, ls.Y, ls.ρ,
                                           ls.alpha_buf, ls.head, ls.hist_fill, ls.m, ls.precond)
        _lbfgs_precompute_M_d!(ig, ls, p)
        copyto!(ls.R_old, residual(ig))
        # Backtracking line search
        step = 1.0; ls_iters = 0
        t_ls = @elapsed for lsi in 1:10
            ls_iters = lsi
            _lbfgs_trial_rhs!(ig, ls, step, p)
            norm_R_trial = sqrt(sum(abs2, residual(ig)))
            isfinite(norm_R_trial) && norm_R_trial < norm_R && break
            step *= 0.5
        end
        # Accept step
        U = _displacement(ig)
        @. U = U + step * ls.d
        FEC._update_for_assembly!(p, ig.asm.dof, U)
        _lbfgs_update_M_dU!(ig, ls, step)
        norm_dU    = step * sqrt(sum(abs2, ls.d))
        new_norm_R = sqrt(sum(abs2, residual(ig)))
        new_rel_R  = initial_norm > 0.0 ? new_norm_R / initial_norm : new_norm_R
        met_tol   = norm_dU    < ns.abs_increment_tol ||
                    new_norm_R < ns.abs_residual_tol   ||
                    new_rel_R  < ns.rel_residual_tol
        converged = met_tol && iter > ns.min_iters
        _carina_logf(8, :solve,
            "Iter [%d] |R| = %.3e : |r| = %.3e : |ΔU| = %.3e : step = %.2e : LS = %d : t_dir = %.0fms : t_ls = %.0fms : %s",
            iter, new_norm_R, new_rel_R,
            norm_dU, step, ls_iters, t_dir*1e3, t_ls*1e3, _status_str(converged))
        # History update
        new_head = mod1(ls.head + 1, ls.m)
        R_cur = residual(ig)
        @. ls.S[new_head] = step * ls.d
        @. ls.Y[new_head] = ls.R_old - R_cur
        ys = dot(ls.Y[new_head], ls.S[new_head])
        if ys > 0.0
            ls.ρ[new_head] = 1.0 / ys
            ls.head        = new_head
            ls.hist_fill   = min(ls.hist_fill + 1, ls.m)
        end
        converged && break
    end
    ig.failed[] = !converged
end

# ---- ExplicitSolver ----

function solve!(::ExplicitSolver, ig::CentralDifferenceIntegrator, p)
    evaluate!(ig, p) || (ig.failed[] = true; return)
    R_eff = residual(ig)
    @. ig.A = R_eff / ig.m_lumped
    ig.failed[] = false
end

# --------------------------------------------------------------------------- #
# Generic FEC.evolve! — works for ALL integrators
# --------------------------------------------------------------------------- #

function FEC.evolve!(ig, p)
    FEC.update_time!(p)
    FEC.update_bc_values!(p)
    predict!(ig, p)
    solve!(nonlinear_solver(ig), ig, p)
    ig.failed[] && return nothing
    correct!(ig, p)
    _finalize_step!(ig, p)
    return nothing
end

# --------------------------------------------------------------------------- #
# State save / restore
# --------------------------------------------------------------------------- #

# QuasiStatic: always save/restore ig.solution for adaptive stepping rollback.
function _save_state!(ig::QuasiStaticIntegrator, p)
    copyto!(ig.U_save, ig.solution)
end
function _restore_state!(ig::QuasiStaticIntegrator, p)
    copyto!(ig.solution, ig.U_save)
    copyto!(p.field.data, p.field_old.data)
    # Reset state_new from state_old so retried Newton starts from clean state
    p.state_new.data .= p.state_old.data
    FEC._update_for_assembly!(p, ig.asm.dof, ig.solution)
end

# Newmark and CentralDifference: save/restore U, V, A
const _DynamicIntegrator = Union{NewmarkIntegrator, CentralDifferenceIntegrator}

function _save_state!(ig::_DynamicIntegrator, p)
    copyto!(ig.U_save, ig.U)
    copyto!(ig.V_save, ig.V)
    copyto!(ig.A_save, ig.A)
end
function _restore_state!(ig::_DynamicIntegrator, p)
    copyto!(ig.U, ig.U_save)
    copyto!(ig.V, ig.V_save)
    copyto!(ig.A, ig.A_save)
    FEC._update_for_assembly!(p, ig.asm.dof, ig.U)
    p.field_old.data .= p.field.data
    p.state_new.data .= p.state_old.data
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

# --- CFL stable time step hook ---
_pre_step_hook!(integrator, sim) = nothing

function _pre_step_hook!(ig::CentralDifferenceIntegrator, sim)
    ig.CFL <= 0.0 && return
    ig.stable_dt_interval <= 0 && return
    ig.stable_dt_counter += 1
    ig.stable_dt_counter < ig.stable_dt_interval && return
    ig.stable_dt_counter = 0
    stable_dt = _compute_stable_dt(ig.asm, sim.params, ig.CFL)
    ig.time_step = min(stable_dt, ig.max_time_step)
end
