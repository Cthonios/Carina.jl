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

import FiniteElementContainers as FEC
using LinearAlgebra
import Krylov
import LinearOperators: LinearOperator

# --------------------------------------------------------------------------- #
# QuasiStaticIntegrator
# --------------------------------------------------------------------------- #

mutable struct QuasiStaticIntegrator{Solver <: FEC.NewtonSolver, Vec}
    solver          ::Solver
    solution        ::Vec     # preallocated unknown vector (size = n_unknowns)
    time_step       ::Float64
    min_time_step   ::Float64
    max_time_step   ::Float64
    decrease_factor ::Float64
    increase_factor ::Float64
    failed          ::Base.RefValue{Bool}
end

function QuasiStaticIntegrator(solver::FEC.NewtonSolver,
                                time_step::Float64,
                                min_time_step::Float64,
                                max_time_step::Float64,
                                decrease_factor::Float64,
                                increase_factor::Float64)
    solution = similar(solver.linear_solver.ΔUu)
    fill!(solution, zero(eltype(solution)))
    return QuasiStaticIntegrator(solver, solution, time_step, min_time_step, max_time_step,
                                  decrease_factor, increase_factor, Ref(false))
end

function FEC.evolve!(integrator::QuasiStaticIntegrator, p)
    solver   = integrator.solver
    asm      = solver.linear_solver.assembler
    dof      = asm.dof
    solution = integrator.solution

    FEC.update_time!(p)
    FEC.update_bc_values!(p)

    FEC.solve!(solver, solution, p)

    # Check convergence by inspecting the last Newton increment and residual.
    norm_ΔUu = sqrt(sum(abs2, solver.linear_solver.ΔUu))
    norm_R   = sqrt(sum(abs2, FEC.residual(asm)))

    # Non-finite residual signals a constitutive failure (e.g. J ≤ 0).
    converged = isfinite(norm_R) &&
                (norm_ΔUu < solver.abs_increment_tol ||
                 norm_R   < solver.abs_residual_tol)

    integrator.failed[] = !converged

    FEC._update_for_assembly!(p, dof, solution)
    p.h1_field_old.data .= p.h1_field.data

    return nothing
end

# --------------------------------------------------------------------------- #
# QuasiStaticLBFGSIntegrator — L-BFGS quasi-static
# --------------------------------------------------------------------------- #
#
# Minimises the total potential energy  Π(U) = ∫W(∇u)dV − f·u
# using L-BFGS instead of Newton-Krylov.
#
# Each outer iteration needs only one residual assembly per line-search trial
# (~13 ms on the RX 7600 for 530k DOF). No stiffness matvec is ever called.
#
# History is reset each load step; BB scaling (γ₀ = sy/yy from the last pair)
# provides curvature information once the first step is accepted.

mutable struct QuasiStaticLBFGSIntegrator{Sol, Vec, PC <: Preconditioner}
    solver   ::Sol
    m        ::Int          # L-BFGS history size (ring buffer capacity)
    U        ::Vec          # current unknown vector
    R_eff    ::Vec          # current −R(U)
    R_old    ::Vec          # R_eff snapshot before each step (for y = R_old − R_new)
    d        ::Vec          # L-BFGS descent direction
    q        ::Vec          # two-loop work vector / trial-point scratch
    S        ::Vector{Vec}  # ring-buffer position differences  s_k = α·d_k
    Y        ::Vector{Vec}  # ring-buffer gradient differences  y_k = Δg_k
    ρ        ::Vector{Float64}
    alpha_buf::Vector{Float64}
    head     ::Int
    hist_fill::Int
    precond  ::PC
    # Adaptive time stepping
    time_step       ::Float64
    min_time_step   ::Float64
    max_time_step   ::Float64
    decrease_factor ::Float64
    increase_factor ::Float64
    failed          ::Base.RefValue{Bool}
    # Rollback state
    U_save   ::Vec
end

function QuasiStaticLBFGSIntegrator(solver::Sol, m::Int;
                                     precond::Preconditioner=NoPreconditioner(),
                                     time_step::Float64=0.0,
                                     min_time_step::Float64=0.0,
                                     max_time_step::Float64=0.0,
                                     decrease_factor::Float64=1.0,
                                     increase_factor::Float64=1.0) where {Sol}
    ΔUu = solver.linear_solver.ΔUu
    T   = eltype(ΔUu)
    mk() = (v = similar(ΔUu); fill!(v, zero(T)); v)

    U, R_eff, R_old, d, q, U_save = mk(), mk(), mk(), mk(), mk(), mk()
    S = [mk() for _ in 1:m]
    Y = [mk() for _ in 1:m]
    ρ         = zeros(Float64, m)
    alpha_buf = zeros(Float64, m)

    return QuasiStaticLBFGSIntegrator(
        solver, m,
        U, R_eff, R_old, d, q,
        S, Y, ρ, alpha_buf, 0, 0,
        precond,
        time_step, min_time_step, max_time_step,
        decrease_factor, increase_factor,
        Ref(false),
        U_save,
    )
end

function FEC.evolve!(integrator::QuasiStaticLBFGSIntegrator, p)
    (; solver, m, precond, U, R_eff, R_old, d, q, S, Y, ρ, alpha_buf) = integrator

    FEC.update_time!(p)
    FEC.update_bc_values!(p)

    asm = solver.linear_solver.assembler
    dof = asm.dof

    # Reset L-BFGS history each load step.
    integrator.head      = 0
    integrator.hist_fill = 0

    # Initial residual.
    FEC.assemble_vector!(asm, FEC.residual, U, p)
    FEC.assemble_vector_neumann_bc!(asm, U, p)
    R_int0 = FEC.residual(asm)
    if !isfinite(sqrt(sum(abs2, R_int0)))
        integrator.failed[] = true
        return nothing
    end
    @. R_eff = -R_int0

    initial_norm = sqrt(sum(abs2, R_eff))
    _carina_logf(8, :solve, "Iter [0] |R| = %.3e : |r| = %.3e : %s",
                 initial_norm, 1.0, _status_str(false))

    max_iters   = solver.max_iters
    abs_res_tol = solver.abs_residual_tol
    rel_res_tol = solver.rel_residual_tol
    abs_inc_tol = solver.abs_increment_tol

    converged = false
    for iter in 1:max_iters
        norm_R = sqrt(sum(abs2, R_eff))
        rel_R  = initial_norm > 0.0 ? norm_R / initial_norm : norm_R

        # L-BFGS descent direction  d = H·R_eff
        t_dir = @elapsed begin
            _lbfgs_two_loop!(d, q, R_eff, S, Y, ρ, alpha_buf,
                              integrator.head, integrator.hist_fill, m, precond)
        end

        copyto!(R_old, R_eff)

        # Backtracking line search on ‖R(U + α·d)‖.
        α        = 1.0
        ls_iters = 0
        t_ls = @elapsed begin
            for ls in 1:10
                ls_iters = ls
                @. q = U + α * d
                FEC.assemble_vector!(asm, FEC.residual, q, p)
                FEC.assemble_vector_neumann_bc!(asm, q, p)
                R_int_trial = FEC.residual(asm)
                @. R_eff = -R_int_trial
                norm_R_trial = sqrt(sum(abs2, R_eff))
                if isfinite(norm_R_trial) && norm_R_trial < norm_R
                    break
                end
                α *= 0.5
            end
        end

        @. U = U + α * d

        norm_dU    = α * sqrt(sum(abs2, d))
        new_norm_R = sqrt(sum(abs2, R_eff))
        new_rel_R  = initial_norm > 0.0 ? new_norm_R / initial_norm : new_norm_R

        converged = norm_dU    < abs_inc_tol ||
                    new_norm_R < abs_res_tol  ||
                    new_rel_R  < rel_res_tol

        _carina_logf(8, :solve,
            "Iter [%d] |R| = %.3e : |r| = %.3e : |ΔU| = %.3e : α = %.2e : LS = %d : t_dir = %.0fms : t_ls = %.0fms : %s",
            iter, new_norm_R, new_rel_R, norm_dU, α, ls_iters,
            t_dir * 1e3, t_ls * 1e3, _status_str(converged))

        # L-BFGS history update: s_k = α·d,  y_k = R_old − R_eff_new
        new_head = mod1(integrator.head + 1, m)
        @. S[new_head] = α * d
        @. Y[new_head] = R_old - R_eff
        ys = dot(Y[new_head], S[new_head])
        if ys > 0.0
            ρ[new_head]          = 1.0 / ys
            integrator.head      = new_head
            integrator.hist_fill = min(integrator.hist_fill + 1, m)
        end

        converged && break
    end

    integrator.failed[] = !converged

    FEC._update_for_assembly!(p, dof, U)
    p.h1_field_old.data .= p.h1_field.data

    return nothing
end

# --------------------------------------------------------------------------- #
# NewmarkIntegrator — implicit Newmark-β
# --------------------------------------------------------------------------- #

mutable struct NewmarkIntegrator{Solver <: FEC.NewtonSolver, Vec, KrySolver, PC <: Preconditioner}
    solver::Solver
    β::Float64
    γ::Float64
    use_direct::Bool
    krylov_method::Symbol
    krylov_itmax::Int
    krylov_rtol::Float64
    U::Vec;      V::Vec;      A::Vec
    U_prev::Vec; V_prev::Vec; A_prev::Vec
    krylov_solver::KrySolver
    scratch::Vec
    U_pred::Vec
    dU::Vec
    R_eff::Vec
    precond::PC
    # Adaptive time stepping
    time_step       ::Float64
    min_time_step   ::Float64
    max_time_step   ::Float64
    decrease_factor ::Float64
    increase_factor ::Float64
    failed          ::Base.RefValue{Bool}
    # Rollback state
    U_save::Vec; V_save::Vec; A_save::Vec
end

function NewmarkIntegrator(solver::FEC.NewtonSolver, β::Float64, γ::Float64;
                            use_direct::Bool=false,
                            krylov_method::Symbol=:minres,
                            krylov_itmax::Int=1000,
                            krylov_rtol::Float64=1e-8,
                            precond::Preconditioner=NoPreconditioner(),
                            time_step::Float64=0.0,
                            min_time_step::Float64=0.0,
                            max_time_step::Float64=0.0,
                            decrease_factor::Float64=1.0,
                            increase_factor::Float64=1.0)
    ΔUu = solver.linear_solver.ΔUu
    n   = length(ΔUu)
    T   = eltype(ΔUu)
    S   = typeof(ΔUu)

    mk() = (v = similar(ΔUu); fill!(v, zero(T)); v)

    U, V, A             = mk(), mk(), mk()
    U_prev, V_prev, A_prev = mk(), mk(), mk()
    scratch, U_pred, dU, R_eff = mk(), mk(), mk(), mk()
    U_save, V_save, A_save = mk(), mk(), mk()

    kry = if use_direct
        nothing
    elseif krylov_method == :cg
        Krylov.CgWorkspace(n, n, S)
    else
        Krylov.MinresWorkspace(n, n, S)
    end

    return NewmarkIntegrator(
        solver, β, γ, use_direct, krylov_method, krylov_itmax, krylov_rtol,
        U, V, A, U_prev, V_prev, A_prev,
        kry, scratch, U_pred, dU, R_eff,
        precond,
        time_step, min_time_step, max_time_step, decrease_factor, increase_factor,
        Ref(false),
        U_save, V_save, A_save,
    )
end

# Matrix-free effective stiffness: y = (K + c_M·M)·v
# Uses action kernels (stiffness_action / mass_action) that compute K_q·v and
# M_q·v per quadrature point without forming the 24×24 element matrices,
# eliminating GPU register spilling (Level 2 fix).
function _eff_stiffness_matvec!(y, v, asm, U, c_M, p, scratch)
    FEC.assemble_matrix_free_action!(asm, FEC.stiffness_action, U, v, p)
    copyto!(scratch, asm.stiffness_action_storage.data)
    FEC.assemble_matrix_free_action!(asm, FEC.mass_action, U, v, p)
    @. asm.stiffness_action_storage.data =
        scratch + c_M * asm.stiffness_action_storage.data
    copyto!(y, FEC.hvp(asm, v))
    return y
end

function FEC.evolve!(integrator::NewmarkIntegrator, p)
    (; solver, β, γ, use_direct,
       krylov_method, krylov_itmax, krylov_rtol, precond,
       U, V, A, U_prev, V_prev, A_prev,
       krylov_solver, scratch, U_pred, dU, R_eff) = integrator

    FEC.update_time!(p)
    FEC.update_bc_values!(p)

    Δt  = FEC.time_step(p.times)
    c_M = 1.0 / (β * Δt^2)

    asm = solver.linear_solver.assembler
    dof = asm.dof
    n   = length(U)

    copyto!(U_prev, U); copyto!(V_prev, V); copyto!(A_prev, A)

    @. U = U_prev + Δt * V_prev + Δt^2 * (0.5 - β) * A_prev
    @. V = V_prev + Δt * (1.0 - γ) * A_prev
    copyto!(U_pred, U)

    K_eff_op = if !use_direct
        LinearOperator(
            Float64, n, n, true, true,
            (y, v) -> _eff_stiffness_matvec!(y, v, asm, U, c_M, p, scratch),
        )
    else
        nothing
    end

    M_op = if !use_direct && !(precond isa NoPreconditioner)
        LinearOperator(
            Float64, n, n, true, true,
            (y, v) -> (@. y = precond.inv_diag * v; y),
        )
    else
        nothing
    end

    FEC.assemble_vector!(asm, FEC.residual, U, p)
    FEC.assemble_vector_neumann_bc!(asm, U, p)
    @. dU = U - U_pred
    FEC.assemble_matrix_free_action!(asm, FEC.mass_action, U, dU, p)
    R_int = FEC.residual(asm)

    # Detect constitutive failures (e.g. J ≤ 0 → NaN in neo-Hookean).
    if !isfinite(sqrt(sum(abs2, R_int)))
        integrator.failed[] = true
        return nothing
    end

    M_dU  = FEC.hvp(asm, dU)
    @. R_eff = -(R_int + c_M * M_dU)
    initial_norm = sqrt(sum(abs2, R_eff))
    _carina_logf(8, :solve, "Iter [0] |R| = %.3e : |r| = %.3e : %s",
                 initial_norm, 1.0, _status_str(false))

    converged = false
    for iter in 1:solver.max_iters
        t_asm = @elapsed begin
            FEC.assemble_vector!(asm, FEC.residual, U, p)
            FEC.assemble_vector_neumann_bc!(asm, U, p)

            @. dU = U - U_pred
            FEC.assemble_matrix_free_action!(asm, FEC.mass_action, U, dU, p)
        end

        R_int = FEC.residual(asm)
        M_dU  = FEC.hvp(asm, dU)
        @. R_eff = -(R_int + c_M * M_dU)

        norm_R = sqrt(sum(abs2, R_eff))
        rel_R  = initial_norm > 0.0 ? norm_R / initial_norm : norm_R

        # Effective Krylov rtol.  For Krylov solvers with a preconditioner the
        # atol criterion fires in the *preconditioned* space, which can be orders
        # of magnitude smaller than the physical space.  We therefore set
        # atol = 0 in the Krylov call and rely solely on rtol.  The user-supplied
        # krylov_rtol is the relative convergence target.
        eff_kry_rtol = krylov_rtol

        if use_direct
            t_kry = @elapsed begin
                FEC.assemble_stiffness!(asm, FEC.stiffness, U, p)
                FEC.assemble_mass!(asm, FEC.mass, U, p)
                @. asm.stiffness_storage += c_M * asm.mass_storage
                K_eff_sparse = FEC.stiffness(asm)
                F_lu         = lu(K_eff_sparse)
                ΔUu          = F_lu \ R_eff
            end
            norm_dU   = sqrt(sum(abs2, ΔUu))
            kry_iters = -1
            kry_solved = true
            converged = norm_dU   < solver.abs_increment_tol ||
                        norm_R    < solver.abs_residual_tol   ||
                        rel_R     < solver.rel_residual_tol
            _carina_logf(8, :solve,
                "Iter [%d] |R| = %.3e : |r| = %.3e : |ΔU| = %.3e : t_asm = %.3fs : t_lu = %.3fs : %s",
                iter, norm_R, rel_R, norm_dU, t_asm, t_kry, _status_str(converged))
        else
            t_kry = @elapsed begin
                if M_op === nothing
                    Krylov.krylov_solve!(krylov_solver, K_eff_op, R_eff;
                                  atol=0.0,
                                  rtol=eff_kry_rtol,
                                  itmax=krylov_itmax)
                else
                    Krylov.krylov_solve!(krylov_solver, K_eff_op, R_eff;
                                  M=M_op,
                                  atol=0.0,
                                  rtol=eff_kry_rtol,
                                  itmax=krylov_itmax)
                end
            end
            ΔUu        = Krylov.solution(krylov_solver)
            kry_iters  = krylov_solver.stats.niter
            kry_solved = krylov_solver.stats.solved
            norm_dU    = sqrt(sum(abs2, ΔUu))
            converged = norm_dU   < solver.abs_increment_tol ||
                        norm_R    < solver.abs_residual_tol   ||
                        rel_R     < solver.rel_residual_tol
            _carina_logf(8, :solve,
                "Iter [%d] |R| = %.3e : |r| = %.3e : |ΔU| = %.3e : t_asm = %.3fs : t_kry = %.3fs : Krylov = %d/%d (%s) : %s",
                iter, norm_R, rel_R, norm_dU,
                t_asm, t_kry,
                kry_iters, krylov_itmax, kry_solved ? "conv" : "STALL",
                _status_str(converged))
        end
        @debug "Newmark Newton" iter norm_R rel_R

        U .+= ΔUu

        converged && break
    end

    integrator.failed[] = !converged

    @. A = c_M * (U - U_pred)
    @. V = V + Δt * γ * A

    FEC._update_for_assembly!(p, dof, U)
    p.h1_field_old.data .= p.h1_field.data

    return nothing
end

# --------------------------------------------------------------------------- #
# NewmarkLBFGSIntegrator — L-BFGS implicit Newmark-β
# --------------------------------------------------------------------------- #
#
# Minimises the Newmark energy  Φ(U) = Π_el(U) + (c_M/2)‖U−U_pred‖²_M
# using L-BFGS instead of Newton-Krylov.
#
# Each outer iteration needs:
#   • one residual assembly   (∇Φ = −R_eff)         ≈ 13 ms on GPU
#   • one mass matvec M·d     (for line-search trick) ≈ 124 ms on GPU
#   • per line-search trial: residual assembly only  ≈ 13 ms
#
# No stiffness matvec is ever called, eliminating the GPU register-spill
# bottleneck (573 ms per call) in _assemble_block_matrix_action_kernel!.
#
# Line-search trick: precompute M·d once, then use
#   M·(U + α·d − U_pred) = M·(U − U_pred) + α·M·d
# so trial residuals need only a residual assembly (no extra mass matvec).
# This is exact when M is configuration-independent (standard Lagrangian FEM
# with reference density — valid for neo-Hookean, linear elastic, etc.).

# Two-loop recursion: computes d = H_k · R_eff (L-BFGS descent direction).
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
        # Essential on the first step when c_M = 1/(β·Δt²) ≫ 1 because without
        # dimensional scaling the gradient direction has units of force (N) while
        # the iterate has units of displacement (m), and α=1 overshoots by ~c_M.
        @. d = precond.inv_diag * q
    else
        copyto!(d, q)   # H₀ = I (fallback; only safe when problem is well-scaled)
    end

    # Second loop: oldest → newest (i = hfill → oldest, i = 1 → newest)
    for i in hfill:-1:1
        idx = mod1(head - i + 1, m)
        β = ρ[idx] * dot(Y[idx], d)
        @. d = d + (alpha[i] - β) * S[idx]
    end

    return d
end

mutable struct NewmarkLBFGSIntegrator{Sol, Vec, PC <: Preconditioner}
    solver   ::Sol
    β        ::Float64
    γ        ::Float64
    m        ::Int          # L-BFGS history size (ring buffer capacity)
    U  ::Vec; V  ::Vec; A  ::Vec
    U_prev::Vec; V_prev::Vec; A_prev::Vec
    U_pred   ::Vec           # Newmark predictor position
    R_eff    ::Vec           # current -(R_int + c_M·M·(U−U_pred))
    R_old    ::Vec           # R_eff snapshot before each step (for y update)
    M_dU     ::Vec           # M·(U−U_pred), maintained incrementally each step
    d        ::Vec           # L-BFGS descent direction
    q        ::Vec           # two-loop work vector / trial-point scratch
    M_d      ::Vec           # M·d precomputed for cheap line-search evaluations
    S        ::Vector{Vec}   # ring-buffer position differences  s_k = α·d_k
    Y        ::Vector{Vec}   # ring-buffer gradient differences  y_k = Δg_k
    ρ        ::Vector{Float64}    # curvature scalars  1 / (y·s)
    alpha_buf::Vector{Float64}    # two-loop α scratch (scalar, host)
    head     ::Int            # last-written ring-buffer slot (1-indexed; 0 = empty)
    hist_fill::Int            # number of valid history entries (0..m)
    precond  ::PC             # initial Hessian H₀ (Jacobi recommended for GPU dynamics)
    # Adaptive time stepping
    time_step       ::Float64
    min_time_step   ::Float64
    max_time_step   ::Float64
    decrease_factor ::Float64
    increase_factor ::Float64
    failed          ::Base.RefValue{Bool}
    # Rollback state
    U_save::Vec; V_save::Vec; A_save::Vec
end

function NewmarkLBFGSIntegrator(solver::Sol, β::Float64, γ::Float64, m::Int;
                                 precond::Preconditioner=NoPreconditioner(),
                                 time_step::Float64=0.0,
                                 min_time_step::Float64=0.0,
                                 max_time_step::Float64=0.0,
                                 decrease_factor::Float64=1.0,
                                 increase_factor::Float64=1.0) where {Sol}
    ΔUu = solver.linear_solver.ΔUu
    T   = eltype(ΔUu)
    mk() = (v = similar(ΔUu); fill!(v, zero(T)); v)

    U, V, A                = mk(), mk(), mk()
    U_prev, V_prev, A_prev = mk(), mk(), mk()
    U_pred                 = mk()
    R_eff, R_old           = mk(), mk()
    M_dU, d, q, M_d        = mk(), mk(), mk(), mk()
    U_save, V_save, A_save = mk(), mk(), mk()

    S = [mk() for _ in 1:m]
    Y = [mk() for _ in 1:m]
    ρ         = zeros(Float64, m)
    alpha_buf = zeros(Float64, m)

    return NewmarkLBFGSIntegrator(
        solver, β, γ, m,
        U, V, A, U_prev, V_prev, A_prev,
        U_pred, R_eff, R_old, M_dU, d, q, M_d,
        S, Y, ρ, alpha_buf, 0, 0,
        precond,
        time_step, min_time_step, max_time_step,
        decrease_factor, increase_factor,
        Ref(false),
        U_save, V_save, A_save,
    )
end

function FEC.evolve!(integrator::NewmarkLBFGSIntegrator, p)
    (; solver, β, γ, m, precond,
       U, V, A, U_prev, V_prev, A_prev,
       U_pred, R_eff, R_old, M_dU, d, q, M_d,
       S, Y, ρ, alpha_buf) = integrator

    FEC.update_time!(p)
    FEC.update_bc_values!(p)

    Δt  = FEC.time_step(p.times)
    c_M = 1.0 / (β * Δt^2)

    asm = solver.linear_solver.assembler
    dof = asm.dof

    # ---- Newmark predictor ----
    copyto!(U_prev, U); copyto!(V_prev, V); copyto!(A_prev, A)
    @. U = U_prev + Δt * V_prev + Δt^2 * (0.5 - β) * A_prev
    @. V = V_prev + Δt * (1.0 - γ) * A_prev
    copyto!(U_pred, U)

    # Reset L-BFGS history each time step (warm-start not used; the effective
    # stiffness changes every step so old curvature pairs mislead the search).
    integrator.head      = 0
    integrator.hist_fill = 0

    # ---- Initial residual at predictor ----
    # dU = U − U_pred = 0 at start, so the inertia term vanishes.
    FEC.assemble_vector!(asm, FEC.residual, U, p)
    FEC.assemble_vector_neumann_bc!(asm, U, p)
    R_int0 = FEC.residual(asm)
    if !isfinite(sqrt(sum(abs2, R_int0)))
        integrator.failed[] = true
        return nothing
    end
    fill!(M_dU, zero(eltype(M_dU)))   # M·(U_pred − U_pred) = 0
    @. R_eff = -R_int0                # inertia term is zero at U_pred

    initial_norm = sqrt(sum(abs2, R_eff))
    _carina_logf(8, :solve, "Iter [0] |R| = %.3e : |r| = %.3e : %s",
                 initial_norm, 1.0, _status_str(false))

    max_iters   = solver.max_iters
    abs_res_tol = solver.abs_residual_tol
    rel_res_tol = solver.rel_residual_tol
    abs_inc_tol = solver.abs_increment_tol

    converged = false
    for iter in 1:max_iters
        norm_R = sqrt(sum(abs2, R_eff))
        rel_R  = initial_norm > 0.0 ? norm_R / initial_norm : norm_R

        # ---- L-BFGS descent direction  d = H·R_eff ----
        t_dir = @elapsed begin
            _lbfgs_two_loop!(d, q, R_eff, S, Y, ρ, alpha_buf,
                              integrator.head, integrator.hist_fill, m, precond)
        end

        # ---- Precompute M·d for incremental line-search evaluations ----
        t_Md = @elapsed begin
            FEC.assemble_matrix_free_action!(asm, FEC.mass_action, U, d, p)
            copyto!(M_d, FEC.hvp(asm, d))
        end

        # Snapshot R_eff before the step (needed for y = R_old − R_eff_new).
        copyto!(R_old, R_eff)

        # ---- Backtracking line search on ‖R_eff‖ ----
        # Trial residual at U + α·d uses the precomputed M·d:
        #   M·(U + α·d − U_pred) = M_dU + α·M_d  (no extra mass assembly)
        α        = 1.0
        ls_iters = 0
        t_ls = @elapsed begin
            for ls in 1:10
                ls_iters = ls
                @. q = U + α * d    # trial point (q reused as scratch)
                FEC.assemble_vector!(asm, FEC.residual, q, p)
                FEC.assemble_vector_neumann_bc!(asm, q, p)
                R_int_trial = FEC.residual(asm)
                @. R_eff = -(R_int_trial + c_M * (M_dU + α * M_d))
                norm_R_trial = sqrt(sum(abs2, R_eff))
                if isfinite(norm_R_trial) && norm_R_trial < norm_R
                    break   # sufficient decrease found; R_eff is now up to date
                end
                α *= 0.5
            end
        end

        # Accept step and maintain M_dU = M·(U_new − U_pred) incrementally.
        @. U = U + α * d
        @. M_dU = M_dU + α * M_d

        norm_dU    = α * sqrt(sum(abs2, d))
        new_norm_R = sqrt(sum(abs2, R_eff))
        new_rel_R  = initial_norm > 0.0 ? new_norm_R / initial_norm : new_norm_R

        converged = norm_dU    < abs_inc_tol ||
                    new_norm_R < abs_res_tol  ||
                    new_rel_R  < rel_res_tol

        _carina_logf(8, :solve,
            "Iter [%d] |R| = %.3e : |r| = %.3e : |ΔU| = %.3e : α = %.2e : LS = %d : t_dir = %.0fms : t_Md = %.0fms : t_ls = %.0fms : %s",
            iter, new_norm_R, new_rel_R, norm_dU, α, ls_iters,
            t_dir * 1e3, t_Md * 1e3, t_ls * 1e3, _status_str(converged))
        @debug "Newmark L-BFGS" iter new_norm_R new_rel_R norm_dU

        # ---- L-BFGS history update ----
        # s_k = α·d_k,  y_k = g_{k+1} − g_k = (−R_eff_new) − (−R_old) = R_old − R_eff
        new_head = mod1(integrator.head + 1, m)
        @. S[new_head] = α * d
        @. Y[new_head] = R_old - R_eff
        ys = dot(Y[new_head], S[new_head])
        if ys > 0.0
            ρ[new_head]          = 1.0 / ys
            integrator.head      = new_head
            integrator.hist_fill = min(integrator.hist_fill + 1, m)
        end

        converged && break
    end

    integrator.failed[] = !converged

    # ---- Newmark velocity and acceleration update ----
    @. A = c_M * (U - U_pred)
    @. V = V + Δt * γ * A

    FEC._update_for_assembly!(p, dof, U)
    p.h1_field_old.data .= p.h1_field.data

    return nothing
end

# --------------------------------------------------------------------------- #
# CentralDifferenceIntegrator — explicit central difference
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
# Shared adaptive-stepping helpers
# --------------------------------------------------------------------------- #

# --- State save/restore ---

function _save_state!(ig::NewmarkIntegrator, p)
    copyto!(ig.U_save, ig.U)
    copyto!(ig.V_save, ig.V)
    copyto!(ig.A_save, ig.A)
end

function _restore_state!(ig::NewmarkIntegrator, p)
    copyto!(ig.U, ig.U_save)
    copyto!(ig.V, ig.V_save)
    copyto!(ig.A, ig.A_save)
    dof = ig.solver.linear_solver.assembler.dof
    FEC._update_for_assembly!(p, dof, ig.U)
    p.h1_field_old.data .= p.h1_field.data
end

function _save_state!(ig::NewmarkLBFGSIntegrator, p)
    copyto!(ig.U_save, ig.U)
    copyto!(ig.V_save, ig.V)
    copyto!(ig.A_save, ig.A)
end

function _restore_state!(ig::NewmarkLBFGSIntegrator, p)
    copyto!(ig.U, ig.U_save)
    copyto!(ig.V, ig.V_save)
    copyto!(ig.A, ig.A_save)
    dof = ig.solver.linear_solver.assembler.dof
    FEC._update_for_assembly!(p, dof, ig.U)
    p.h1_field_old.data .= p.h1_field.data
end

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

# QuasiStatic: FEC.solve! writes into solution then syncs to h1_field via
# _update_for_assembly!.  On rollback, copy old field back and reset solution.
_save_state!(ig::QuasiStaticIntegrator, p)    = nothing
function _restore_state!(ig::QuasiStaticIntegrator, p)
    copyto!(p.h1_field.data, p.h1_field_old.data)
    dof = ig.solver.linear_solver.assembler.dof
    FEC._update_for_assembly!(p, dof, ig.solution)
end

function _save_state!(ig::QuasiStaticLBFGSIntegrator, p)
    copyto!(ig.U_save, ig.U)
end
function _restore_state!(ig::QuasiStaticLBFGSIntegrator, p)
    copyto!(ig.U, ig.U_save)
    dof = ig.solver.linear_solver.assembler.dof
    FEC._update_for_assembly!(p, dof, ig.U)
    p.h1_field_old.data .= p.h1_field.data
end

# --- Step size adjustment ---

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
