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
function _eff_stiffness_matvec!(y, v, asm, U, c_M, p, scratch)
    FEC.assemble_matrix_action!(asm, FEC.stiffness, U, v, p)
    copyto!(scratch, asm.stiffness_action_storage.data)
    FEC.assemble_matrix_action!(asm, FEC.mass, U, v, p)
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
    FEC.assemble_matrix_action!(asm, FEC.mass, U, dU, p)
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
        FEC.assemble_vector!(asm, FEC.residual, U, p)
        FEC.assemble_vector_neumann_bc!(asm, U, p)

        @. dU = U - U_pred
        FEC.assemble_matrix_action!(asm, FEC.mass, U, dU, p)

        R_int = FEC.residual(asm)
        M_dU  = FEC.hvp(asm, dU)
        @. R_eff = -(R_int + c_M * M_dU)

        norm_R = sqrt(sum(abs2, R_eff))
        rel_R  = initial_norm > 0.0 ? norm_R / initial_norm : norm_R

        if use_direct
            FEC.assemble_stiffness!(asm, FEC.stiffness, U, p)
            FEC.assemble_mass!(asm, FEC.mass, U, p)
            @. asm.stiffness_storage += c_M * asm.mass_storage
            K_eff_sparse = FEC.stiffness(asm)
            F_lu         = lu(K_eff_sparse)
            ΔUu          = F_lu \ R_eff
            norm_dU      = sqrt(sum(abs2, ΔUu))
            kry_iters    = -1
        else
            if M_op === nothing
                Krylov.krylov_solve!(krylov_solver, K_eff_op, R_eff;
                              atol=solver.abs_residual_tol,
                              rtol=krylov_rtol,
                              itmax=krylov_itmax)
            else
                Krylov.krylov_solve!(krylov_solver, K_eff_op, R_eff;
                              M=M_op,
                              atol=solver.abs_residual_tol,
                              rtol=krylov_rtol,
                              itmax=krylov_itmax)
            end
            ΔUu       = Krylov.solution(krylov_solver)
            kry_iters = krylov_solver.stats.niter
            norm_dU   = sqrt(sum(abs2, ΔUu))
        end

        U .+= ΔUu

        converged = norm_dU   < solver.abs_increment_tol ||
                    norm_R    < solver.abs_residual_tol   ||
                    rel_R     < solver.rel_residual_tol
        if use_direct
            _carina_logf(8, :solve, "Iter [%d] |R| = %.3e : |r| = %.3e : |ΔU| = %.3e : %s",
                         iter, norm_R, rel_R, norm_dU, _status_str(converged))
        else
            _carina_logf(8, :solve, "Iter [%d] |R| = %.3e : |r| = %.3e : |ΔU| = %.3e : Krylov = %d : %s",
                         iter, norm_R, rel_R, norm_dU, kry_iters, _status_str(converged))
        end
        @debug "Newmark Newton" iter norm_R rel_R

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
