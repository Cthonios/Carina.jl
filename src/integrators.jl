# Newmark-β time integrator for second-order (dynamic) problems.
#
# GPU-ready design: the effective stiffness K_eff = K + c_M * M is applied
# matrix-free via two calls to FEC.assemble_matrix_action! per Krylov
# iteration.  No sparse matrix is ever materialised.
#
# Algorithm (displacement-based Newmark-β):
#   Predictor:
#     Ũ = U_n + Δt·V_n + Δt²·(0.5-β)·A_n
#     Ṽ = V_n + Δt·(1-γ)·A_n
#   Newton (matrix-free MINRES):
#     R_eff = R_int + c_M·M·(U - Ũ)      c_M = 1/(β·Δt²)
#     K_eff·ΔU = -R_eff                  K_eff = K + c_M·M
#     U += ΔU
#   Corrector:
#     A_{n+1} = c_M·(U - Ũ)
#     V_{n+1} = Ṽ + Δt·γ·A_{n+1}

import FiniteElementContainers as FEC
using LinearAlgebra
import Krylov
import LinearOperators: LinearOperator

# --------------------------------------------------------------------------- #
# Struct
# --------------------------------------------------------------------------- #

struct NewmarkIntegrator{Solver <: FEC.NewtonSolver, Vec, KrySolver}
    solver::Solver
    β::Float64
    γ::Float64
    krylov_itmax::Int     # max MINRES iterations per Newton step
    krylov_rtol::Float64  # relative residual tolerance for MINRES
    # State vectors (n_total_dofs for condensed DOF manager)
    U::Vec;      V::Vec;      A::Vec
    U_prev::Vec; V_prev::Vec; A_prev::Vec
    # Preallocated scratch to avoid per-iteration allocations
    krylov_solver::KrySolver  # reused across Newton iterations
    scratch::Vec               # holds K·v during effective-stiffness matvec
    U_pred::Vec                # Newmark predictor Ũ
    dU::Vec                    # U - Ũ  (inertial RHS scratch)
    R_eff::Vec                 # effective residual (negated for Krylov RHS)
end

function NewmarkIntegrator(solver::FEC.NewtonSolver, β::Float64, γ::Float64;
                            krylov_itmax::Int=1000, krylov_rtol::Float64=1e-8)
    ΔUu = solver.linear_solver.ΔUu
    n   = length(ΔUu)
    T   = eltype(ΔUu)
    S   = typeof(ΔUu)

    mk() = (v = similar(ΔUu); fill!(v, zero(T)); v)

    U, V, A             = mk(), mk(), mk()
    U_prev, V_prev, A_prev = mk(), mk(), mk()
    scratch, U_pred, dU, R_eff = mk(), mk(), mk(), mk()

    kry = Krylov.MinresWorkspace(n, n, S)
    return NewmarkIntegrator(
        solver, β, γ, krylov_itmax, krylov_rtol,
        U, V, A, U_prev, V_prev, A_prev,
        kry, scratch, U_pred, dU, R_eff,
    )
end

# --------------------------------------------------------------------------- #
# Matrix-free effective stiffness: y = (K + c_M·M)·v  with identity at
# constrained DOFs.
#
# Writes into asm.stiffness_action_storage (shared buffer); safe because
# this function is only called during the Krylov inner loop where no other
# code touches that storage.
# --------------------------------------------------------------------------- #

function _eff_stiffness_matvec!(y, v, asm, U, c_M, p, scratch)
    # K·v (raw, no constraint adjustment yet)
    FEC.assemble_matrix_action!(asm, FEC.stiffness, U, v, p)
    copyto!(scratch, asm.stiffness_action_storage.data)

    # M·v (raw)
    FEC.assemble_matrix_action!(asm, FEC.mass, U, v, p)

    # K·v + c_M·M·v  in stiffness_action_storage
    @. asm.stiffness_action_storage.data =
        scratch + c_M * asm.stiffness_action_storage.data

    # Apply constraint adjustment once: (I-G)·(K_eff·v) + G·v
    # FEC.hvp modifies stiffness_action_storage.data in-place and returns it.
    copyto!(y, FEC.hvp(asm, v))
    return y
end

# --------------------------------------------------------------------------- #
# evolve!
# --------------------------------------------------------------------------- #

function FEC.evolve!(integrator::NewmarkIntegrator, p)
    (; solver, β, γ,
       U, V, A, U_prev, V_prev, A_prev,
       krylov_solver, scratch, U_pred, dU, R_eff) = integrator

    FEC.update_time!(p)
    FEC.update_bc_values!(p)

    Δt  = FEC.time_step(p.times)
    c_M = 1.0 / (β * Δt^2)

    asm = solver.linear_solver.assembler
    dof = asm.dof
    n   = length(U)

    # 1. Save state at t_n
    copyto!(U_prev, U); copyto!(V_prev, V); copyto!(A_prev, A)

    # 2. Predictor
    @. U = U_prev + Δt * V_prev + Δt^2 * (0.5 - β) * A_prev  # Ũ
    @. V = V_prev + Δt * (1.0 - γ) * A_prev                   # Ṽ  (V = V_pred)
    copyto!(U_pred, U)

    # Build matrix-free operator once per time step.
    # The closure captures U by reference: in-place U .+= ΔUu keeps it current.
    K_eff_op = LinearOperator(
        Float64, n, n, true, true,
        (y, v) -> _eff_stiffness_matvec!(y, v, asm, U, c_M, p, scratch),
    )

    # 3. Newton iterations
    # Assemble the initial residual for iteration [0] log (before any Newton step).
    FEC.assemble_vector!(asm, FEC.residual, U, p)
    FEC.assemble_vector_neumann_bc!(asm, U, p)
    @. dU = U - U_pred
    FEC.assemble_matrix_action!(asm, FEC.mass, U, dU, p)
    R_int = FEC.residual(asm)
    M_dU  = FEC.hvp(asm, dU)
    @. R_eff = -(R_int + c_M * M_dU)
    initial_norm = sqrt(sum(abs2, R_eff))
    _carina_logf(8, :solve, "Iter [0] |R| = %.3e : |r| = %.3e : %s",
                 initial_norm, 1.0, _status_str(false))

    for iter in 1:solver.max_iters
        # Assemble residual
        FEC.assemble_vector!(asm, FEC.residual, U, p)
        FEC.assemble_vector_neumann_bc!(asm, U, p)

        # Inertial RHS:  c_M · M·(U - Ũ)
        @. dU = U - U_pred
        FEC.assemble_matrix_action!(asm, FEC.mass, U, dU, p)
        # hvp applies: (I-G)·M·dU + G·dU.  For constrained DOFs dU[i]=0 → 0. ✓

        # R_eff = R_int + c_M·M·dU   (negated for Krylov: K_eff·ΔU = -R_eff)
        R_int = FEC.residual(asm)
        M_dU  = FEC.hvp(asm, dU)
        @. R_eff = -(R_int + c_M * M_dU)

        norm_R = sqrt(sum(abs2, R_eff))
        rel_R  = initial_norm > 0.0 ? norm_R / initial_norm : norm_R

        Krylov.krylov_solve!(krylov_solver, K_eff_op, R_eff;
                      atol=solver.abs_residual_tol,
                      rtol=integrator.krylov_rtol,
                      itmax=integrator.krylov_itmax)
        ΔUu       = Krylov.solution(krylov_solver)
        kry_iters = krylov_solver.stats.niter
        norm_dU   = sqrt(sum(abs2, ΔUu))

        U .+= ΔUu

        converged = sqrt(sum(abs2, ΔUu)) < solver.abs_increment_tol ||
                    norm_R < solver.abs_residual_tol                 ||
                    rel_R  < solver.rel_residual_tol
        _carina_logf(8, :solve, "Iter [%d] |R| = %.3e : |r| = %.3e : |ΔU| = %.3e : Krylov = %d : %s",
                     iter, norm_R, rel_R, norm_dU, kry_iters, _status_str(converged))
        @debug "Newmark Newton" iter norm_R rel_R kry_iters

        converged && break
    end

    # 4. Corrector
    @. A = c_M * (U - U_pred)
    @. V = V + Δt * γ * A          # V = Ṽ + Δt·γ·A_{n+1}

    # 5. Sync full field and save old
    FEC._update_for_assembly!(p, dof, U)
    p.h1_field_old.data .= p.h1_field.data

    return nothing
end
