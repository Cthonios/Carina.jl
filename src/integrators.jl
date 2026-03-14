# Newmark-β time integrator for second-order (dynamic) problems.
#
# GPU-ready design: the effective stiffness K_eff = K + c_M * M is applied
# matrix-free via two calls to FEC.assemble_matrix_action! per Krylov
# iteration.  No sparse matrix is ever materialised.
#
# For CPU-only workflows a direct (sparse LU) path is also available.
# Set use_direct=true and K_eff is assembled as a SparseMatrixCSC, factored
# once per Newton step with UMFPACK, and solved in one backsolve.
#
# Algorithm (displacement-based Newmark-β):
#   Predictor:
#     Ũ = U_n + Δt·V_n + Δt²·(0.5-β)·A_n
#     Ṽ = V_n + Δt·(1-γ)·A_n
#   Newton:
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
# CentralDifferenceIntegrator
# --------------------------------------------------------------------------- #
#
# Explicit Newmark-β with β=0, γ=0.5 (central difference / velocity Verlet).
# No linear solve required — acceleration is computed element-wise from the
# lumped mass and the net force.
#
# Algorithm:
#   Predictor:
#     U_{n+1} = U_n + Δt·V_n + ½·Δt²·A_n
#     V*      = V_n + Δt·(1-γ)·A_n
#   Force assembly:
#     R = f_int(U_{n+1}) - f_ext
#   Acceleration:
#     A_{n+1} = -R / m_lumped   (element-wise)
#   Corrector:
#     V_{n+1} = V* + Δt·γ·A_{n+1}

struct CentralDifferenceIntegrator{Asm, Vec}
    γ::Float64
    asm::Asm          # FEC assembler (needed for force assembly)
    U::Vec            # displacement (full DOF size)
    V::Vec            # velocity
    A::Vec            # acceleration
    m_lumped::Vec     # diagonal lumped mass (row sums of consistent mass)
end

function CentralDifferenceIntegrator(γ::Float64, asm, m_lumped::Vec) where {Vec}
    n  = length(m_lumped)
    T  = eltype(m_lumped)
    mk() = (v = similar(m_lumped); fill!(v, zero(T)); v)
    return CentralDifferenceIntegrator(γ, asm, mk(), mk(), mk(), m_lumped)
end

function FEC.evolve!(integrator::CentralDifferenceIntegrator, p)
    (; γ, asm, U, V, A, m_lumped) = integrator

    FEC.update_time!(p)
    FEC.update_bc_values!(p)

    Δt  = FEC.time_step(p.times)
    dof = asm.dof

    # 1. Predictor
    @. U = U + Δt * V + 0.5 * Δt^2 * A
    @. V = V + (1.0 - γ) * Δt * A

    # 2. Assemble net residual R = f_int - f_ext  (constraint-adjusted)
    FEC.assemble_vector!(asm, FEC.residual, U, p)
    FEC.assemble_vector_neumann_bc!(asm, U, p)
    R = FEC.residual(asm)

    # 3. New acceleration: A = -R / m_lumped  (element-wise)
    @. A = -R / m_lumped

    # 4. Velocity corrector
    @. V = V + γ * Δt * A

    # 5. Sync full field and save old
    FEC._update_for_assembly!(p, dof, U)
    p.h1_field_old.data .= p.h1_field.data

    return nothing
end

# --------------------------------------------------------------------------- #
# NewmarkIntegrator — implicit Newmark-β
# --------------------------------------------------------------------------- #

struct NewmarkIntegrator{Solver <: FEC.NewtonSolver, Vec, KrySolver, PC <: Preconditioner}
    solver::Solver
    β::Float64
    γ::Float64
    use_direct::Bool          # true → sparse LU per Newton step (CPU only)
    krylov_method::Symbol     # :cg or :minres (used when use_direct=false)
    krylov_itmax::Int         # max Krylov iterations per Newton step
    krylov_rtol::Float64      # relative residual tolerance for Krylov solver
    # State vectors (n_total_dofs for condensed DOF manager)
    U::Vec;      V::Vec;      A::Vec
    U_prev::Vec; V_prev::Vec; A_prev::Vec
    # Preallocated scratch to avoid per-iteration allocations
    krylov_solver::KrySolver  # Krylov workspace, or Nothing when use_direct=true
    scratch::Vec               # holds K·v during effective-stiffness matvec
    U_pred::Vec                # Newmark predictor Ũ
    dU::Vec                    # U - Ũ  (inertial RHS scratch)
    R_eff::Vec                 # effective residual (negated for Krylov RHS)
    # Preconditioner (NoPreconditioner or JacobiPreconditioner; unused when use_direct)
    precond::PC
end

function NewmarkIntegrator(solver::FEC.NewtonSolver, β::Float64, γ::Float64;
                            use_direct::Bool=false,
                            krylov_method::Symbol=:minres,
                            krylov_itmax::Int=1000,
                            krylov_rtol::Float64=1e-8,
                            precond::Preconditioner=NoPreconditioner())
    ΔUu = solver.linear_solver.ΔUu
    n   = length(ΔUu)
    T   = eltype(ΔUu)
    S   = typeof(ΔUu)

    mk() = (v = similar(ΔUu); fill!(v, zero(T)); v)

    U, V, A             = mk(), mk(), mk()
    U_prev, V_prev, A_prev = mk(), mk(), mk()
    scratch, U_pred, dU, R_eff = mk(), mk(), mk(), mk()

    # No Krylov workspace needed for the direct path.
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
    )
end

# --------------------------------------------------------------------------- #
# Matrix-free effective stiffness: y = (K + c_M·M)·v  with identity at
# constrained DOFs.
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
    copyto!(y, FEC.hvp(asm, v))
    return y
end

# --------------------------------------------------------------------------- #
# evolve!
# --------------------------------------------------------------------------- #

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

    # 1. Save state at t_n
    copyto!(U_prev, U); copyto!(V_prev, V); copyto!(A_prev, A)

    # 2. Predictor
    @. U = U_prev + Δt * V_prev + Δt^2 * (0.5 - β) * A_prev  # Ũ
    @. V = V_prev + Δt * (1.0 - γ) * A_prev                   # Ṽ
    copyto!(U_pred, U)

    # Matrix-free operator (built once; only used in Krylov path).
    K_eff_op = if !use_direct
        LinearOperator(
            Float64, n, n, true, true,
            (y, v) -> _eff_stiffness_matvec!(y, v, asm, U, c_M, p, scratch),
        )
    else
        nothing
    end

    # Preconditioner operator (Krylov path only).
    M_op = if !use_direct && !(precond isa NoPreconditioner)
        LinearOperator(
            Float64, n, n, true, true,
            (y, v) -> (@. y = precond.inv_diag * v; y),
        )
    else
        nothing
    end

    # 3. Newton iterations — initial residual for [0] log.
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

        @. dU = U - U_pred
        FEC.assemble_matrix_action!(asm, FEC.mass, U, dU, p)

        R_int = FEC.residual(asm)
        M_dU  = FEC.hvp(asm, dU)
        @. R_eff = -(R_int + c_M * M_dU)

        norm_R = sqrt(sum(abs2, R_eff))
        rel_R  = initial_norm > 0.0 ? norm_R / initial_norm : norm_R

        if use_direct
            # ------------------------------------------------------------------
            # Direct sparse LU path (CPU only).
            # Assemble K_eff = K + c_M·M as a sparse matrix, factor, backsolve.
            # assemble_stiffness! overwrites stiffness_storage with K (raw).
            # assemble_mass! overwrites mass_storage with M (raw).
            # Combining them in stiffness_storage before calling FEC.stiffness
            # gives the constraint-adjusted K_eff when sparse! runs.
            # ------------------------------------------------------------------
            FEC.assemble_stiffness!(asm, FEC.stiffness, U, p)
            FEC.assemble_mass!(asm, FEC.mass, U, p)
            @. asm.stiffness_storage += c_M * asm.mass_storage
            K_eff_sparse = FEC.stiffness(asm)   # SparseMatrixCSC, constraint-adjusted
            F_lu         = lu(K_eff_sparse)
            ΔUu          = F_lu \ R_eff
            norm_dU      = sqrt(sum(abs2, ΔUu))
            kry_iters    = -1
        else
            # ------------------------------------------------------------------
            # Matrix-free Krylov path (CG or MINRES; GPU-compatible).
            # ------------------------------------------------------------------
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

    # 4. Corrector
    @. A = c_M * (U - U_pred)
    @. V = V + Δt * γ * A

    # 5. Sync full field and save old
    FEC._update_for_assembly!(p, dof, U)
    p.h1_field_old.data .= p.h1_field.data

    return nothing
end
