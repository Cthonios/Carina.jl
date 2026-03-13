# Newmark time integrator for second-order (dynamic) problems.
# Self-contained implementation that uses FEC only for assembly.
#
# Algorithm (displacement-based Newmark-β):
#   Predictor (free DOFs):
#     Ũ_u = U_n + Δt * V_n + Δt² * (0.5 - β) * A_n
#     Ṽ_u = V_n + Δt * (1 - γ) * A_n
#   Newton:
#     K_eff = K_int + c_M * M_uu,    c_M = 1/(β Δt²)
#     R_eff = R_int + M_uu * c_M * (Uu - Ũ_u)
#     ΔUu   = -K_eff \ R_eff;   Uu += ΔUu
#   Corrector:
#     A_{n+1} = c_M * (Uu - Ũ_u)
#     V_{n+1} = Ṽ_u + Δt * γ * A_{n+1}

import FiniteElementContainers as FEC
using LinearAlgebra

# --------------------------------------------------------------------------- #
# Struct
# --------------------------------------------------------------------------- #

"""
    NewmarkIntegrator(solver, β, γ)

Second-order Newmark-β integrator.  `solver` must be an `FEC.NewtonSolver`
that wraps an `FEC.DirectLinearSolver` (condensed DOF manager required).

`β = 0.25, γ = 0.5` gives the unconditionally stable average acceleration
method (trapezoidal rule).
"""
struct NewmarkIntegrator{Solver <: FEC.NewtonSolver, Vec <: AbstractVector{Float64}}
    solver::Solver
    β::Float64
    γ::Float64
    # Free-DOF state vectors (size = n_free)
    U::Vec      # displacement at t_n
    V::Vec      # velocity
    A::Vec      # acceleration
    U_prev::Vec # displacement at t_{n-1}
    V_prev::Vec
    A_prev::Vec
end

function NewmarkIntegrator(solver::FEC.NewtonSolver, β::Float64, γ::Float64)
    n = length(solver.linear_solver.ΔUu)
    U = zeros(n); V = zeros(n); A = zeros(n)
    return NewmarkIntegrator(solver, β, γ, U, V, A, copy(U), copy(V), copy(A))
end

# --------------------------------------------------------------------------- #
# evolve!
# --------------------------------------------------------------------------- #

function FEC.evolve!(integrator::NewmarkIntegrator, p)
    (; solver, β, γ, U, V, A, U_prev, V_prev, A_prev) = integrator

    FEC.update_time!(p)
    FEC.update_bc_values!(p)

    Δt  = FEC.time_step(p.times)
    c_M = 1.0 / (β * Δt^2)

    # 1. Save state at t_n
    copyto!(U_prev, U)
    copyto!(V_prev, V)
    copyto!(A_prev, A)

    # 2. Newmark predictor (free DOFs)
    @. U = U_prev + Δt * V_prev + Δt^2 * (0.5 - β) * A_prev   # Ũ_u
    V_pred = @. V_prev + Δt * (1.0 - γ) * A_prev               # Ṽ_u (temp)
    U_pred = copy(U)                                             # keep Ũ_u

    asm = solver.linear_solver.assembler
    dof = asm.dof

    # 3. Newton iterations
    for iter in 1:solver.max_iters
        FEC.assemble_vector!(asm, FEC.residual, U, p)
        FEC.assemble_vector_neumann_bc!(asm, U, p)
        FEC.assemble_stiffness!(asm, FEC.stiffness, U, p)
        FEC.assemble_mass!(asm, FEC.mass, U, p)

        R_int = FEC.residual(asm)
        K_int = copy(FEC.stiffness(asm))  # copy: stiffness/mass share pattern.cscnzval
        M_uu  = FEC.mass(asm)

        R_eff = R_int .+ M_uu * (c_M .* (U .- U_pred))
        K_eff = K_int .+ c_M .* M_uu

        ΔUu = -K_eff \ R_eff
        U  .+= ΔUu

        if norm(ΔUu) < solver.abs_increment_tol && norm(R_eff) < solver.abs_residual_tol
            break
        end
    end

    # 4. Corrector
    @. A = c_M * (U - U_pred)
    @. V = V_pred + Δt * γ * A

    # 5. Sync full field
    FEC._update_for_assembly!(p, dof, U)
    p.h1_field_old.data .= p.h1_field.data

    return nothing
end
