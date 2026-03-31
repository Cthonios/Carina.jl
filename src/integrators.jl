# Time integrators for Carina.
#
# 3-axis separation of concerns:
#   Integrators   — predict!, evaluate!, setup_jacobian!, apply_increment!, residual, correct!
#   Nonlinear     — solve!(ns, ig, p)           [nonlinear_solvers.jl]
#   Linear        — _linear_solve!(ls, ig, p, ops) [linear_solvers.jl]
#
# A single generic FEC.evolve!(ig, p) works for ALL integrators.
#
# Three integrators:
#   QuasiStaticIntegrator       — pseudo-time quasi-static Newton (no inertia)
#   NewmarkIntegrator           — implicit Newmark-β (GPU-ready matrix-free + direct LU)
#   CentralDifferenceIntegrator — explicit central difference (GPU-ready)
#
# Adaptive stepping protocol:
#   • time_step / min/max_time_step / decrease/increase_factor
#   • failed — Ref{Bool} set by FEC.evolve! to signal non-convergence
#   • _save_state! / _restore_state! — rollback on step failure
#   • _increase_step! / _decrease_step! — adjust time_step
#   • _pre_step_hook! — called before each sub-step (CFL update, no-op today)

import FiniteElementContainers as FEC
using LinearAlgebra

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
    initial_equilibrium::Bool  # solve for equilibrium at t₀ before time stepping
end

function QuasiStaticIntegrator(ns::NS, asm, template::Vec;
                                time_step::Float64=0.0,
                                min_time_step::Float64=0.0,
                                max_time_step::Float64=0.0,
                                decrease_factor::Float64=1.0,
                                increase_factor::Float64=1.0,
                                initial_equilibrium::Bool=true) where {NS, Vec}
    T  = eltype(template)
    mk() = (v = similar(template); fill!(v, zero(T)); v)
    solution = mk()
    U_save   = mk()
    R_eff    = mk()
    return QuasiStaticIntegrator(ns, asm, solution, time_step, min_time_step, max_time_step,
                                  decrease_factor, increase_factor, Ref(false), U_save, R_eff,
                                  initial_equilibrium)
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
    FEC.assemble_vector_source!(asm, U, p)
    R = FEC.residual(asm)
    @. ig.R_eff = -R
    return isfinite(sqrt(sum(abs2, ig.R_eff)))
end

function evaluate!(ig::NewmarkIntegrator, p)
    (; asm, U, U_pred, dU, R_eff, c_M, α_hht, F_int_n) = ig
    @. dU = U - U_pred
    FEC.assemble_vector!(asm, FEC.residual, U, p)
    FEC.assemble_vector_neumann_bc!(asm, U, p)
    FEC.assemble_vector_source!(asm, U, p)
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
    FEC.assemble_vector_source!(asm, U, p)
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
setup_jacobian!(ig::QuasiStaticIntegrator{<:SteepestDescentSolver}, p)              = true
setup_jacobian!(ig::NewmarkIntegrator{<:SteepestDescentSolver}, p)                  = true

# ---- apply_increment! ----

function apply_increment!(ig, ΔU, p)
    U = _displacement(ig)
    U .+= ΔU
    FEC._update_for_assembly!(p, ig.asm.dof, U)
end

# ---- _displacement ----

_displacement(ig::QuasiStaticIntegrator)        = ig.solution
_displacement(ig::NewmarkIntegrator)            = ig.U
_displacement(ig::CentralDifferenceIntegrator)  = ig.U

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

# ---- nonlinear_solver ----

nonlinear_solver(ig) = ig.nonlinear_solver
nonlinear_solver(::CentralDifferenceIntegrator) = ExplicitSolver()

# ---- _mark_failed_on_nonconvergence ----

_mark_failed_on_nonconvergence(::QuasiStaticIntegrator) = true
_mark_failed_on_nonconvergence(::NewmarkIntegrator)     = true

# --------------------------------------------------------------------------- #
# const _DynamicIntegrator
# --------------------------------------------------------------------------- #

const _DynamicIntegrator = Union{NewmarkIntegrator, CentralDifferenceIntegrator}

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
    _carina_logf(0, :recover, "Step failed → reducing Δt to %.2e", new_dt)
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
