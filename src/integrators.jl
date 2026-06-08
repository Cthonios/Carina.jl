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
using StaticArrays

# --------------------------------------------------------------------------- #
# Assembly caching — module-level state to avoid changing integrator struct
# layouts (which would trigger GPU kernel recompilation).
#
# _asm_flags controls whether stiffness/mass assembly is skipped (cached).
# _K_cache / _M_cache store sparse-matrix value arrays for the Newmark
# assembled path (K_eff = K + c_M·M overwrites stiffness_storage in place).
# _factorization_cache holds a cached LU or IC factorization (Any-typed to
# avoid parametric struct changes).
#
# Initialized by _init_assembly_cache!() called from simulation.jl.
# --------------------------------------------------------------------------- #

const _asm_flags = AssemblyFlags()
const _K_cache   = Float64[]
const _M_cache   = Float64[]
const _factorization_cache = Ref{Any}(nothing)
const _precond_op_cache    = Ref{Any}(nothing)
const _gpu_cholesky_L      = Ref{Any}(nothing)  # GPU sparse lower triangular factor
const _cpu_asm_ref         = Ref{Any}(nothing)  # CPU assembler reference for GPU Cholesky
const _cpu_params_ref      = Ref{Any}(nothing)  # CPU params reference for GPU Cholesky
const _backend_ref         = Ref{KA.Backend}(KA.CPU())  # active KernelAbstractions backend
const _nonlinear_status_test = Ref{Any}(nothing)  # parsed termination criteria

# Math errors from constitutive models (e.g. J2 plasticity raising a negative
# number to a fractional power) that should be treated as evaluation failures
# rather than hard crashes.
const _MATH_ERRORS = Union{DomainError, InexactError, OverflowError}

function _init_assembly_cache!(asm, is_linear::Bool)
    _asm_flags.compute_stiffness     = true
    _asm_flags.compute_mass          = true
    _asm_flags.compute_factorization = true
    _asm_flags.is_linear             = is_linear
    _asm_flags.c_M_cached            = 0.0
    _factorization_cache[]           = nothing
    _precond_op_cache[]              = nothing
    _gpu_cholesky_L[]                = nothing
    _cpu_asm_ref[]                   = nothing
    _cpu_params_ref[]                = nothing
    empty!(_K_cache)
    empty!(_M_cache)
    # Matrix-free assemblers (central difference) leave these buffers empty.
    if hasproperty(asm, :stiffness_storage) && !isempty(asm.stiffness_storage)
        append!(_K_cache, asm.stiffness_storage)
        append!(_M_cache, asm.mass_storage)
    end
    return nothing
end

# --------------------------------------------------------------------------- #
# Point loads (Neumann BCs on node sets) — module-level state
# --------------------------------------------------------------------------- #

const _point_loads = PointLoad[]
const _point_load_coords = Ref{Vector{Float64}}(Float64[])

function _init_point_loads!(loads::Vector{PointLoad}, coords::AbstractVector{Float64})
    empty!(_point_loads)
    append!(_point_loads, loads)
    _point_load_coords[] = Vector{Float64}(coords)
    return nothing
end

# Add point load contributions to the residual vector R_eff (sign: positive = external force).
function _apply_point_loads!(R_eff, t::Float64)
    isempty(_point_loads) && return
    X = _point_load_coords[]
    for pl in _point_loads
        pl.unk_idx == 0 && continue   # constrained DOF, skip
        coords = SVector{3, Float64}(X[(pl.node-1)*3+1], X[(pl.node-1)*3+2], X[(pl.node-1)*3+3])
        R_eff[pl.unk_idx] += pl.func(coords, t)
    end
    return nothing
end

# --------------------------------------------------------------------------- #
# QuasiStaticIntegrator{NS, Vec}
# --------------------------------------------------------------------------- #

mutable struct QuasiStaticIntegrator{NS <: AbstractNonlinearSolver, Asm, Vec}
    nonlinear_solver ::NS
    asm              ::Asm
    U                ::Vec   # free-DOF displacement accumulator
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
                                initial_equilibrium::Bool=false) where {NS, Vec}
    T  = eltype(template)
    mk() = (v = similar(template); fill!(v, zero(T)); v)
    U        = mk()
    U_save   = mk()
    R_eff    = mk()
    return QuasiStaticIntegrator(ns, asm, U, time_step, min_time_step, max_time_step,
                                  decrease_factor, increase_factor, Ref(false), U_save, R_eff,
                                  initial_equilibrium)
end

# --------------------------------------------------------------------------- #
# NewmarkIntegrator{NS, Vec}
# --------------------------------------------------------------------------- #

mutable struct NewmarkIntegrator{NS <: AbstractNonlinearSolver, Asm, Vec}
    nonlinear_solver ::NS
    asm              ::Asm
    β                ::Float64
    γ                ::Float64
    α_hht            ::Float64
    # ---- Norma-shape full-DOF integrator state ----
    # U, V, A are full-DOF (length n_total = length(asm.dof)):
    #   free DOFs  carry the Newmark-evolved displacement/velocity/
    #              acceleration.  Updated each step by predict!/correct!/
    #              apply_increment!.
    #   BC   DOFs  carry the prescribed values g(t), g'(t), g''(t) from
    #              p.dirichlet_bcs.bc_cache, written each step by
    #              predict! via FEC.update_field_dirichlet_bcs!(U, V, A,
    #              bcs).  This mirrors Norma.jl's apply_bc, which writes
    #              all three quantities at every step.
    # The full-DOF representation is what makes the inertial residual
    # M·A naturally include the M_{f,BC}·g''(t) cross-term: the
    # element kernel reads u_el and v_el from full-DOF fields populated
    # with merged free+BC content, and produces the action with both
    # blocks correctly coupled.  Free-DOF storage cannot express the
    # BC state at all, so V[BC] / A[BC] would silently be zero — wrong
    # for any consumer that reads them (output, energy, analytical
    # comparison).
    U   ::Vec; V   ::Vec; A   ::Vec
    U_prev::Vec; V_prev::Vec; A_prev::Vec
    U_pred ::Vec
    dU     ::Vec   # full-DOF: U − U_pred. dU[BC] = g''(t)/c_M, see predict!
    R_eff  ::Vec   # free-DOF Newton residual
    F_int_n::Vec   # free-DOF: F_int at t_n (HHT-α; zeroed when α_hht=0)
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
    T = eltype(template)
    n_full = length(asm.dof)
    mk_full() = (v = similar(template, n_full); fill!(v, zero(T)); v)
    mk_free() = (v = similar(template);         fill!(v, zero(T)); v)

    U, V, A                  = mk_full(), mk_full(), mk_full()
    U_prev, V_prev, A_prev   = mk_full(), mk_full(), mk_full()
    U_pred                   = mk_full()
    dU                       = mk_full()
    R_eff                    = mk_free()
    F_int_n                  = mk_free()
    U_save, V_save, A_save   = mk_full(), mk_full(), mk_full()

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
    # ---- Norma-shape full-DOF integrator state ----
    # See the matching comment on NewmarkIntegrator.U/V/A above for why
    # all three are full-DOF.  Even though M is lumped here and the
    # M_{f,BC} cross-term is structurally zero, V/A still need to carry
    # the prescribed g'(t)/g''(t) at BC nodes so that output, energy
    # accounting and post-processing at the boundary read the correct
    # values rather than silent zeros.
    U::Vec; V::Vec; A::Vec
    m_lumped::Vec   # free-DOF lumped mass (a = R_eff/m_lumped uses free slice)
    R_eff   ::Vec   # free-DOF effective residual = -R_int
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
    # Rollback state (full-DOF)
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
    T = eltype(m_lumped)
    n_full = length(asm.dof)
    mk_full() = (v = similar(m_lumped, n_full); fill!(v, zero(T)); v)
    mk_free() = (v = similar(m_lumped);         fill!(v, zero(T)); v)
    U, V, A                = mk_full(), mk_full(), mk_full()
    R_eff                  = mk_free()
    U_save, V_save, A_save = mk_full(), mk_full(), mk_full()
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
    free  = ig.asm.dof.unknown_dofs
    # Save full-DOF previous state (free + BC at t_n) before mutating.
    copyto!(U_prev, U); copyto!(V_prev, V); copyto!(A_prev, A)
    # Newmark predict on free slots only.
    @views @. U[free] = U_prev[free] + Δt * V_prev[free] +
                         Δt^2 * (0.5 - β) * A_prev[free]
    @views @. V[free] = V_prev[free] + Δt * (1.0 - γ) * A_prev[free]
    # Apply Dirichlet BCs at t_{n+1}:
    #   U[BC] = g(t_{n+1}),  V[BC] = g'(t_{n+1}),  A[BC] = g''(t_{n+1}).
    # bc_cache was already advanced to t_{n+1} in evolve!.
    FEC.update_field_dirichlet_bcs!(U, V, A, p.dirichlet_bcs)
    # Build full-DOF predictor.  Free slots: the Newmark predictor we
    # just computed (a[free] = 0 implies U[free] = U_pred[free] at the
    # start of Newton).  BC slots: engineered so that the corrector
    # relation A = c_M·(U − U_pred) yields A[BC] = g''(t_{n+1}) exactly,
    # i.e., U_pred[BC] = U[BC] − g''(t_{n+1})/c_M.  Then dU[BC] =
    # g''(t_{n+1})/c_M and the inertial term c_M·M·dU evaluated at the
    # full-DOF v_full agrees with Norma's `M · integrator.acceleration`
    # on every row.
    copyto!(U_pred, U)
    c_M_inv = 1.0 / ig.c_M
    cache = p.dirichlet_bcs.bc_cache
    FEC.fec_foreach(cache.dofs) do I
        d = cache.dofs[I]
        U_pred[d] = U[d] - c_M_inv * cache.vals_dot_dot[I]
    end
    fill!(dU, zero(eltype(dU)))
    return nothing
end

function predict!(ig::CentralDifferenceIntegrator, p)
    Δt   = FEC.time_step(p.times)
    free = ig.asm.dof.unknown_dofs
    # Standard CD predict on free slots only.
    @views @. ig.U[free] += Δt * ig.V[free] + 0.5 * Δt^2 * ig.A[free]
    @views @. ig.V[free] += (1.0 - ig.γ) * Δt * ig.A[free]
    # Apply prescribed BC values at t_{n+1} so subsequent assembly and
    # post-processing see g(t)/g'(t)/g''(t) at constrained nodes.
    FEC.update_field_dirichlet_bcs!(ig.U, ig.V, ig.A, p.dirichlet_bcs)
    return nothing
end

# ---- evaluate! ----
# Assembles R_eff = -(force residual), stored in ig.R_eff. Returns isfinite.

function evaluate!(ig::QuasiStaticIntegrator, p)
    U = ig.U; asm = ig.asm
    try
        FEC.assemble_vector!(asm, FEC.residual, U, p)
        FEC.assemble_vector_neumann_bc!(asm, U, p)
        FEC.assemble_vector_source!(asm, U, p)
    catch e
        e isa _MATH_ERRORS || rethrow()
        _carina_logf(4, :solve, "evaluate!: caught %s during assembly", typeof(e))
        return false
    end
    R = FEC.residual(asm)
    @. ig.R_eff = -R
    _apply_point_loads!(ig.R_eff, FEC.current_time(p.times))
    return isfinite(sqrt(sum(abs2, ig.R_eff)))
end

function evaluate!(ig::NewmarkIntegrator, p)
    (; asm, U, U_pred, dU, R_eff, c_M, α_hht, F_int_n) = ig
    Uu = view(U, asm.dof.unknown_dofs)
    # dU is full-DOF: dU[free] = U[free] − U_pred[free] (Newmark corrector
    # increment, = β·Δt²·A[free]), and dU[BC] = g''(t_{n+1})/c_M from the
    # U_pred construction in predict!.  So c_M·dU = A everywhere.
    @. dU = U - U_pred
    try
        FEC.assemble_vector!(asm, FEC.residual, Uu, p)
        FEC.assemble_vector_neumann_bc!(asm, Uu, p)
        FEC.assemble_vector_source!(asm, Uu, p)
        # Inertial residual: M · A on free rows, evaluated via M · (c_M·dU)
        # with full-DOF dU.  The full-DOF action carries M_{f,BC}·dU_BC
        # = M_{f,BC}·g''(t_{n+1})/c_M, so c_M times the result includes
        # the M_{f,BC}·g''(t_{n+1}) cross-term that mirrors Norma's
        # `model.mass * integrator.acceleration` on the free rows.
        FEC.assemble_matrix_free_action_full!(
            asm, FEC.mass_action, U, dU, p
        )
    catch e
        e isa _MATH_ERRORS || rethrow()
        _carina_logf(4, :solve, "evaluate!: caught %s during assembly", typeof(e))
        return false
    end
    R_int = FEC.residual(asm)
    M_dU  = FEC.hvp(asm, Uu)
    @. R_eff = -((1 + α_hht) * R_int + c_M * M_dU - α_hht * F_int_n)
    _apply_point_loads!(R_eff, FEC.current_time(p.times))
    return isfinite(sqrt(sum(abs2, R_eff)))
end

function evaluate!(ig::CentralDifferenceIntegrator, p)
    asm = ig.asm
    Uu = view(ig.U, asm.dof.unknown_dofs)
    try
        FEC.assemble_vector!(asm, FEC.residual, Uu, p)
        FEC.assemble_vector_neumann_bc!(asm, Uu, p)
        FEC.assemble_vector_source!(asm, Uu, p)
    catch e
        e isa _MATH_ERRORS || rethrow()
        _carina_logf(4, :solve, "evaluate!: caught %s during assembly", typeof(e))
        return false
    end
    R_int = FEC.residual(asm)
    @. ig.R_eff = -R_int
    _apply_point_loads!(ig.R_eff, FEC.current_time(p.times))
    return isfinite(sqrt(sum(abs2, ig.R_eff)))
end

# ---- setup_jacobian! ----
# Assemble K_eff (or update precond) at current U.
# Returns true on success, false on exception (may set ig.failed[]).

function setup_jacobian!(ig::QuasiStaticIntegrator{<:NewtonSolver{DirectLinearSolver}}, p)
    af = _asm_flags
    if af.compute_stiffness
        FEC.assemble_stiffness!(ig.asm, FEC.stiffness, ig.U, p)
        af.is_linear && (af.compute_stiffness = false)
    end
    return true
end

function setup_jacobian!(ig::NewmarkIntegrator{<:NewtonSolver{DirectLinearSolver}}, p)
    asm = ig.asm; Uu = _displacement(ig); c_M = ig.c_M; af = _asm_flags
    if af.compute_stiffness
        FEC.assemble_stiffness!(asm, FEC.stiffness, Uu, p)
        if af.is_linear
            copyto!(_K_cache, asm.stiffness_storage)
            af.compute_stiffness = false
        end
        af.compute_factorization = true
    else
        copyto!(asm.stiffness_storage, _K_cache)
    end
    if af.compute_mass
        FEC.assemble_mass!(asm, FEC.mass, Uu, p)
        copyto!(_M_cache, asm.mass_storage)
        af.compute_mass = false
        af.compute_factorization = true
    else
        copyto!(asm.mass_storage, _M_cache)
    end
    # c_M changes with adaptive time stepping → K_eff changes → refactorize
    if c_M != af.c_M_cached
        af.c_M_cached = c_M
        af.compute_factorization = true
    end
    @. asm.stiffness_storage += c_M * asm.mass_storage
    return true
end

function setup_jacobian!(ig::QuasiStaticIntegrator{<:NewtonSolver{<:KrylovLinearSolver}}, p)
    ls = ig.nonlinear_solver.linear_solver; U = ig.U; asm = ig.asm
    af = _asm_flags
    if ls.assembled
        if af.compute_stiffness
            FEC.assemble_stiffness!(asm, FEC.stiffness, U, p)
            af.is_linear && (af.compute_stiffness = false)
        end
        _update_jacobi_precond_assembled!(ls.precond, FEC.stiffness(asm))
        _update_chebyshev_precond_assembled!(ls.precond, FEC.stiffness(asm))
    else
        # Matrix-free path: update preconditioner from true diag(K) via
        # the diagonal extraction kernel.  For linear elastic, cache after
        # first call (K is constant).
        if !af.is_linear || af.compute_factorization
            _update_jacobi_precond_qs!(ls.precond, asm, U, ls.ones_v, p)
            _update_chebyshev_precond_qs!(ls.precond, asm, U, p)
            af.is_linear && (af.compute_factorization = false)
        end
    end
    return true
end

function setup_jacobian!(ig::NewmarkIntegrator{<:NewtonSolver{<:KrylovLinearSolver}}, p)
    ls = ig.nonlinear_solver.linear_solver; asm = ig.asm
    Uu = _displacement(ig); c_M = ig.c_M
    af = _asm_flags
    if ls.assembled
        try
            if af.compute_stiffness
                FEC.assemble_stiffness!(asm, FEC.stiffness, Uu, p)
                if af.is_linear
                    copyto!(_K_cache, asm.stiffness_storage)
                    af.compute_stiffness = false
                end
                af.compute_factorization = true
            else
                copyto!(asm.stiffness_storage, _K_cache)
            end
            if af.compute_mass
                FEC.assemble_mass!(asm, FEC.mass, Uu, p)
                copyto!(_M_cache, asm.mass_storage)
                af.compute_mass = false
                af.compute_factorization = true
            else
                copyto!(asm.mass_storage, _M_cache)
            end
            if c_M != af.c_M_cached
                af.c_M_cached = c_M
                af.compute_factorization = true
            end
            @. asm.stiffness_storage += c_M * asm.mass_storage
            _update_jacobi_precond_assembled!(ls.precond, FEC.stiffness(asm))
            _update_chebyshev_precond_assembled!(ls.precond, FEC.stiffness(asm))
        catch e
            e isa _MATH_ERRORS || rethrow()
            _carina_logf(4, :solve, "setup_jacobian!: caught %s", typeof(e))
            ig.failed[] = true
            return false
        end
    else
        # Matrix-free path: update from true diag(K_eff) via diagonal kernel.
        # For linear elastic with constant dt, cache after first call.
        if !af.is_linear || af.compute_factorization
            _update_jacobi_precond_eff!(ls.precond, asm, Uu, ls.ones_v, c_M, p, ls.scratch)
            _update_chebyshev_precond_eff!(ls.precond, asm, Uu, c_M, p, ls.scratch)
            af.is_linear && (af.compute_factorization = false)
        end
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
# For QuasiStaticIntegrator: ig.U is free-DOF, return it as-is.
# For dynamic integrators (Newmark, CD): ig.U is full-DOF, return a
# free-DOF view via dof.unknown_dofs.  Downstream FEC assembly calls
# (assemble_vector!, assemble_stiffness!, ...) want Uu sized n_unknown
# so they can scatter into p.field via _update_for_assembly!.

_displacement(ig::QuasiStaticIntegrator) = ig.U
_displacement(ig::NewmarkIntegrator)           =
    view(ig.U, ig.asm.dof.unknown_dofs)
_displacement(ig::CentralDifferenceIntegrator) =
    view(ig.U, ig.asm.dof.unknown_dofs)

# ---- residual accessor ----

residual(ig) = ig.R_eff

# ---- correct! ----

correct!(::QuasiStaticIntegrator, p) = nothing

# Free-DOF corrector for dynamic integrators.  BC slots of V, A are
# already correct (set by predict! via FEC.update_field_dirichlet_bcs!
# to g'(t_{n+1}) and g''(t_{n+1})); leaving them alone preserves the
# prescribed values exactly, whereas applying the Newmark formula to
# BC slots would re-derive them through `c_M·(U[BC] − U_pred[BC])` and
# pick up a 1-ULP discrepancy from the engineered U_pred[BC].
function correct!(ig::NewmarkIntegrator, p)
    Δt = FEC.time_step(p.times)
    free = ig.asm.dof.unknown_dofs
    @views @. ig.A[free] = ig.c_M * (ig.U[free] - ig.U_pred[free])
    @views @. ig.V[free] += Δt * ig.γ * ig.A[free]
    return nothing
end

function correct!(ig::CentralDifferenceIntegrator, p)
    Δt = FEC.time_step(p.times)
    free = ig.asm.dof.unknown_dofs
    @views @. ig.V[free] += ig.γ * Δt * ig.A[free]
    return nothing
end

# ---- _finalize_step! ----

function _finalize_step!(ig, p)
    FEC._update_for_assembly!(p, ig.asm.dof, _displacement(ig))
    p.field_old.data .= p.field.data
    # Promote converged state: state_old ← state_new
    p.state_old.data .= p.state_new.data
end

function _finalize_step!(ig::NewmarkIntegrator, p)
    Uu = _displacement(ig)
    FEC._update_for_assembly!(p, ig.asm.dof, Uu)
    p.field_old.data .= p.field.data
    p.state_old.data .= p.state_new.data
    if ig.α_hht != 0.0 && !ig.failed[]
        FEC.assemble_vector!(ig.asm, FEC.residual, Uu, p)
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

# Write g(t), g'(t), g''(t) (from the current bc_cache state) into the
# BC slots of the integrator's full-DOF U / V / A.  Called once at
# initialization (in simulation.jl) so the t=0 output has the right
# values at constrained nodes; subsequently called inside predict! at
# each step.  No-op for QuasiStaticIntegrator (no V/A).
_propagate_dirichlet_bcs_to_state!(::QuasiStaticIntegrator, p) = nothing
function _propagate_dirichlet_bcs_to_state!(ig::_DynamicIntegrator, p)
    FEC.update_field_dirichlet_bcs!(ig.U, ig.V, ig.A, p.dirichlet_bcs)
    return nothing
end

# --------------------------------------------------------------------------- #
# Generic FEC.evolve! — works for ALL integrators
# --------------------------------------------------------------------------- #

function FEC.evolve!(ig, p)
    FEC.update_time!(p)
    FEC.update_bc_values!(p, ig.asm)
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

# QuasiStatic: always save/restore ig.U for adaptive stepping rollback.
function _save_state!(ig::QuasiStaticIntegrator, p)
    copyto!(ig.U_save, ig.U)
end
function _restore_state!(ig::QuasiStaticIntegrator, p)
    copyto!(ig.U, ig.U_save)
    copyto!(p.field.data, p.field_old.data)
    # Reset state_new from state_old so retried Newton starts from clean state
    p.state_new.data .= p.state_old.data
    FEC._update_for_assembly!(p, ig.asm.dof, ig.U)
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
    FEC._update_for_assembly!(p, ig.asm.dof, _displacement(ig))
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
