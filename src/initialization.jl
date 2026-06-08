# Initial condition application and initial solves.
#
# Functions that apply displacement/velocity ICs and compute
# initial acceleration (dynamic) or initial equilibrium (quasi-static).

import FiniteElementContainers as FEC
import Krylov
using LinearAlgebra
using StaticArrays

# ---------------------------------------------------------------------------
# Initial displacement ICs
# ---------------------------------------------------------------------------

# Apply initial displacement ICs to the displacement vector U and field.
# Each entry: {node set: <name>, component: x|y|z, function: <expr>}
function _apply_initial_displacement_ics!(integrator, mesh, asm_cpu, p, p_cpu,
                                           disp_ics, backend, t0::Float64)
    isempty(disp_ics) && return
    dof = asm_cpu.dof
    X   = p_cpu.coords.data

    inv_map = zeros(Int, length(dof))
    for (i, fd) in enumerate(dof.unknown_dofs)
        inv_map[fd] = i
    end

    # Build IC in the full DOF space, then extract unknown DOFs for the
    # integrator (which is n_free-sized with use_condensed=false).
    U_full = zeros(Float64, length(dof))
    for entry in disp_ics
        var_sym  = _component_to_string(entry["component"])
        func     = _make_function(entry["function"])
        nset_sym = entry["node_set"]
        bk       = FEC.BCBookKeeping(mesh, dof, var_sym; nset_name=nset_sym)
        for (full_dof, node) in zip(bk.dofs, bk.nodes)
            unk_idx = inv_map[full_dof]
            unk_idx == 0 && continue
            coords = SVector{3, Float64}(X[(node-1)*3+1], X[(node-1)*3+2], X[(node-1)*3+3])
            U_full[full_dof] = func(coords, t0)
        end
    end
    # Extract unknown DOFs into integrator's reduced vector
    U_unk = U_full[dof.unknown_dofs]
    copyto!(_displacement(integrator), U_unk)
    # Update field so the assembly sees the initial displacement.
    # Always update CPU params first; for GPU, sync field data afterward.
    FEC._update_for_assembly!(p_cpu, asm_cpu.dof, U_unk)
    if !(backend isa KA.CPU)
        copyto!(p.field.data, p_cpu.field.data)
    end
end

# ---------------------------------------------------------------------------
# Initial velocity ICs
# ---------------------------------------------------------------------------

# Apply initial velocity ICs to the velocity vector V.
# Each entry: {node set: <name>, component: x|y|z, function: <expr>}
# Shared by NewmarkIntegrator and CentralDifferenceIntegrator.
function _apply_initial_velocity_ics!(integrator::_DynamicIntegrator, mesh, asm_cpu, p_cpu, vel_ics, t0::Float64)
    isempty(vel_ics) && return
    dof = asm_cpu.dof                # always CPU dof manager for index arithmetic
    X   = p_cpu.coords.data       # flat, node-major: [x₁,y₁,z₁, x₂,y₂,z₂, ...]

    # Inverse map: full_dof_idx -> index in unknown_dofs (0 = constrained DOF)
    n_unk   = length(dof.unknown_dofs)
    inv_map = zeros(Int, length(dof))
    for (i, fd) in enumerate(dof.unknown_dofs)
        inv_map[fd] = i
    end

    # Build IC in the full DOF space, then extract unknown DOFs for the
    # integrator (which is n_free-sized with use_condensed=false).
    V_full = zeros(Float64, length(dof))
    for entry in vel_ics
        var_sym  = _component_to_string(entry["component"])
        func     = _make_function(entry["function"])
        nset_sym = entry["node_set"]
        bk       = FEC.BCBookKeeping(mesh, dof, var_sym; nset_name=nset_sym)
        for (full_dof, node) in zip(bk.dofs, bk.nodes)
            unk_idx = inv_map[full_dof]
            unk_idx == 0 && continue   # skip constrained DOFs
            coords = SVector{3, Float64}(X[(node-1)*3+1], X[(node-1)*3+2], X[(node-1)*3+3])
            V_full[full_dof] = func(coords, t0)
        end
    end
    # integrator.V is full-DOF in the Norma-shape integrator state; write
    # only the free slice from the IC values.  BC slots get g'(t_0)
    # populated separately at the start of the time loop via
    # FEC.update_field_dirichlet_bcs! in predict!.
    #
    # IC values live on the CPU (the parser walks CPU coords and evaluates
    # CPU expression functions); `integrator.V` may live on GPU.  `copyto!`
    # handles the CPU→GPU sync; a broadcasted `.=` would build a CPU view
    # and pass it through a GPU kernel, which fails as non-bitstype.
    V_unk = V_full[dof.unknown_dofs]
    copyto!(view(integrator.V, dof.unknown_dofs), V_unk)
end

# No-op for integrators that do not support initial velocity ICs.
function _apply_initial_velocity_ics!(integrator, mesh, asm_cpu, p_cpu, vel_ics, t0)
    isempty(vel_ics) || _carina_log(0, :warning, "Initial velocity ICs ignored for non-Newmark integrator.")
end

# ---------------------------------------------------------------------------
# Traveling-wave initial conditions
#
# For a propagating wave u(x, t) = f(s − c·t) along axis s ∈ {x, y, z},
# the initial conditions are
#
#     u₀(x) = f(s)            (user-supplied displacement profile)
#     v₀(x) = −c · ∂u₀/∂s     (derived symbolically)
#
# A user can write the displacement profile in the TOML and Carina
# derives the velocity field via [`FEC.Expressions.differentiate`](@ref) —
# no hand-coded derivatives, no ForwardDiff at IC time.
# ---------------------------------------------------------------------------
function _apply_initial_traveling_wave_ics!(integrator::_DynamicIntegrator, mesh, asm_cpu,
                                              p, p_cpu, tw_ics, backend, t0::Float64)
    isempty(tw_ics) && return
    dof = asm_cpu.dof
    X   = p_cpu.coords.data

    inv_map = zeros(Int, length(dof))
    for (i, fd) in enumerate(dof.unknown_dofs)
        inv_map[fd] = i
    end

    U_full = zeros(Float64, length(dof))
    V_full = zeros(Float64, length(dof))
    touched_displacement = false

    for entry in tw_ics
        var_sym  = _component_to_string(entry["component"])
        u_str    = _inline_expr_bindings(String(entry["displacement"]))
        u_expr   = FEC.Expressions.ScalarExpressionFunction{Float64}(
                       u_str, _CARINA_EXPR_VARS)
        dir_idx  = _direction_to_idx(String(entry["direction"]))
        c        = _f64(entry["wave_speed"])
        du_ds    = FEC.Expressions.differentiate(u_expr, dir_idx)
        nset_sym = entry["node_set"]
        bk       = FEC.BCBookKeeping(mesh, dof, var_sym; nset_name=nset_sym)

        touched_displacement = true
        for (full_dof, node) in zip(bk.dofs, bk.nodes)
            unk_idx = inv_map[full_dof]
            unk_idx == 0 && continue
            coords = SVector{3, Float64}(X[(node-1)*3+1], X[(node-1)*3+2], X[(node-1)*3+3])
            U_full[full_dof] = u_expr(coords, t0)
            V_full[full_dof] = -c * du_ds(coords, t0)
        end
    end

    if touched_displacement
        U_unk = U_full[dof.unknown_dofs]
        copyto!(_displacement(integrator), U_unk)
        FEC._update_for_assembly!(p_cpu, asm_cpu.dof, U_unk)
        if !(backend isa KA.CPU)
            copyto!(p.field.data, p_cpu.field.data)
        end
    end

    V_unk = V_full[dof.unknown_dofs]
    copyto!(view(integrator.V, dof.unknown_dofs), V_unk)
end

# No-op for integrators that do not carry a velocity state.
function _apply_initial_traveling_wave_ics!(integrator, mesh, asm_cpu, p, p_cpu,
                                              tw_ics, backend, t0::Float64)
    isempty(tw_ics) || _carina_log(0, :warning,
        "Traveling-wave initial conditions ignored for non-dynamic integrator.")
end

# ---------------------------------------------------------------------------
# Initial acceleration (dynamic integrators)
# ---------------------------------------------------------------------------

# Compute the consistent initial acceleration A₀ = M⁻¹·(F_ext − F_int(U₀)) for
# Newmark integrators.  Called once after ICs are applied, before the first step.
# Mirrors Norma's `initialize(Newmark, ...)` which solves the same system.
function _compute_initial_acceleration!(integrator::NewmarkIntegrator, asm_cpu, p_cpu)
    _carina_log(0, :acceleration, "Computing Initial Acceleration...")
    t_start = time()

    # integrator.U is full-DOF; assembly expects the free-DOF slice as Uu.
    free = asm_cpu.dof.unknown_dofs
    Uu_cpu = Vector{Float64}(view(integrator.U, free))

    # Assemble residual at initial displacement U₀: R = F_int(U₀) − F_ext
    FEC.assemble_vector!(asm_cpu, FEC.residual, Uu_cpu, p_cpu)
    FEC.assemble_vector_neumann_bc!(asm_cpu, Uu_cpu, p_cpu)
    FEC.assemble_vector_source!(asm_cpu, Uu_cpu, p_cpu)
    rhs = -copy(FEC.residual(asm_cpu))   # F_ext − F_int(U₀)
    _apply_point_loads!(rhs, FEC.current_time(p_cpu.times))

    norm_rhs = sqrt(sum(abs2, rhs))
    if norm_rhs < eps(Float64)
        elapsed = time() - t_start
        _carina_logf(0, :acceleration, "Initial Acceleration = 0 (trivial RHS, %s)", format_time(elapsed))
        return nothing
    end

    # Solve M·A₀ = rhs using CG (M is SPD).  M is the free-DOF reduced
    # consistent mass; A₀ is free-DOF.
    FEC.assemble_mass!(asm_cpu, FEC.mass, Uu_cpu, p_cpu)
    M = FEC.mass(asm_cpu)
    A0, stats = Krylov.cg(M, rhs; atol=0.0, rtol=1e-12, verbose=0)

    elapsed = time() - t_start
    _carina_logf(0, :acceleration,
        "Initial Acceleration: |A₀| = %.2e, CG iters = %d (%s)",
        sqrt(sum(abs2, A0)), stats.niter, format_time(elapsed))

    # Write into the free slice of the full-DOF integrator.A buffer.
    # BC slots will be populated by predict! at the first time step
    # via FEC.update_field_dirichlet_bcs!.  copyto! handles the CPU→GPU
    # transfer when integrator.A lives on a GPU backend.
    copyto!(view(integrator.A, free), A0)
    return nothing
end

function _compute_initial_acceleration!(integrator::CentralDifferenceIntegrator, asm_cpu, p_cpu)
    _carina_log(0, :acceleration, "Computing Initial Acceleration...")
    t_start = time()

    free = asm_cpu.dof.unknown_dofs
    Uu_cpu = Vector{Float64}(view(integrator.U, free))

    # A₀ = M_lumped⁻¹ · (F_ext − F_int(U₀))
    FEC.assemble_vector!(asm_cpu, FEC.residual, Uu_cpu, p_cpu)
    FEC.assemble_vector_neumann_bc!(asm_cpu, Uu_cpu, p_cpu)
    FEC.assemble_vector_source!(asm_cpu, Uu_cpu, p_cpu)
    rhs = -copy(FEC.residual(asm_cpu))   # F_ext − F_int(U₀)
    _apply_point_loads!(rhs, FEC.current_time(p_cpu.times))

    norm_rhs = sqrt(sum(abs2, rhs))
    if norm_rhs < eps(Float64)
        elapsed = time() - t_start
        _carina_logf(0, :acceleration, "Initial Acceleration = 0 (trivial RHS, %s)", format_time(elapsed))
        return nothing
    end

    # Lumped mass is already on CPU (computed during integrator construction).
    # Both m_lumped and A₀ are free-DOF.
    m_cpu = Vector{Float64}(integrator.m_lumped)
    A0 = rhs ./ m_cpu

    elapsed = time() - t_start
    _carina_logf(0, :acceleration,
        "Initial Acceleration: |A₀| = %.2e (%s)",
        sqrt(sum(abs2, A0)), format_time(elapsed))

    copyto!(view(integrator.A, free), A0)
    return nothing
end

# No-op for quasi-static integrators.
_compute_initial_acceleration!(integrator, asm_cpu, p_cpu) = nothing

# ---------------------------------------------------------------------------
# Initial equilibrium solve for quasi-static simulations.
# Solves R(U₀) = 0 at t₀ before time stepping begins.
# ---------------------------------------------------------------------------

function _compute_initial_equilibrium!(integrator::QuasiStaticIntegrator, p)
    integrator.initial_equilibrium || return nothing
    _carina_log(0, :equilibrium, "Establishing Initial Equilibrium...")
    t_start = time()

    # Solve R(U) = 0 at t₀ without advancing time.
    # BC values are already set at t₀ by the preceding FEC.update_bc_values!(p_cpu).
    # We call evaluate! + solve! directly (not FEC.evolve! which advances time).
    solve!(nonlinear_solver(integrator), integrator, p)

    elapsed = time() - t_start
    if integrator.failed[]
        error("Failed to establish initial equilibrium.")
    end
    _carina_log(0, :equilibrium, "Initial Equilibrium established ($(format_time(elapsed)))")
    # Promote converged state so time stepping starts from equilibrium.
    _finalize_step!(integrator, p)
    return nothing
end

# No-op for dynamic integrators.
_compute_initial_equilibrium!(integrator, p) = nothing
