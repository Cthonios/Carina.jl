# Initial condition application and initial solves.
#
# Functions that apply displacement/velocity ICs and compute
# initial acceleration (dynamic) or initial equilibrium (quasi-static).

import FiniteElementContainers as FEC
import Krylov
using LinearAlgebra

# ---------------------------------------------------------------------------
# Initial displacement ICs
# ---------------------------------------------------------------------------

# Apply initial displacement ICs to the displacement vector U and field.
# Each entry: {node set: <name>, component: x|y|z, function: <expr>}
function _apply_initial_displacement_ics!(integrator::_DynamicIntegrator, mesh, asm_cpu, p, p_cpu,
                                           disp_ics, device, t0::Float64)
    isempty(disp_ics) && return
    dof = asm_cpu.dof
    X   = p_cpu.coords.data

    n_unk   = length(dof.unknown_dofs)
    inv_map = zeros(Int, length(dof))
    for (i, fd) in enumerate(dof.unknown_dofs)
        inv_map[fd] = i
    end

    # Build on CPU (full DOF vector), then bulk-copy to integrator.U
    # (which may be a non-scalar-indexable array like an FEC storage type).
    U_cpu = zeros(Float64, length(dof))
    for entry in disp_ics
        var_sym  = _component_to_symbol(entry["component"])
        func     = _make_function(entry["function"])
        nset_sym = Symbol(entry["node set"])
        bk       = FEC.BCBookKeeping(mesh, dof, var_sym; nset_name=nset_sym)
        for (full_dof, node) in zip(bk.dofs, bk.nodes)
            unk_idx = inv_map[full_dof]
            unk_idx == 0 && continue
            coords = @view X[(node-1)*3+1 : (node-1)*3+3]
            U_cpu[full_dof] = Base.invokelatest(func, coords, t0)
        end
    end
    copyto!(integrator.U, U_cpu)
    # Update field so the assembly sees the initial displacement.
    # Always update CPU params first; for GPU, sync field data afterward.
    FEC._update_for_assembly!(p_cpu, asm_cpu.dof, U_cpu)
    if device != :cpu
        copyto!(p.field.data, p_cpu.field.data)
    end
end

function _apply_initial_displacement_ics!(integrator, mesh, asm_cpu, p, p_cpu, disp_ics, device, t0)
    isempty(disp_ics) || @warn "Displacement ICs ignored for non-dynamic integrator."
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

    # Build on CPU (full DOF vector), then bulk-copy to integrator.V
    # (which may be a non-scalar-indexable array like an FEC storage type).
    V_cpu = zeros(Float64, length(dof))
    for entry in vel_ics
        var_sym  = _component_to_symbol(entry["component"])
        func     = _make_function(entry["function"])
        nset_sym = Symbol(entry["node set"])
        bk       = FEC.BCBookKeeping(mesh, dof, var_sym; nset_name=nset_sym)
        for (full_dof, node) in zip(bk.dofs, bk.nodes)
            unk_idx = inv_map[full_dof]
            unk_idx == 0 && continue   # skip constrained DOFs
            coords = @view X[(node-1)*3+1 : (node-1)*3+3]
            V_cpu[full_dof] = Base.invokelatest(func, coords, t0)
        end
    end
    copyto!(integrator.V, V_cpu)
end

# No-op for integrators that do not support initial velocity ICs.
function _apply_initial_velocity_ics!(integrator, mesh, asm_cpu, p_cpu, vel_ics, t0)
    isempty(vel_ics) || @warn "Initial velocity ICs ignored for non-Newmark integrator."
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

    U_cpu = Vector{Float64}(integrator.U)
    n     = length(U_cpu)

    # Assemble residual at initial displacement U₀: R = F_int(U₀) − F_ext
    FEC.assemble_vector!(asm_cpu, FEC.residual, U_cpu, p_cpu)
    FEC.assemble_vector_neumann_bc!(asm_cpu, U_cpu, p_cpu)
    FEC.assemble_vector_body_force!(asm_cpu, U_cpu, p_cpu)
    rhs = -copy(FEC.residual(asm_cpu))   # F_ext − F_int(U₀)

    norm_rhs = sqrt(sum(abs2, rhs))
    if norm_rhs < eps(Float64)
        elapsed = time() - t_start
        _carina_logf(0, :acceleration, "Initial Acceleration = 0 (trivial RHS, %s)", format_time(elapsed))
        return nothing
    end

    # Solve M·A₀ = rhs using CG (M is SPD)
    FEC.assemble_mass!(asm_cpu, FEC.mass, U_cpu, p_cpu)
    M = FEC.mass(asm_cpu)
    A0, stats = Krylov.cg(M, rhs; atol=0.0, rtol=1e-12, verbose=0)

    elapsed = time() - t_start
    _carina_logf(0, :acceleration,
        "Initial Acceleration: |A₀| = %.2e, CG iters = %d (%s)",
        sqrt(sum(abs2, A0)), stats.niter, format_time(elapsed))

    copyto!(integrator.A, A0)
    return nothing
end

function _compute_initial_acceleration!(integrator::CentralDifferenceIntegrator, asm_cpu, p_cpu)
    _carina_log(0, :acceleration, "Computing Initial Acceleration...")
    t_start = time()

    U_cpu = Vector{Float64}(integrator.U)
    n     = length(U_cpu)

    # A₀ = M_lumped⁻¹ · (F_ext − F_int(U₀))
    FEC.assemble_vector!(asm_cpu, FEC.residual, U_cpu, p_cpu)
    FEC.assemble_vector_neumann_bc!(asm_cpu, U_cpu, p_cpu)
    FEC.assemble_vector_body_force!(asm_cpu, U_cpu, p_cpu)
    rhs = -copy(FEC.residual(asm_cpu))   # F_ext − F_int(U₀)

    norm_rhs = sqrt(sum(abs2, rhs))
    if norm_rhs < eps(Float64)
        elapsed = time() - t_start
        _carina_logf(0, :acceleration, "Initial Acceleration = 0 (trivial RHS, %s)", format_time(elapsed))
        return nothing
    end

    # Lumped mass is already on CPU (computed during integrator construction)
    m_cpu = Vector{Float64}(integrator.m_lumped)
    A0 = rhs ./ m_cpu

    elapsed = time() - t_start
    _carina_logf(0, :acceleration,
        "Initial Acceleration: |A₀| = %.2e (%s)",
        sqrt(sum(abs2, A0)), format_time(elapsed))

    copyto!(integrator.A, A0)
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
    Base.invokelatest(solve!, nonlinear_solver(integrator), integrator, p)

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
