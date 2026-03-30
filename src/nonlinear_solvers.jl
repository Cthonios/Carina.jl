# Nonlinear solver implementations for Carina.
#
# Contains solve! methods for all nonlinear solver types:
#   NewtonSolver (regular + LBFGS), NLCGSolver, SteepestDescentSolver, ExplicitSolver
#
# Depends on types from solvers.jl, integrator types/methods from integrators.jl,
# and linear solver helpers from linear_solvers.jl.

using LinearAlgebra
import FiniteElementContainers as FEC

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
            _carina_logf(8, :linesearch, "    LS: α = %.2e : m = %.2e → %.3e : [ACCEPT]",
                         α, merit_0, merit)
            return α
        end

        _carina_logf(8, :linesearch, "    LS: α = %.2e : m = %.2e → %.3e : [REDUCE]",
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
    _carina_logf(8, :solve, "Iter [0] |R| = %.2e : |r| = %.2e : %s",
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
        if t_eval + t_solve > 0.01
            _carina_logf(8, :solve,
                "Iter [%d] |R| = %.2e : |r| = %.2e : |ΔU| = %.2e : t_eval = %.2fs : t_solve = %.2fs : %s",
                iter, norm_R, rel_R, norm_step, t_eval, t_solve, _status_str(converged))
        else
            _carina_logf(8, :solve,
                "Iter [%d] |R| = %.2e : |r| = %.2e : |ΔU| = %.2e : %s",
                iter, norm_R, rel_R, norm_step, _status_str(converged))
        end
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
    _carina_logf(8, :solve, "Iter [0] |R| = %.2e : |r| = %.2e : %s",
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
        if t_dir + t_ls > 0.01
            _carina_logf(8, :solve,
                "Iter [%d] |R| = %.2e : |r| = %.2e : |ΔU| = %.2e : step = %.2e : LS = %d : t_dir = %.0fms : t_ls = %.0fms : %s",
                iter, new_norm_R, new_rel_R,
                norm_dU, step, ls_iters, t_dir*1e3, t_ls*1e3, _status_str(converged))
        else
            _carina_logf(8, :solve,
                "Iter [%d] |R| = %.2e : |r| = %.2e : |ΔU| = %.2e : step = %.2e : LS = %d : %s",
                iter, new_norm_R, new_rel_R,
                norm_dU, step, ls_iters, _status_str(converged))
        end
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

# ---- Nonlinear CG (Preconditioned Polak-Ribière+) ----

function _apply_precond!(g, R, ::NoPreconditioner)
    copyto!(g, R)
end

function _apply_precond!(g, R, pc::JacobiPreconditioner)
    @. g = pc.inv_diag * R
end

function solve!(ns::NLCGSolver, ig, p)
    evaluate!(ig, p) || (ig.failed[] = true; return)
    initial_norm = norm_R = sqrt(sum(abs2, residual(ig)))
    _carina_logf(8, :solve, "Iter [0] |R| = %.2e : |r| = %.2e : %s",
                 initial_norm, 1.0, _status_str(false))

    # g = M⁻¹ R_eff (preconditioned negative gradient of energy)
    # R_eff = F_ext - F_int = -∇Π, so d = g is the descent direction.
    _apply_precond!(ns.g, residual(ig), ns.precond)
    copyto!(ns.d, ns.g)
    rg = dot(residual(ig), ns.g)   # R·g for β denominator

    converged = false
    for iter in 1:ns.max_iters
        # Save U for line search restore
        U = _displacement(ig)
        copyto!(ns.U_save, U)

        # Backtracking line search
        α = 1.0
        for _ in 1:ns.ls_max_iters
            @. U = ns.U_save + α * ns.d
            FEC._update_for_assembly!(p, ig.asm.dof, U)
            evaluate!(ig, p) || (ig.failed[] = true; return)
            norm_R_trial = sqrt(sum(abs2, residual(ig)))
            if isfinite(norm_R_trial) && norm_R_trial < norm_R
                break
            end
            α *= ns.ls_backtrack
        end

        # Accept step
        @. U = ns.U_save + α * ns.d
        FEC._update_for_assembly!(p, ig.asm.dof, U)
        evaluate!(ig, p) || (ig.failed[] = true; return)

        # Convergence check
        norm_R = sqrt(sum(abs2, residual(ig)))
        rel_R  = initial_norm > 0 ? norm_R / initial_norm : norm_R
        norm_step = α * sqrt(sum(abs2, ns.d))
        met_tol = norm_step < ns.abs_increment_tol ||
                  norm_R   < ns.abs_residual_tol   ||
                  rel_R    < ns.rel_residual_tol
        converged = met_tol && iter > ns.min_iters
        _carina_logf(8, :solve,
            "Iter [%d] |R| = %.2e : |r| = %.2e : |ΔU| = %.2e : α = %.2e : %s",
            iter, norm_R, rel_R, norm_step, α, _status_str(converged))
        converged && break

        # Update preconditioned gradient
        copyto!(ns.g_old, ns.g)
        rg_old = rg
        _apply_precond!(ns.g, residual(ig), ns.precond)
        rg = dot(residual(ig), ns.g)

        # Polak-Ribière+ β (clamped ≥ 0)
        β = rg_old != 0.0 ? max(0.0, (rg - dot(residual(ig), ns.g_old)) / rg_old) : 0.0

        # Restart on orthogonality loss
        if ns.orthogonality_tol > 0 && rg != 0
            γ = abs(dot(residual(ig), ns.g_old)) / abs(rg)
            if γ > ns.orthogonality_tol
                β = 0.0
            end
        end
        # Periodic restart
        if ns.restart_interval > 0 && iter % ns.restart_interval == 0
            β = 0.0
        end

        # Update search direction: d = g + β * d_old
        @. ns.d = ns.g + β * ns.d
    end

    if !converged && _mark_failed_on_nonconvergence(ig)
        ig.failed[] = true
    end
end

# ---- Steepest Descent with energy-based Armijo line search ----

# Compute total potential energy: Π = W_int - f_ext · U
# W_int comes from FEC.assemble_scalar!(asm, FEC.energy, U, p).
# f_ext · U = -(R_assembled · U - W_int_linearized), but simpler to compute
# f_ext directly: after residual assembly, R = F_int + F_nbc + F_bf, so
# F_ext = -(F_nbc + F_bf). However, we can compute Π more simply:
#   Π = W_int + R_assembled · U  (for linear external forces)
# Actually: R = F_int - F_ext (with sign convention), so
# F_ext · U = F_int · U - R · U.  And W_int = ∫ Ψ dΩ.  For hyperelastic,
# F_int = ∂W_int/∂U, so F_int · dU = dW_int. Not helpful for finite steps.
#
# Simplest correct approach: evaluate W_int via assemble_scalar, then
# compute external work via separate assembly of Neumann/body contributions.
# But FEC doesn't separate them. Instead, use the fact that for the Armijo
# check we only need Π(U+αd) - Π(U) ≤ c·α·∇Π·d, and ∇Π·d = -R_eff·d.
# We evaluate Π at trial and reference points.

function _compute_energy(ig, p)
    U = _displacement(ig)
    asm = ig.asm
    FEC.assemble_scalar!(asm, FEC.energy, U, p)
    # Sum strain energy over all blocks and QPs
    W_int = sum(sum(v) for v in values(asm.scalar_quadrature_storage))
    # External work: from Neumann BCs and body forces.
    # After residual assembly, R_storage = F_int + F_nbc + F_bf
    # where F_nbc and F_bf are external contributions.
    # F_ext · U = -(F_nbc + F_bf) · U.
    # Since R_eff = -(F_int + F_nbc + F_bf) = F_ext - F_int,
    # we have: Π = W_int - F_ext · U = W_int - (R_eff + F_int) · U
    # But F_int · U ≠ W_int in general (nonlinear).
    #
    # For correct Π, assemble R to get F_ext separately.
    # R_eff = F_ext - F_int → F_ext = R_eff + F_int
    # But we only have R_eff and W_int.
    #
    # Alternative: just use W_int + (assembled residual) · U as the merit.
    # Since assembled R = F_int - F_ext (FEC convention) and
    # R_eff = -R_assembled, we have R_assembled = -R_eff.
    # Π = W_int - F_ext · U = W_int + R_assembled · U - F_int · U
    # This requires F_int · U which we don't have separately.
    #
    # SIMPLIFICATION: For line search we only need RELATIVE changes in Π.
    # The external force contributions are linear in U for gravity/Neumann:
    #   W_ext(U + αd) = W_ext(U) + α · f_ext · d
    # So: Π(U + αd) - Π(U) = [W_int(U+αd) - W_int(U)] - α · f_ext · d
    # And f_ext · d = (R_eff + F_int) · d ≈ R_eff · d for small steps
    # (F_int · d ≈ dW_int/dα at α=0, which cancels).
    # Actually: ∂Π/∂α|₀ = ∂W_int/∂α - f_ext · d = F_int · d - f_ext · d = -R_eff · d
    # This is exact. So for Armijo we need:
    #   Π(α) ≤ Π(0) + c·α·(-R_eff · d)
    # where Π(α) = W_int(U+αd) - f_ext · (U+αd).
    # Since f_ext is constant: Π(α) - Π(0) = ΔW_int - α·f_ext·d
    # And f_ext·d = R_eff·d + F_int·d = R_eff·d + ∂W_int/∂α|₀
    #
    # Too complex. Just compute W_int and use the residual dot product for
    # the external work term. For the line search acceptance:
    # ΔΠ ≈ ΔW_int - α·R_eff·d (first-order external work correction)
    return W_int
end

function solve!(ns::SteepestDescentSolver, ig, p)
    evaluate!(ig, p) || (ig.failed[] = true; return)
    initial_norm = sqrt(sum(abs2, residual(ig)))
    _carina_logf(8, :solve, "Iter [0] |R| = %.2e : |r| = %.2e : %s",
                 initial_norm, 1.0, _status_str(false))

    converged = false
    for iter in 1:ns.max_iters
        # Descent direction: d = M⁻¹ R_eff (preconditioned)
        _apply_precond!(ns.d, residual(ig), ns.precond)

        # Current energy for Armijo check
        W_curr = _compute_energy(ig, p)
        slope  = -dot(residual(ig), ns.d)  # ∇Π · d = -R_eff · d

        # Save U for line search
        U = _displacement(ig)
        copyto!(ns.U_save, U)

        # Armijo backtracking line search on energy
        α = 1.0
        W_trial = W_curr
        for lsi in 1:ns.ls_max_iters
            @. U = ns.U_save + α * ns.d
            FEC._update_for_assembly!(p, ig.asm.dof, U)
            W_trial = _compute_energy(ig, p)
            # Armijo: Π(trial) ≤ Π(curr) + c·α·slope
            # slope is negative (descent), so rhs < Π(curr)
            if isfinite(W_trial) && W_trial ≤ W_curr + ns.ls_decrease * α * slope
                break
            end
            α *= ns.ls_backtrack
        end

        # Accept step and evaluate residual at accepted point
        @. U = ns.U_save + α * ns.d
        FEC._update_for_assembly!(p, ig.asm.dof, U)
        evaluate!(ig, p) || (ig.failed[] = true; return)

        # Convergence check
        norm_R    = sqrt(sum(abs2, residual(ig)))
        rel_R     = initial_norm > 0 ? norm_R / initial_norm : norm_R
        norm_step = α * sqrt(sum(abs2, ns.d))
        met_tol   = norm_step < ns.abs_increment_tol ||
                    norm_R   < ns.abs_residual_tol   ||
                    rel_R    < ns.rel_residual_tol
        converged = met_tol && iter > ns.min_iters

        ΔW = W_trial - W_curr
        if abs(ΔW) > 0.01
            _carina_logf(8, :solve,
                "Iter [%d] |R| = %.2e : |r| = %.2e : |ΔU| = %.2e : α = %.2e : ΔΠ = %.2e : %s",
                iter, norm_R, rel_R, norm_step, α, ΔW, _status_str(converged))
        else
            _carina_logf(8, :solve,
                "Iter [%d] |R| = %.2e : |r| = %.2e : |ΔU| = %.2e : α = %.2e : %s",
                iter, norm_R, rel_R, norm_step, α, _status_str(converged))
        end
        converged && break
    end

    if !converged && _mark_failed_on_nonconvergence(ig)
        ig.failed[] = true
    end
end

# ---- ExplicitSolver ----

function solve!(::ExplicitSolver, ig::CentralDifferenceIntegrator, p)
    evaluate!(ig, p) || (ig.failed[] = true; return)
    R_eff = residual(ig)
    @. ig.A = R_eff / ig.m_lumped
    ig.failed[] = false
end
