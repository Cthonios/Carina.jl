# Complete Simo-Hughes J2 finite-deformation plasticity model.
#
# Stress update: BOX 9.1 (Computational Inelasticity, p 319)
# Consistent tangent: BOX 9.2 (pp 320-321)
#
# Formulation:
#   Kinematics: multiplicative split F = Fᵉ Fᵖ, isochoric plastic flow
#   Elasticity: neo-Hookean-type with vol/dev split: U(J) + μ/2(tr b̄ᵉ - 3)
#   Yield: von Mises f = ‖s‖ - √(2/3)(σ_y + K α)
#   Flow: associated, isochoric
#   Hardening: linear isotropic
#
# State variables (NS = 7):
#   Z[1:6] = [b̄ᵉ₁₁, b̄ᵉ₂₂, b̄ᵉ₃₃, b̄ᵉ₂₃, b̄ᵉ₁₃, b̄ᵉ₁₂] (Voigt, symmetric)
#   Z[7]   = α (equivalent plastic strain)
#
# Properties (NP = 4): [κ, μ, σ_y, K]
#   κ   : bulk modulus
#   μ   : shear modulus
#   σ_y : initial yield stress
#   K   : linear isotropic hardening modulus

import ConstitutiveModels as CM
using Tensors, StaticArrays

# ---------------------------------------------------------------------------
# Helper: SymmetricTensor ↔ state vector (Voigt order: 11,22,33,23,13,12)
# ---------------------------------------------------------------------------

@inline function _sym_to_voigt6(A::SymmetricTensor{2,3,T}) where T
    SVector{6,T}(A[1,1], A[2,2], A[3,3], A[2,3], A[1,3], A[1,2])
end

@inline function _voigt6_to_sym(v::AbstractVector{T}) where T
    SymmetricTensor{2,3,T}((v[1], v[6], v[5], v[6], v[2], v[4], v[5], v[4], v[3]))
end

# ---------------------------------------------------------------------------
# Stress update (BOX 9.1)
# ---------------------------------------------------------------------------
#
# Inputs:
#   props  = [κ, μ, σ_y, K]
#   F      = total deformation gradient at t_{n+1}
#   F_old  = total deformation gradient at t_n (needed for relative f)
#   state_old = [b̄ᵉ_voigt(6), α]
#
# For quasi-static: F_old = I (reference) and F = I + ∇u (total).
# The relative deformation gradient is f = F · F_old⁻¹.
# For the first implementation, we store b̄ᵉ and recompute from F directly.
#
# Actually, BOX 9.1 uses the RELATIVE deformation gradient f_{n+1} = 1 + ∇ₓu_n.
# But in a total-Lagrangian FEM code (like FEC), we have the total F and ∇u.
# We can compute f = F_{n+1} · F_n⁻¹, but we don't have F_n stored.
#
# Alternative: store b̄ᵉ in the state (as above), and update it using:
#   b̄ᵉ_trial = F̄ · b̄ᵉ_n · F̄ᵀ   where F̄ = J^{-1/3} F · Fp_old⁻¹ ...
#
# Simplification: Since Fp is isochoric (det Fp = 1), we have J = det F.
# And b̄ᵉ = J^{-2/3} Fe · Feᵀ = J^{-2/3} F · Cp⁻¹ · Fᵀ.
# At t_n: b̄ᵉ_n is stored in state.
# At t_{n+1}: the trial elastic predictor uses the relative deformation:
#   f = F_{n+1} · F_n⁻¹
#   f̃ = [det f]^{-1/3} · f
#   b̄ᵉ_trial = f̃ · b̄ᵉ_n · f̃ᵀ
#
# Since we don't store F_n, we use the fact that in FEC, the physics kernel
# receives ∇u (displacement gradient w.r.t. reference) and state_old.
# F = I + ∇u is the total deformation gradient from reference.
# But b̄ᵉ_n was computed from F_n (the previous converged F).
# Without F_n, we can't compute f = F · F_n⁻¹.
#
# SOLUTION: Store Fp in the state (like CM does), then:
#   Fe = F · Fp⁻¹
#   b̄ᵉ = J^{-2/3} Fe · Feᵀ
# This avoids needing F_n entirely.
# State: [Fp_voigt(9), α] = 10 variables (same as CM).

# ---------------------------------------------------------------------------
# Stress update using Fp storage (compatible with FEC's total-Lagrangian API)
# ---------------------------------------------------------------------------

@inline function _sh_j2_stress(
    props,
    F::Tensor{2,3,T,9},
    state_old::AbstractVector,
) where T
    κ = T(props[1]); μ = T(props[2]); σ_y = T(props[3]); K = T(props[4])

    Fp_old = Tensor{2,3,T,9}(ntuple(i -> T(state_old[i]), Val(9)))
    α_n    = T(state_old[10])

    # Total Jacobian and isochoric deformation
    J    = det(F)
    Jm23 = J^(-T(2)/3)

    # Trial elastic left Cauchy-Green (isochoric): b̄ᵉ_trial = J^{-2/3} Fe_tr · Fe_trᵀ
    Fe_tr     = F ⋅ inv(Fp_old)
    be_bar_tr = symmetric(Jm23 * (Fe_tr ⋅ Fe_tr'))

    # Trial deviatoric Kirchhoff stress: s_trial = μ dev[b̄ᵉ_trial]
    s_trial      = μ * dev(be_bar_tr)
    s_trial_norm = norm(s_trial)

    # Effective shear modulus: μ̄ = μ/3 tr[b̄ᵉ_trial]
    μ̄ = μ * tr(be_bar_tr) / 3

    # Yield function
    f_trial = s_trial_norm - sqrt(T(2)/3) * (σ_y + K * α_n)

    I2 = one(SymmetricTensor{2,3,T})

    if f_trial ≤ zero(T)
        # Elastic step
        s_new      = s_trial
        be_bar_new = be_bar_tr
        α_new      = α_n
        Δγ         = zero(T)
    else
        # Plastic step — radial return (BOX 9.1, step 4)
        n = s_trial / s_trial_norm   # unit normal

        # Consistency parameter
        Δγ = f_trial / (2μ̄ + T(2)/3 * K)   # = f_trial / (2μ̄(1 + K/(3μ̄)))

        # Return map
        s_new = s_trial - 2μ̄ * Δγ * n
        α_new = α_n + sqrt(T(2)/3) * Δγ

        # Update b̄ᵉ (eq 9.3.33): b̄ᵉ = s/μ + (1/3) tr[b̄ᵉ_trial] I
        Ie_bar = tr(be_bar_tr) / 3
        be_bar_new = s_new / μ + Ie_bar * I2
    end

    # Kirchhoff stress: τ = J p 1 + s
    # Pressure: p = U'(J) where U(J) = κ/2 (J-1)² → p = κ(J-1)
    # (alternative: U = κ/4(J²-2logJ-1) → p = κ/2(J-1/J))
    # Use the simpler quadratic form for now:
    p = κ * (J - one(T))
    τ = J * p * I2 + s_new

    # PK1 stress: P = τ · F⁻ᵀ
    P = Tensor{2,3,T,9}(τ) ⋅ inv(F)'

    # Energy (not critical for Newton convergence)
    W = κ / 2 * (J - 1)^2 + μ / 2 * (tr(be_bar_new) - T(3))

    # Update Fp: from b̄ᵉ_new, recover Fe_new and Fp_new
    # b̄ᵉ = J^{-2/3} Fe · Feᵀ → Fe = J^{1/3} (b̄ᵉ)^{1/2} · R
    # For the state update, we use the relation:
    #   Fp_new = (Fe_new)⁻¹ · F
    # where Fe_new can be recovered from be_bar_new.
    # Since b̄ᵉ_new = J^{-2/3} Fe_new · Fe_newᵀ,
    #   Fe_new · Fe_newᵀ = J^{2/3} b̄ᵉ_new
    #   Fe_new = J^{1/3} V · R  where V = sqrt(b̄ᵉ_new), R = rotation
    # The simplest approach: Fe_new = J^{1/3} sqrt(b̄ᵉ_new) · R_trial
    # where R_trial comes from the polar decomposition of Fe_tr.
    #
    # Actually, for the return map the rotation doesn't change:
    # the radial return only modifies the stretch (eigenvalues of b̄ᵉ),
    # not the eigenvectors. So Fe_new shares the rotation of Fe_tr.
    #
    # Simplest: Fp_new = Fe_new⁻¹ · F where Fe_new preserves Fe_tr's rotation.

    if f_trial ≤ zero(T)
        Fp_new = Fp_old
    else
        # Polar decomposition of Fe_tr to get R
        # Fe_tr = V_tr · R_tr where V_tr = sqrt(Fe_tr · Fe_trᵀ)
        # For the isochoric part: F̃e_tr = J^{-1/3} Fe_tr
        # b̄ᵉ_trial = F̃e_tr · F̃e_trᵀ
        # F̃e_new = sqrt(b̄ᵉ_new) · R_tr   (same rotation)
        # Fe_new = J^{1/3} F̃e_new
        # Fp_new = Fe_new⁻¹ · F

        # Get rotation from trial: R = (b̄ᵉ_trial)^{-1/2} · F̃e_tr
        be_tr_sqrt_inv = Tensor{2,3,T,9}(CM._matrix_function(x -> 1/sqrt(x), be_bar_tr))
        Fe_tr_iso = Jm23^(T(3)/2) * Fe_tr   # = J^{-1/3} Fe_tr... wait, Jm23 = J^{-2/3}
        # F̃e_tr = [det Fe_tr]^{-1/3} · Fe_tr = J^{-1/3} · Fe_tr (since det Fp = 1)
        Fe_tr_iso2 = J^(-T(1)/3) * Fe_tr
        R_tr = be_tr_sqrt_inv ⋅ Fe_tr_iso2

        be_new_sqrt = Tensor{2,3,T,9}(CM._matrix_function(sqrt, be_bar_new))
        Fe_new_iso = be_new_sqrt ⋅ R_tr
        Fe_new = J^(T(1)/3) * Fe_new_iso
        Fp_new = inv(Fe_new) ⋅ F
    end

    fp = Fp_new.data
    state_new = SVector{10,T}(
        fp[1], fp[2], fp[3], fp[4], fp[5], fp[6], fp[7], fp[8], fp[9], α_new
    )
    return W, P, state_new, s_new, be_bar_tr, s_trial_norm, μ̄, Δγ, α_n
end

# ---------------------------------------------------------------------------
# Consistent tangent (BOX 9.2)
# ---------------------------------------------------------------------------

@inline function _sh_j2_tangent(
    props,
    F::Tensor{2,3,T,9},
    state_old::AbstractVector,
    P::Tensor{2,3,T,9},
    s_new::SymmetricTensor{2,3,T},
    be_bar_tr::SymmetricTensor{2,3,T},
    s_trial_norm::T,
    μ̄::T,
    Δγ::T,
    α_n::T,
) where T
    κ = T(props[1]); μ = T(props[2]); σ_y = T(props[3]); K = T(props[4])

    J = det(F)
    F_inv = inv(F)
    S = F_inv ⋅ P   # 2nd Piola-Kirchhoff

    I2 = one(SymmetricTensor{2,3,T})
    I4_sym = one(SymmetricTensor{4,3,T})

    # BOX 9.2, Step 1: Volumetric spatial tangent
    #
    # C = (JU')'·J · (1⊗1) − 2·J·U' · I_sym + C̄
    #
    # For U(J) = κ/2 (J−1)²:
    #   U'  = κ(J−1),  JU' = κJ(J−1)
    #   (JU')' = d/dJ[κJ(J−1)] = κ(2J−1)
    #   (JU')'·J = κ·J·(2J−1)
    #   2·J·U'   = 2·κ·J·(J−1)
    coeff_1x1 = κ * J * (2*J - one(T))       # (JU')' · J
    coeff_I   = 2 * κ * J * (J - one(T))      # 2 · J · U'
    c_vol = coeff_1x1 * (I2 ⊗ I2) - coeff_I * I4_sym

    # Unit normal n = s_trial / ‖s_trial‖ (recomputed from be_bar_tr)
    s_trial_recomp = μ * dev(be_bar_tr)
    n = s_trial_norm > zero(T) ? s_trial_recomp / s_trial_norm :
        zero(SymmetricTensor{2,3,T})

    # Deviatoric trial tangent: C̄_trial (BOX 9.2, step 1)
    c_dev_trial = 2μ̄ * (I4_sym - T(1)/3 * (I2 ⊗ I2)) -
                  T(2)/3 * s_trial_norm * (n ⊗ I2 + I2 ⊗ n)

    f_trial = s_trial_norm - sqrt(T(2)/3) * (σ_y + K * α_n)

    if f_trial ≤ zero(T)
        # Elastic
        CC_spatial = c_vol + c_dev_trial
    else
        # BOX 9.2, Step 2: Scaling factors
        β₀ = one(T) + K / (3μ̄)
        β₁ = 2μ̄ * Δγ / s_trial_norm
        β₂ = (one(T) - one(T)/β₀) * T(2)/3 * s_trial_norm / μ * Δγ
        β₃ = one(T)/β₀ - β₁ + β₂
        β₄ = (one(T)/β₀ - β₁) * s_trial_norm / μ̄

        n_sq = symmetric(n ⋅ n)
        c_dev_n2 = symmetric(n ⊗ dev(n_sq) + dev(n_sq) ⊗ n) / 2

        # BOX 9.2, Step 3
        CC_spatial = c_vol + c_dev_trial -
                     β₁ * c_dev_trial -
                     2μ̄ * β₃ * (n ⊗ n) -
                     2μ̄ * β₄ * c_dev_n2
    end

    # Pull-back: spatial → material (∂S/∂C form)
    CC = MArray{Tuple{3,3,3,3},T,4,81}(ntuple(_ -> zero(T), Val(81)))
    for A in 1:3, B in 1:3, C in 1:3, D in 1:3
        val = zero(T)
        for a in 1:3, b in 1:3, c in 1:3, d in 1:3
            val += F_inv[A, a] * F_inv[B, b] * CC_spatial[a, b, c, d] * F_inv[C, c] * F_inv[D, d]
        end
        CC[A, B, C, D] = val
    end

    # convect_tangent: CC + S → ∂P/∂F
    return _convect_tangent(CC, S, F)
end

# ---------------------------------------------------------------------------
# Override CM API for FiniteDefJ2Plasticity
# ---------------------------------------------------------------------------

function CM.pk1_stress(
    ::CM.FiniteDefJ2Plasticity,
    props, Δt,
    ∇u, θ, Z_old, Z_new
)
    F = ∇u + one(∇u)
    W, P, state_new_vec, _, _, _, _, _, _ = _sh_j2_stress(props, F, Z_old)
    Z_new .= state_new_vec
    return P
end

function CM.material_tangent(
    ::CM.FiniteDefJ2Plasticity,
    props, Δt,
    ∇u, θ, Z_old, Z_new
)
    F = ∇u + one(∇u)
    W, P, state_new_vec, s_new, be_bar_tr, s_trial_norm, μ̄, Δγ, α_n =
        _sh_j2_stress(props, F, Z_old)
    Z_new .= state_new_vec
    return _sh_j2_tangent(props, F, Z_old, P,
                           s_new, be_bar_tr, s_trial_norm, μ̄, Δγ, α_n)
end

function CM.helmholtz_free_energy(
    ::CM.FiniteDefJ2Plasticity,
    props, Δt,
    ∇u, θ, Z_old, Z_new
)
    F = ∇u + one(∇u)
    W, _, state_new_vec, _, _, _, _, _, _ = _sh_j2_stress(props, F, Z_old)
    Z_new .= state_new_vec
    return W
end
