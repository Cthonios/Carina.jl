# Simo-Hughes consistent elastoplastic tangent for J2 finite-deformation
# plasticity (BOX 9.2 from Computational Inelasticity, pp 320-321).
#
# This replaces the frozen-Fp approximation in ConstitutiveModels.jl with
# the exact consistent tangent, giving quadratic Newton convergence.
#
# The algorithm:
# 1. From the converged stress update, extract spatial quantities:
#    b̄ᵉ (isochoric elastic left Cauchy-Green), s (deviatoric Kirchhoff),
#    n (unit normal), μ̄ (effective shear modulus), Δγ
# 2. Compute the β scaling factors (BOX 9.2, step 2)
# 3. Assemble the spatial consistent tangent c_ep (BOX 9.2, step 3)
# 4. Compute S (2nd PK) and CC (material tangent ∂S/∂C) via pull-back
# 5. Compute ∂P/∂F via convect_tangent (Norma pattern)

import ConstitutiveModels as CM
using Tensors, StaticArrays

# ---------------------------------------------------------------------------
# convect_tangent: CC (∂S/∂C) + S → AA (∂P/∂F)
#
# AA[i,J,k,L] = δ_ik · S[L,J] + Σ_AB F[i,A] · CC[A,J,L,B] · F[k,B]
#
# From Norma's constitutive.jl — standard push-forward of material tangent.
# ---------------------------------------------------------------------------

@inline function _convect_tangent(
    CC::MArray{Tuple{3,3,3,3},T,4,81},
    S::Tensor{2,3,T,9},
    F::Tensor{2,3,T,9},
) where T
    AA = MArray{Tuple{3,3,3,3},T,4,81}(undef)
    for i in 1:3, J in 1:3, k in 1:3, L in 1:3
        val = ifelse(i == k, one(T), zero(T)) * S[L, J]
        for A in 1:3, B in 1:3
            val += F[i, A] * CC[A, J, L, B] * F[k, B]
        end
        AA[i, J, k, L] = val
    end
    return Tensor{4,3,T,81}(ntuple(i -> AA[i], Val(81)))
end

# ---------------------------------------------------------------------------
# Simo-Hughes consistent tangent for FiniteDefJ2Plasticity
# ---------------------------------------------------------------------------

@inline function _j2_tangent_simo_hughes(
    props,
    F::Tensor{2,3,T,9},
    state_old::AbstractVector,
    P::Tensor{2,3,T,9},
    state_new::AbstractVector,
) where T
    λ = T(props[1]); μ = T(props[2]); σ_y = T(props[3]); H = T(props[4])
    κ = λ + T(2)/3 * μ   # bulk modulus

    Fp_old = Tensor{2,3,T,9}(ntuple(i -> T(state_old[i]), Val(9)))
    eqps   = T(state_old[10])
    Fp_new = Tensor{2,3,T,9}(ntuple(i -> T(state_new[i]), Val(9)))
    Δγ     = T(state_new[10]) - eqps  # = Δεᵖ

    # Elastic deformation gradient and Jacobian
    Fe     = F ⋅ inv(Fp_new)
    J      = det(Fe)   # J = det(F) since det(Fp) = 1 (isochoric plasticity)
    F_inv  = inv(F)

    # 2nd Piola-Kirchhoff stress: S = F⁻¹ · P
    S = F_inv ⋅ P

    # Isochoric elastic left Cauchy-Green: b̄ᵉ = J^{-2/3} Fᵉ Fᵉᵀ
    Jm23   = J^(-T(2)/3)
    be_bar = symmetric(Jm23 * (Fe ⋅ Fe'))

    # Deviatoric Kirchhoff stress: s = μ dev[b̄ᵉ]
    s_dev  = μ * dev(be_bar)
    s_norm = norm(s_dev)

    # Unit normal (flow direction in stress space)
    n = s_norm > zero(T) ? s_dev / s_norm : zero(SymmetricTensor{2,3,T})

    # Effective shear modulus: μ̄ = μ (1/3) tr[b̄ᵉ]
    μ̄ = μ * tr(be_bar) / 3

    # Pressure: p = κ (J - 1/J) / 2  [for U(J) = κ/4 (J² - 1 - 2 log J)]
    # More precisely: p = dU/dJ where U = κ/4 (J² - 2 log J - 1)
    #                 p = κ/2 (J - 1/J)
    p = κ / 2 * (J - 1 / J)

    # ---------------------------------------------------------------------------
    # BOX 9.2, Step 1: Spatial elasticity tensor C (hyperelastic part)
    # ---------------------------------------------------------------------------
    # C = (J U'')' J 1⊗1 - 2 J U' I + C̄
    # where:
    #   U'  = p/J = κ/2 (1 - 1/J²)
    #   U'' = κ/2 (1 + 1/J²) → (J U'')' J = κ J² (1/2 + 1/(2J²)) ≈ κ(J² + 1)/2
    #   Actually: d(JU')/dJ = d(p)/dJ = κ/2 (1 + 1/J²)
    #   So: (d(p)/dJ) * J = κ/2 * J * (1 + 1/J²) = κ/2 * (J + 1/J)
    #
    # For the deviatoric part, C̄ is the trial spatial tangent:
    #   C̄ = 2μ̄ [I_sym - (1/3) 1⊗1 - (2/3) ‖s‖ (n⊗1 + 1⊗n)]

    I2 = one(SymmetricTensor{2,3,T})
    I4_sym = one(SymmetricTensor{4,3,T})   # symmetric 4th-order identity

    dpJ = κ / 2 * (J + 1 / J)   # d(p)/dJ * J

    # Volumetric part: (dpJ) 1⊗1 - 2p I_sym
    c_vol = dpJ * (I2 ⊗ I2) - 2 * p * I4_sym

    # Deviatoric trial part: C̄_trial
    c_dev_trial = 2μ̄ * (I4_sym - T(1)/3 * (I2 ⊗ I2)) -
                  T(2)/3 * s_norm * (n ⊗ I2 + I2 ⊗ n)

    f_tr = s_norm - sqrt(T(2)/3) * (σ_y + H * eqps)

    if f_tr ≤ zero(T)
        # Elastic step: c_ep = C = c_vol + c_dev_trial
        CC_spatial = c_vol + c_dev_trial
    else
        # ---------------------------------------------------------------------------
        # BOX 9.2, Step 2: Scaling factors
        # ---------------------------------------------------------------------------
        k_prime = H   # k'(α) = K for linear hardening

        β₀ = 1 + k_prime / (3μ̄)
        β₁ = 2μ̄ * Δγ / s_norm
        β₂ = (1 - 1/β₀) * T(2)/3 * (s_norm / μ) * Δγ
        β₃ = 1/β₀ - β₁ + β₂
        β₄ = (1/β₀ - β₁) * s_norm / μ̄

        # n² = n·n (symmetric tensor product)
        n_sq = symmetric(n ⋅ n)

        # ---------------------------------------------------------------------------
        # BOX 9.2, Step 3: Consistent tangent
        # ---------------------------------------------------------------------------
        # c_ep = C_trial - β₁ C̄_trial - 2μ̄ β₃ (n ⊗ n) - 2μ̄ β₄ sym[n ⊗ dev(n²)]ˢ
        c_dev_n2 = symmetric(n ⊗ dev(n_sq) + dev(n_sq) ⊗ n) / 2

        CC_spatial = c_vol + c_dev_trial -
                     β₁ * c_dev_trial -
                     2μ̄ * β₃ * (n ⊗ n) -
                     2μ̄ * β₄ * c_dev_n2
    end

    # ---------------------------------------------------------------------------
    # Pull-back: spatial tangent c → material tangent CC (∂S/∂C form)
    # CC[A,B,C,D] = F⁻¹[A,a] F⁻¹[B,b] c[a,b,c,d] F⁻¹[C,c] F⁻¹[D,d]
    # (factor of J cancels: c is in Kirchhoff stress, S = J F⁻¹ σ F⁻ᵀ)
    # ---------------------------------------------------------------------------
    CC = MArray{Tuple{3,3,3,3},T,4,81}(ntuple(_ -> zero(T), Val(81)))
    for A in 1:3, B in 1:3, C in 1:3, D in 1:3
        val = zero(T)
        for a in 1:3, b in 1:3, c in 1:3, d in 1:3
            val += F_inv[A, a] * F_inv[B, b] * CC_spatial[a, b, c, d] * F_inv[C, c] * F_inv[D, d]
        end
        CC[A, B, C, D] = val
    end

    # ---------------------------------------------------------------------------
    # convect_tangent: CC + S → AA (∂P/∂F)
    # ---------------------------------------------------------------------------
    return _convect_tangent(CC, S, F)
end

# ---------------------------------------------------------------------------
# Override CM.material_tangent for FiniteDefJ2Plasticity
#
# The default CM implementation uses the frozen-Fp approximation which
# gives only linear Newton convergence under active plasticity.
# This override uses the exact Simo-Hughes consistent tangent (BOX 9.2).
# ---------------------------------------------------------------------------

function CM.material_tangent(
    ::CM.FiniteDefJ2Plasticity,
    props, Δt,
    ∇u, θ, Z_old, Z_new
)
    F = ∇u + one(∇u)
    _, P, state_new_vec = CM._j2_stress(props, F, Z_old)
    Z_new .= state_new_vec
    return _j2_tangent_simo_hughes(props, F, Z_old, P, state_new_vec)
end
