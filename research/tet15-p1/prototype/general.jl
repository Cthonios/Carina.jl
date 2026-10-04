# The general form of the element (note, sec:general, case 2): the material is
# called at the modified deformation gradient F~ = s F, s = (J~/J)^(1/3),
# J~ = theta^-1(P_h theta(J)), and the element is the stationarity of
# Pi[u] = sum_q w_q j_q W(F~_q).  This script checks the formulas the
# implementation will use, on one element, before any kernel is written:
#
#   1. Residual.  The hand-derived stress of eq. (P-general-material),
#        P_q = s_q P~_q - p~_q J~_q F_q^-T + pbar_q theta'(J_q) J_q F_q^-T,
#        pbar = P_h( p~ / theta'(J~) ),  p~ = tr(sigma~)/3,
#      scattered against grad N, against the gradient of Pi by forward-mode
#      automatic differentiation over the 45 nodal displacements.
#   2. Tangent.  The closed form of eqs. (tangent-general) and (d2s) against
#      the Hessian of Pi by automatic differentiation, and its symmetry.
#   3. Reductions the note claims.  (a) With a material that has the split
#      and a quadratic volumetric energy (the J2 model, theta = J - 1), the
#      general residual and tangent equal those of the implemented split
#      form, and the material's state is the same, in the elastic and in the
#      plastic regime.  (b) With the split and a non-quadratic volumetric
#      energy (neo-Hookean, theta = J - 1), the general residual equals the
#      case-1 residual with the second projection pbar = P_h W_vol'(thetabar).
#      (c) With the constant projection and theta = J - 1,
#      P_q = s_q P~_q + p~ (J_q - J~) F_q^-T.
#
# The element is the fifteen-node enriched tetrahedron in its nodal basis on
# a distorted TETRA10 geometry, with the 14-point Keast rule; both volumetric
# variables, theta = log J (the default) and theta = J - 1, and both
# projections, constant and linear.  The materials come from
# ConstitutiveModels through the generic interface alone
# (helmholtz_free_energy, pk1_stress, material_tangent).
#
#   julia --project=<Carina> research/tet15-p1/prototype/general.jl

include("element.jl")
using ForwardDiff, Printf, Random
using Tensors
import ConstitutiveModels as CM

# --------------------------------------------------------------------------
# Volumetric variable: theta(J), its derivatives and its inverse
# --------------------------------------------------------------------------
struct LogJ end
struct JMinusOne end
θ(::LogJ, J)  = log(J);  θ1(::LogJ, J)  = 1 / J;  θ2(::LogJ, J)  = -1 / J^2;  θinv(::LogJ, t)  = exp(t)
θ(::JMinusOne, J) = J - 1; θ1(::JMinusOne, J) = one(J); θ2(::JMinusOne, J) = zero(J); θinv(::JMinusOne, t) = t + 1
name(::LogJ) = "log J"; name(::JMinusOne) = "J - 1"

# --------------------------------------------------------------------------
# One element: physical gradients of the fifteen nodal functions at the
# quadrature points, the weights, and the projection basis.
# --------------------------------------------------------------------------
struct Element
    ∇N::Vector{Matrix{Float64}}     # per point, 15 x 3
    JxW::Vector{Float64}
    χ::Vector{Vector{Float64}}      # per point, length 1 or 4
end

function element(pd::Int; distort = 0.08)
    X = tet10_coords(; distort)
    A = nodal_transform()
    pts, wts = keast14()
    el = ref_element(2)
    ∇N = Matrix{Float64}[]; JxW = Float64[]; χ = Vector{Float64}[]
    for q in eachindex(wts)
        ξ = pts[:, q]
        _, dNr = enriched_shape(ξ)
        dNr = (A * dNr)
        Jm = X * Matrix(RFE.shape_function_gradient(el, ξ))
        push!(∇N, dNr / Jm)
        push!(JxW, wts[q] * det(Jm))
        push!(χ, collect(pressure_basis(Val(pd == 0 ? 1 : 4), ξ)))
    end
    return Element(∇N, JxW, χ)
end

const NDOF = 45
grad_u(e::Element, q, u) = Tensor{2, 3}((i, j) -> sum(u[3(a - 1) + i] * e.∇N[q][a, j] for a in 1:15))
# Direction k = 3(a-1)+i: the gradient of the nodal basis function (a, i).
dir_grad(e::Element, q, k) = (a = (k - 1) ÷ 3 + 1; i = (k - 1) % 3 + 1;
                              Tensor{2, 3}((m, n) -> m == i ? e.∇N[q][a, n] : 0.0))

function projection(e::Element, vt, u)
    T = eltype(u); nc = length(e.χ[1])
    M = zeros(T, nc, nc); b = zeros(T, nc)
    for q in eachindex(e.JxW)
        J = det(grad_u(e, q, u) + one(Tensor{2, 3, T}))
        M .+= e.JxW[q] .* (e.χ[q] * e.χ[q]')
        b .+= (e.JxW[q] * θ(vt, J)) .* e.χ[q]
    end
    return M \ b, M
end

# --------------------------------------------------------------------------
# Materials through the generic interface
# --------------------------------------------------------------------------
struct Material{M}
    model::M
    props::Vector{Float64}
    nstate::Int
end
W(m::Material, Zo, Zn, F) = CM.helmholtz_free_energy(m.model, m.props, Zo, Zn, 0.0, F - one(F), 0.0)
P(m::Material, Zo, Zn, F) = CM.pk1_stress(m.model, m.props, Zo, Zn, 0.0, F - one(F), 0.0)
A(m::Material, Zo, Zn, F) = CM.material_tangent(m.model, m.props, Zo, Zn, 0.0, F - one(F), 0.0)
state0(m::Material, T = Float64) = m.nstate == 0 ? T[] : T.(CM.initialize_state(m.model))

const E0, ν0 = 1.0e3, 0.3
const INPUTS = Dict{String, Any}("density" => 1.0, "Young's modulus" => E0, "Poisson's ratio" => ν0)
neohookean() = (m = CM.Hyperelastic(CM.NeoHookean()); Material(m, CM.initialize_props(m, INPUTS), 0))
j2(σy)       = (m = CM.FiniteDefJ2Plasticity();
                Material(m, CM.initialize_props(m, merge(INPUTS, Dict("yield stress" => σy, "hardening modulus" => 0.1 * E0))), 10))

# --------------------------------------------------------------------------
# The general element
# --------------------------------------------------------------------------
# Energy, generic in the element type so that ForwardDiff can differentiate it.
function energy(e::Element, vt, m::Material, Zo, u::AbstractVector{T}) where {T}
    θ̄, _ = projection(e, vt, u)
    Π = zero(T)
    for q in eachindex(e.JxW)
        F = grad_u(e, q, u) + one(Tensor{2, 3, T})
        J = det(F)
        J̃ = θinv(vt, dot(e.χ[q], θ̄))
        s = cbrt(J̃ / J)
        Π += e.JxW[q] * W(m, Zo[q], zeros(T, m.nstate), s * F)
    end
    return Π
end

# The quantities of the second pass at every point, kept for the tangent.
struct Point{T}
    F::Tensor{2, 3, T, 9}; Finv::Tensor{2, 3, T, 9}; J::T; J̃::T; s::T
    P̃::Tensor{2, 3, T, 9}; p̃::T; A::Tensor{4, 3, T, 81}; Zn::Vector{T}
end

function passes(e::Element, vt, m::Material, Zo, u)
    θ̄, M = projection(e, vt, u)
    nc = length(θ̄)
    pts = Point{Float64}[]
    b̄ = zeros(nc)
    for q in eachindex(e.JxW)
        F = grad_u(e, q, u) + one(Tensor{2, 3, Float64})
        J = det(F); J̃ = θinv(vt, dot(e.χ[q], θ̄)); s = cbrt(J̃ / J)
        F̃ = s * F
        Zn = zeros(m.nstate); Zn2 = zeros(m.nstate)
        P̃ = P(m, Zo[q], Zn, F̃)
        Aq = A(m, Zo[q], Zn2, F̃)
        p̃ = (P̃ ⊡ F̃) / (3J̃)
        b̄ .+= (e.JxW[q] * p̃ / θ1(vt, J̃)) .* e.χ[q]
        push!(pts, Point(F, inv(F), J, J̃, s, P̃, p̃, Aq, Zn))
    end
    return pts, M \ b̄, M, θ̄
end

function residual(e::Element, vt, m::Material, Zo, u)
    pts, p̄, _, _ = passes(e, vt, m, Zo, u)
    R = zeros(NDOF)
    for q in eachindex(e.JxW)
        pt = pts[q]
        FinvT = transpose(pt.Finv)
        Pq = pt.s * pt.P̃ - (pt.p̃ * pt.J̃) * FinvT + (dot(e.χ[q], p̄) * θ1(vt, pt.J) * pt.J) * FinvT
        for k in 1:NDOF
            R[k] += e.JxW[q] * (Pq ⊡ dir_grad(e, q, k))
        end
    end
    return R, pts
end

# Closed-form tangent: eqs. (tangent-general) and (d2s), applied to every pair
# of nodal directions.
function tangent(e::Element, vt, m::Material, Zo, u)
    pts, p̄, M, θ̄ = passes(e, vt, m, Zo, u)
    nq = length(e.JxW)
    # First variations per direction k and point q.
    δF  = [dir_grad(e, q, k) for q in 1:nq, k in 1:NDOF]
    δJ  = [pts[q].J * (transpose(pts[q].Finv) ⊡ δF[q, k]) for q in 1:nq, k in 1:NDOF]
    δθ̄  = [M \ sum(e.JxW[q] * θ1(vt, pts[q].J) * δJ[q, k] .* e.χ[q] for q in 1:nq) for k in 1:NDOF]
    δJ̃  = [dot(e.χ[q], δθ̄[k]) / θ1(vt, pts[q].J̃) for q in 1:nq, k in 1:NDOF]
    δs  = [pts[q].s / 3 * (δJ̃[q, k] / pts[q].J̃ - δJ[q, k] / pts[q].J) for q in 1:nq, k in 1:NDOF]
    δF̃  = [pts[q].s * δF[q, k] + δs[q, k] * pts[q].F for q in 1:nq, k in 1:NDOF]
    K = zeros(NDOF, NDOF)
    for k in 1:NDOF, l in k:NDOF
        # Second variations of J and of thetabar for the pair (k, l).
        δ2J = [pts[q].J * ((transpose(pts[q].Finv) ⊡ δF[q, k]) * (transpose(pts[q].Finv) ⊡ δF[q, l])
                           - tr(pts[q].Finv ⋅ δF[q, k] ⋅ pts[q].Finv ⋅ δF[q, l])) for q in 1:nq]
        δ2θ̄ = M \ sum(e.JxW[q] * (θ2(vt, pts[q].J) * δJ[q, k] * δJ[q, l] + θ1(vt, pts[q].J) * δ2J[q]) .* e.χ[q]
                      for q in 1:nq)
        v = 0.0
        for q in 1:nq
            pt = pts[q]
            δ2J̃ = dot(e.χ[q], δ2θ̄) / θ1(vt, pt.J̃) - θ2(vt, pt.J̃) / θ1(vt, pt.J̃) * δJ̃[q, k] * δJ̃[q, l]
            δ2s = δs[q, k] / 3 * (δJ̃[q, l] / pt.J̃ - δJ[q, l] / pt.J) +
                  pt.s / 3 * (δ2J̃ / pt.J̃ - δJ̃[q, k] * δJ̃[q, l] / pt.J̃^2 - δ2J[q] / pt.J + δJ[q, k] * δJ[q, l] / pt.J^2)
            v += e.JxW[q] * (δF̃[q, k] ⊡ pt.A ⊡ δF̃[q, l] +
                             pt.P̃ ⊡ (δs[q, k] * δF[q, l] + δs[q, l] * δF[q, k] + δ2s * pt.F))
        end
        K[k, l] = v; K[l, k] = v
    end
    return K
end

# --------------------------------------------------------------------------
# The implemented split form, for the J2 reduction (theta = J - 1):
# P_q = P_iso(F_q) + kappa thetabar_q theta'(J) J F^-T, tangent
# sum JxW [dF1 : A_iso : dF2 + pbar (dg/dF : dF2) : dF1] + kappa d1thetabar' M d2thetabar.
# --------------------------------------------------------------------------
function split_residual_tangent(e::Element, m::Material, Zo, u)
    vt = JMinusOne()
    θ̄, M = projection(e, vt, u)
    κ = CM.bulk_modulus(m.model, m.props)
    nq = length(e.JxW)
    R = zeros(NDOF); K = zeros(NDOF, NDOF)
    Zn_all = Vector{Float64}[]
    δθ̄ = zeros(length(θ̄), NDOF)
    data = []
    for q in 1:nq
        ∇u = grad_u(e, q, u); F = ∇u + one(∇u); J = det(F); Finv = inv(F); FinvT = transpose(Finv)
        Zn = zeros(m.nstate); Zn2 = zeros(m.nstate)
        Piso = CM.isochoric_pk1_stress(m.model, m.props, Zo[q], Zn, 0.0, ∇u, 0.0)
        Aiso = CM.isochoric_material_tangent(m.model, m.props, Zo[q], Zn2, 0.0, ∇u, 0.0)
        push!(Zn_all, Zn)
        p̄ = κ * dot(e.χ[q], θ̄)
        Pq = Piso + (p̄ * J) * FinvT          # theta' = 1
        for k in 1:NDOF
            R[k] += e.JxW[q] * (Pq ⊡ dir_grad(e, q, k))
            δθ̄[:, k] .+= e.JxW[q] * J * (FinvT ⊡ dir_grad(e, q, k)) .* e.χ[q]
        end
        push!(data, (F, J, Finv, FinvT, Aiso, p̄))
    end
    δθ̄ = M \ δθ̄
    for q in 1:nq
        F, J, Finv, FinvT, Aiso, p̄ = data[q]
        for k in 1:NDOF, l in 1:NDOF
            δ1 = dir_grad(e, q, k); δ2 = dir_grad(e, q, l)
            dg = (J * (FinvT ⊡ δ2)) * FinvT - J * (FinvT ⋅ transpose(δ2) ⋅ FinvT)   # theta'' = 0, theta' = 1
            K[k, l] += e.JxW[q] * (δ1 ⊡ Aiso ⊡ δ2 + p̄ * (dg ⊡ δ1))
        end
    end
    K .+= κ .* (δθ̄' * M * δθ̄)
    return R, K, Zn_all
end

# Case 1 for neo-Hookean (theta = J - 1): W_vol = kappa/2 (1/2 (J^2 - 1) - log J),
# W_vol' = kappa/2 (J - 1/J), P_iso = d/dF [mu/2 (I1bar - 3)],
# P_q = P_iso + pbar_q J F^-T with pbar = P_h W_vol'(1 + thetabar).
function case1_residual(e::Element, m::Material, u)
    vt = JMinusOne()
    θ̄, M = projection(e, vt, u)
    κ, μ = m.props[2], m.props[3]
    Wiso(F) = 0.5μ * (tr(transpose(F) ⋅ F) / cbrt(det(F))^2 - 3)
    b̄ = zeros(length(θ̄))
    nq = length(e.JxW)
    for q in 1:nq
        J̃ = 1 + dot(e.χ[q], θ̄)
        b̄ .+= (e.JxW[q] * 0.5κ * (J̃ - 1 / J̃)) .* e.χ[q]
    end
    p̄ = M \ b̄
    R = zeros(NDOF)
    for q in 1:nq
        F = grad_u(e, q, u) + one(Tensor{2, 3, Float64}); J = det(F)
        Pq = Tensors.gradient(Wiso, F) + (dot(e.χ[q], p̄) * J) * transpose(inv(F))
        for k in 1:NDOF
            R[k] += e.JxW[q] * (Pq ⊡ dir_grad(e, q, k))
        end
    end
    return R
end

# --------------------------------------------------------------------------
# The checks
# --------------------------------------------------------------------------
rel(a, b) = norm(a - b) / max(norm(b), eps())

rng = MersenneTwister(7)
# A displacement with strains of about 20%: a stretch, a shear and a random
# part, so that J and the flow direction vary within the element.
function displacement(rng, amp)
    nodes = enriched_nodes()
    u = zeros(NDOF)
    for (a, ξ) in enumerate(nodes)
        x = ξ
        u[3a - 2] = 0.15x[1] + 0.2x[2] + amp * randn(rng)
        u[3a - 1] = -0.1x[2] + 0.05x[3]^2 + amp * randn(rng)
        u[3a]     = 0.1x[3] - 0.12x[1] * x[2] + amp * randn(rng)
    end
    return u
end
u = displacement(rng, 0.02)

# Jacobian of the hand residual by central differences, the check that applies
# to a dissipative material, whose stress is not the derivative of the
# Helmholtz energy during plastic flow.
function fd_tangent(e::Element, vt, m::Material, Zo, u; h = 1.0e-6)
    K = zeros(NDOF, NDOF)
    for k in 1:NDOF
        up = copy(u); up[k] += h
        um = copy(u); um[k] -= h
        K[:, k] = (residual(e, vt, m, Zo, up)[1] - residual(e, vt, m, Zo, um)[1]) / 2h
    end
    return K
end

println("General element, one distorted TETRA15, 14-point rule, strains about 20%.")
println("Columns: hand residual against the AD gradient of the energy, hand tangent against the AD")
println("Hessian of the energy, hand tangent against the central-difference Jacobian of the hand")
println("residual (h = 1e-6), asymmetry of the hand tangent.  The energy columns do not apply to the")
println("plastic J2 model, whose stress is not the derivative of its Helmholtz energy during flow.")
println("In the plastic rows the central-difference column measures the J2 model's own tangent, which")
println("ConstitutiveModels forms by BOX 9.2 of Simo and Hughes as the symmetric part of the Jacobian")
println("of its stress update; the remainder is the antisymmetric part that BOX 9.2 drops.  Reduction")
println("(a) below shows that the general element reproduces the split element's tangent to rounding,")
println("so whatever remains in these rows is the material's and not the element's.\n")
@printf("%-14s %-7s %-12s %10s %10s %10s %10s\n", "material", "theta", "projection", "R vs dW", "K vs d2W", "K vs dR", "asymmetry")
worst = 0.0; worst_fd = 0.0; worst_fd_c = 0.0
for (mname, mat, conservative) in (("neo-Hookean", neohookean(), true), ("J2 elastic", j2(1.0e30), true),
                                   ("J2 plastic", j2(0.02E0), false))
    Zo = [state0(mat) for _ in 1:14]
    for vt in (LogJ(), JMinusOne()), pd in (0, 1)
        e = element(pd)
        R, _ = residual(e, vt, mat, Zo, u)
        K = tangent(e, vt, mat, Zo, u)
        r3 = rel(K, fd_tangent(e, vt, mat, Zo, u)); r4 = norm(K - K') / norm(K)
        global worst = max(worst, r4)
        conservative ? (global worst_fd_c = max(worst_fd_c, r3)) : (global worst_fd = max(worst_fd, r3))
        if conservative
            f = v -> energy(e, vt, mat, Zo, v)
            r1 = rel(R, ForwardDiff.gradient(f, u)); r2 = rel(K, ForwardDiff.hessian(f, u))
            global worst = max(worst, r1, r2)
            @printf("%-14s %-7s %-12s %10.2e %10.2e %10.2e %10.2e\n", mname, name(vt), pd == 0 ? "constant" : "linear", r1, r2, r3, r4)
        else
            @printf("%-14s %-7s %-12s %10s %10s %10.2e %10.2e\n", mname, name(vt), pd == 0 ? "constant" : "linear", "n/a", "n/a", r3, r4)
        end
    end
end
println("\n   worst of the AD and symmetry columns: ", @sprintf("%.1e", worst), "  (target: 1e-12 or below)")
println("   worst of the central-difference column, conservative rows: ", @sprintf("%.1e", worst_fd_c), "  (target: 1e-7 or below)")
println("   worst of the central-difference column, plastic rows: ", @sprintf("%.1e", worst_fd), "  (the material's tangent)")

println("\nReduction (a): J2 (split, quadratic in theta = J - 1): general against the split form.")
@printf("%-12s %-12s %10s %10s %10s\n", "regime", "projection", "residual", "tangent", "state")
for (mname, mat) in (("elastic", j2(1.0e30)), ("plastic", j2(0.02E0)))
    Zo = [state0(mat) for _ in 1:14]
    for pd in (0, 1)
        e = element(pd)
        R, pts = residual(e, JMinusOne(), mat, Zo, u)
        K = tangent(e, JMinusOne(), mat, Zo, u)
        Rs, Ks, Zn = split_residual_tangent(e, mat, Zo, u)
        dz = maximum(norm(pts[q].Zn - Zn[q]) for q in 1:14)
        plast = maximum(pts[q].Zn[10] for q in 1:14)
        @printf("%-12s %-12s %10.2e %10.2e %10.2e   (max eqps %.3f)\n", mname, pd == 0 ? "constant" : "linear",
                rel(R, Rs), rel(K, Ks), dz, plast)
    end
end

println("\nReduction (b): neo-Hookean (split, W_vol not quadratic, theta = J - 1): general against case 1.")
let mat = neohookean(), Zo = [state0(mat) for _ in 1:14]
    for pd in (0, 1)
        e = element(pd)
        R, _ = residual(e, JMinusOne(), mat, Zo, u)
        @printf("   %-12s residual %10.2e\n", pd == 0 ? "constant" : "linear", rel(R, case1_residual(e, mat, u)))
    end
end

println("\nReduction (c): constant projection, theta = J - 1: P_q = s P~ + p~ (J - J~) F^-T at every point.")
let mat = neohookean(), Zo = [state0(mat) for _ in 1:14]
    e = element(0)
    pts, p̄, _, _ = passes(e, JMinusOne(), mat, Zo, u)
    worst_c = 0.0
    for q in 1:14
        pt = pts[q]; FinvT = transpose(pt.Finv)
        Pgen = pt.s * pt.P̃ - (pt.p̃ * pt.J̃) * FinvT + (p̄[1] * pt.J) * FinvT
        Pred = pt.s * pt.P̃ + (pt.p̃ * (pt.J - pt.J̃)) * FinvT
        worst_c = max(worst_c, rel(Pgen, Pred))
    end
    @printf("   worst over points %10.2e   (pbar - p~ = %.2e)\n", worst_c, abs(p̄[1] - pts[1].p̃))
end
