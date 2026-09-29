# Solid mechanics with a projected volumetric strain (the mean-dilatation
# formulation of research/tet15-p1/note.tex, section "Reduction to the
# mean-dilatation formulation").
#
# For a material whose stored energy splits exactly as
#
#     W(F) = κ/2 θ(J)² + W_iso(F̄),   F̄ = J^{-1/3} F,
#
# the three-field Hu-Washizu functional with the volumetric strain θ̄ and the
# pressure p̄ in one element-wise discontinuous space Q_h reduces to
#
#     Π[u] = ∫ W_iso(F̄(u)) dV + ∫ κ/2 (P_h θ(J(u)))² dV,
#
# with P_h the L² projection onto Q_h.  On one element with quadrature
# points ξ_q, weights w_q and reference Jacobian determinants j_q
# (JxW_q = w_q j_q), and with χ_m(ξ) a basis of Q_h on the element,
#
#     M_mn = Σ_q JxW_q χ_m(ξ_q) χ_n(ξ_q),    b_m = Σ_q JxW_q χ_m(ξ_q) θ(J_q),
#     θ̄ = M⁻¹ b,   θ̄_q = Σ_m χ_m(ξ_q) θ̄_m,   p̄_q = κ θ̄_q.
#
# The stationarity condition of Π gives at each quadrature point the first
# Piola-Kirchhoff stress
#
#     P_q = P_iso,q + p̄_q g_q,   g := ∂θ/∂F = θ'(J) J F⁻ᵀ,
#
# so the residual is the pointwise one with the projected pressure.  The
# tangent couples the quadrature points of the element through θ̄:
#
#     dP_q = dP_iso,q + p̄_q (∂g/∂F)_q : ∇v_q + κ dθ̄_q g_q,
#     dθ̄ = M⁻¹ Σ_q JxW_q χ(ξ_q) (g_q : ∇v_q),
#
# which in matrix form adds κ D M⁻¹ Dᵀ to the sum of the quadrature-point
# matrices, with D = Σ_q JxW_q (G_q vec g_q) χ(ξ_q)ᵀ and G_q the discrete
# gradient of the element.  Q_h is P0 (χ = 1) or P1 in the reference
# coordinates (χ = 1, ξ₁, ξ₂, ξ₃); M is 1×1 or 4×4 and is inverted per element
# per call.  Every kernel makes two passes over the quadrature points, the
# first for θ̄ and the second for the stresses; the geometric mapping is
# recomputed in the second pass rather than stored.
#
# The isochoric response, the volumetric strain measure θ(J) and the bulk
# modulus κ come from the constitutive model through the split interface of
# ConstitutiveModels (has_volumetric_isochoric_split); the model's return map
# runs unchanged at every quadrature point, and no quantity is projected
# except θ.

"""
    ProjectedSolidMechanics(cm, degree)

Solid mechanics in the mean-dilatation formulation: the volumetric strain
θ(J) of the constitutive model `cm` is replaced by its L² projection onto the
element-wise polynomials of degree `degree` (0 or 1) in the reference
coordinates, and the volumetric energy κ/2 θ² is evaluated at the projected
strain.  `cm` must have the volumetric-isochoric split
(`ConstitutiveModels.has_volumetric_isochoric_split`).  The kernels are
assembled by element (`FiniteElementContainers.assembly_granularity` is
`ByElement()`), since
the projection couples the quadrature points of an element.
"""
struct ProjectedSolidMechanics{Model <: CM.AbstractConstitutiveModel, NP, NS, PD} <: FEC.AbstractPhysics{3, NP, NS}
    constitutive_model::Model
end

function ProjectedSolidMechanics(cm::CM.AbstractConstitutiveModel, degree::Int)
    CM.has_volumetric_isochoric_split(cm) || error(
        "The projected volumetric formulation requires a constitutive model with an " *
        "exact volumetric-isochoric split and a quadratic volumetric energy " *
        "(ConstitutiveModels.has_volumetric_isochoric_split); $(typeof(cm)) has none.")
    degree in (0, 1) || error(
        "The projection degree must be 0 (element-wise constant) or 1 (element-wise " *
        "linear); got $degree.")
    NP = CM.num_properties(cm)
    NS = CM.num_state_variables(cm)
    return ProjectedSolidMechanics{typeof(cm), NP, NS, degree}(cm)
end

FEC.assembly_granularity(::ProjectedSolidMechanics) = FEC.ByElement()

projection_degree(::ProjectedSolidMechanics{Model, NP, NS, PD}) where {Model, NP, NS, PD} = PD

# The per-quadrature-point physics of the same model, for the kernels that
# the projection does not change (mass, element length, ...).
@inline function _inner(::ProjectedSolidMechanics{Model, NP, NS, PD}, cm) where {Model, NP, NS, PD}
    return SolidMechanics{Model, NP, NS}(cm)
end
@inline _inner(physics::ProjectedSolidMechanics) = _inner(physics, physics.constitutive_model)

FEC.create_properties(physics::ProjectedSolidMechanics) = FEC.create_properties(_inner(physics))
FEC.create_initial_state(physics::ProjectedSolidMechanics) = FEC.create_initial_state(_inner(physics))

# --------------------------------------------------------------------------- #
# The projection space
# --------------------------------------------------------------------------- #

_num_projection_functions(::Val{0}) = 1
_num_projection_functions(::Val{1}) = 4

@inline _projection_basis(::Val{0}, ξ) = SVector{1, eltype(ξ)}(one(eltype(ξ)))
@inline _projection_basis(::Val{1}, ξ) = SVector{4, eltype(ξ)}(one(eltype(ξ)), ξ[1], ξ[2], ξ[3])

# --------------------------------------------------------------------------- #
# Volumetric kinematics: θ(J), g = ∂θ/∂F = φ(J) F⁻ᵀ with φ = θ'(J) J, and
# ∂g/∂F : δF = φ'(J) J (F⁻ᵀ : δF) F⁻ᵀ − φ F⁻ᵀ δFᵀ F⁻ᵀ with φ' = θ''(J) J + θ'(J).
# --------------------------------------------------------------------------- #

struct _Volumetric{T}
    J::T
    θ::T
    φ::T     # θ'(J) J
    φ1::T    # dφ/dJ
    Finv::Tensor{2, 3, T, 9}
end

@inline function _volumetric(model, ∇u::Tensor{2, 3, T, 9}) where {T}
    F    = ∇u + one(∇u)
    J    = det(F)
    Finv = inv(F)
    θ    = CM.volumetric_strain(model, J)
    θ1   = CM.volumetric_strain_derivative(model, J)
    θ2   = CM.volumetric_strain_second_derivative(model, J)
    return _Volumetric{T}(J, θ, θ1 * J, θ2 * J + θ1, Finv)
end

@inline _g(v::_Volumetric) = v.φ * v.Finv'

# ∂g/∂F : δF
@inline function _dg(v::_Volumetric, δF)
    FinvT = v.Finv'
    return (v.φ1 * v.J * (FinvT ⊡ δF)) * FinvT - v.φ * (FinvT ⋅ δF' ⋅ FinvT)
end

# ∂g/∂F as the 9×9 matrix in the column-major vec order of `_scatter_qp`:
# H[i + 3(j−1), k + 3(l−1)] = φ' J F⁻ᵀ[i,j] F⁻ᵀ[k,l] − φ F⁻ᵀ[i,l] F⁻ᵀ[k,j]
@inline function _dg_matrix(v::_Volumetric{T}) where {T}
    Finv = v.Finv
    c1 = v.φ1 * v.J
    c2 = v.φ
    return SMatrix{9, 9, T, 81}(ntuple(Val(81)) do lin
        l, rem = divrem(lin - 1, 27)
        k, rem = divrem(rem, 9)
        j, i   = divrem(rem, 3)
        i += 1; j += 1; k += 1; l += 1
        c1 * Finv[j, i] * Finv[l, k] - c2 * Finv[l, i] * Finv[j, k]
    end)
end

# --------------------------------------------------------------------------- #
# First pass: the projected volumetric strain of the element
# --------------------------------------------------------------------------- #

@inline function _gradient_at(physics, cell, u_el)
    ∇u = FEC.interpolate_field_gradients(physics, cell, u_el)
    return FEC.modify_field_gradients(FEC.ThreeDimensional(), ∇u)
end

# θ̄ and M⁻¹ of the element; `everted` is true if J ≤ 0 at any quadrature
# point (the kernels then return NaN, as the per-quadrature-point kernels do).
@inline function _project(physics::ProjectedSolidMechanics, ref_fe, x_el, u_el)
    model = physics.constitutive_model
    PD = Val(projection_degree(physics))
    NC = _num_projection_functions(PD)
    T  = eltype(x_el)
    M  = zero(SMatrix{NC, NC, T})
    b  = zero(SVector{NC, T})
    everted = false
    for q in 1:RFE.num_cell_quadrature_points(ref_fe)
        interps = FEC._cell_interpolants(ref_fe, q)
        cell = FEC.map_interpolants(interps, x_el)
        J = det(_gradient_at(physics, cell, u_el) + one(Tensor{2, 3, T, 9}))
        everted = everted | (J <= zero(J))
        χ = _projection_basis(PD, interps.ξ)
        M = M + cell.JxW * (χ * χ')
        b = b + (cell.JxW * CM.volumetric_strain(model, J)) * χ
    end
    Minv = inv(M)
    return Minv * b, Minv, everted
end

# The same with the directional derivative dθ̄ along v_el.
@inline function _project(physics::ProjectedSolidMechanics, ref_fe, x_el, u_el, v_el)
    model = physics.constitutive_model
    PD = Val(projection_degree(physics))
    NC = _num_projection_functions(PD)
    T  = eltype(x_el)
    M  = zero(SMatrix{NC, NC, T})
    b  = zero(SVector{NC, T})
    bv = zero(SVector{NC, T})
    for q in 1:RFE.num_cell_quadrature_points(ref_fe)
        interps = FEC._cell_interpolants(ref_fe, q)
        cell = FEC.map_interpolants(interps, x_el)
        vol = _volumetric(model, _gradient_at(physics, cell, u_el))
        ∇v  = _gradient_at(physics, cell, v_el)
        χ   = _projection_basis(PD, interps.ξ)
        M   = M + cell.JxW * (χ * χ')
        b   = b + (cell.JxW * vol.θ) * χ
        bv  = bv + (cell.JxW * (_g(vol) ⊡ ∇v)) * χ
    end
    Minv = inv(M)
    return Minv * b, Minv * bv, Minv
end

# --------------------------------------------------------------------------- #
# Isochoric directional derivative dP_iso = ∂P_iso/∂∇u : ∇v
# --------------------------------------------------------------------------- #

# Stateless models: one forward-mode dual pass over the isochoric stress.
@inline function _isochoric_dP(
    physics::ProjectedSolidMechanics{Model, NP, 0}, ∇u::Tensor{2, 3, T, 9},
    ∇v::Tensor{2, 3, T, 9}, state_old_q, state_new_q, dt, props_el,
) where {Model, NP, T}
    D = ForwardDiff.Dual{_PK1JVPTag, T, 1}
    ∇u_d = Tensor{2, 3, D, 9}(ntuple(
        i -> D(∇u.data[i], ForwardDiff.Partials{1, T}((∇v.data[i],))), Val(9)))
    P_d = CM.isochoric_pk1_stress(physics.constitutive_model, props_el,
                                  state_old_q, state_new_q, dt, ∇u_d, zero(D))
    return Tensor{2, 3, T, 9}(ntuple(i -> ForwardDiff.partials(P_d.data[i], 1), Val(9)))
end

# Models with state: form the isochoric tangent and contract it.
@inline function _isochoric_dP(
    physics::ProjectedSolidMechanics, ∇u, ∇v, state_old_q, state_new_q, dt, props_el,
)
    A = CM.isochoric_material_tangent(physics.constitutive_model, props_el,
                                      state_old_q, state_new_q, dt, ∇u, 0.0)
    A_v = FEC.extract_stiffness(FEC.ThreeDimensional(), A)
    dP  = A_v * SVector{9, eltype(∇v)}(∇v.data)
    return Tensor{2, 3, eltype(∇v), 9}(dP.data)
end

# --------------------------------------------------------------------------- #
# FEC interface, element level
# --------------------------------------------------------------------------- #

@inline function FEC.residual(
    physics::ProjectedSolidMechanics, ref_fe, x_el, t, dt, u_el, u_el_old,
    states::FEC.ElementState, props_el,
)
    model = physics.constitutive_model
    PD = Val(projection_degree(physics))
    T  = eltype(x_el)
    NDOF = 3 * RFE.num_cell_dofs(ref_fe)
    R = zero(SVector{NDOF, T})
    θ̄, _, everted = _project(physics, ref_fe, x_el, u_el)
    everted && return T(NaN) * R
    κ = CM.bulk_modulus(model, props_el)
    for q in 1:RFE.num_cell_quadrature_points(ref_fe)
        interps = FEC._cell_interpolants(ref_fe, q)
        cell = FEC.map_interpolants(interps, x_el)
        ∇u = _gradient_at(physics, cell, u_el)
        state_old_q, state_new_q = FEC.state_variables(states, q)
        P_iso = CM.isochoric_pk1_stress(model, props_el, state_old_q, state_new_q, dt, ∇u, 0.0)
        vol = _volumetric(model, ∇u)
        p̄ = κ * dot(_projection_basis(PD, interps.ξ), θ̄)
        P = P_iso + p̄ * _g(vol)
        R = R + _scatter_qp(cell.∇N_X, P, cell.JxW)
    end
    return R
end

@inline function FEC.energy(
    physics::ProjectedSolidMechanics, ref_fe, x_el, t, dt, u_el, u_el_old,
    states::FEC.ElementState, props_el,
)
    model = physics.constitutive_model
    PD = Val(projection_degree(physics))
    T  = eltype(x_el)
    NQ = RFE.num_cell_quadrature_points(ref_fe)
    θ̄, _, everted = _project(physics, ref_fe, x_el, u_el)
    κ = CM.bulk_modulus(model, props_el)
    return ntuple(Val(NQ)) do q
        interps = FEC._cell_interpolants(ref_fe, q)
        cell = FEC.map_interpolants(interps, x_el)
        ∇u = _gradient_at(physics, cell, u_el)
        state_old_q, state_new_q = FEC.state_variables(states, q)
        W_iso = CM.isochoric_helmholtz_free_energy(model, props_el, state_old_q, state_new_q, dt, ∇u, 0.0)
        θ̄_q = dot(_projection_basis(PD, interps.ξ), θ̄)
        W = cell.JxW * (W_iso + κ / 2 * θ̄_q * θ̄_q)
        everted ? oftype(W, NaN) : W
    end
end

@inline function FEC.stiffness(
    physics::ProjectedSolidMechanics, ref_fe, x_el, t, dt, u_el, u_el_old,
    states::FEC.ElementState, props_el,
)
    model = physics.constitutive_model
    PD = Val(projection_degree(physics))
    NC = _num_projection_functions(PD)
    T  = eltype(x_el)
    NDOF = 3 * RFE.num_cell_dofs(ref_fe)
    θ̄, Minv, _ = _project(physics, ref_fe, x_el, u_el)
    κ = CM.bulk_modulus(model, props_el)
    K = zero(SMatrix{NDOF, NDOF, T})
    D = zero(SMatrix{NDOF, NC, T})
    for q in 1:RFE.num_cell_quadrature_points(ref_fe)
        interps = FEC._cell_interpolants(ref_fe, q)
        cell = FEC.map_interpolants(interps, x_el)
        ∇u = _gradient_at(physics, cell, u_el)
        state_old_q, state_new_q = FEC.state_variables(states, q)
        A = CM.isochoric_material_tangent(model, props_el, state_old_q, state_new_q, dt, ∇u, 0.0)
        vol = _volumetric(model, ∇u)
        χ = _projection_basis(PD, interps.ξ)
        p̄ = κ * dot(χ, θ̄)
        A_v = FEC.extract_stiffness(FEC.ThreeDimensional(), A) + p̄ * _dg_matrix(vol)
        G = FEC.discrete_gradient(FEC.ThreeDimensional(), cell.∇N_X)
        K = K + cell.JxW * (G * A_v * G')
        D = D + _scatter_qp(cell.∇N_X, _g(vol), cell.JxW) * χ'
    end
    return K + κ * (D * Minv * D')
end

@inline function FEC.stiffness_action(
    physics::ProjectedSolidMechanics, ref_fe, x_el, t, dt, u_el, u_el_old, v_el,
    states::FEC.ElementState, props_el,
)
    model = physics.constitutive_model
    PD = Val(projection_degree(physics))
    T  = eltype(x_el)
    NDOF = 3 * RFE.num_cell_dofs(ref_fe)
    θ̄, dθ̄, _ = _project(physics, ref_fe, x_el, u_el, v_el)
    κ = CM.bulk_modulus(model, props_el)
    Kv = zero(SVector{NDOF, T})
    for q in 1:RFE.num_cell_quadrature_points(ref_fe)
        interps = FEC._cell_interpolants(ref_fe, q)
        cell = FEC.map_interpolants(interps, x_el)
        ∇u = _gradient_at(physics, cell, u_el)
        ∇v = _gradient_at(physics, cell, v_el)
        state_old_q, state_new_q = FEC.state_variables(states, q)
        dP = _isochoric_dP(physics, ∇u, ∇v, state_old_q, state_new_q, dt, props_el)
        vol = _volumetric(model, ∇u)
        χ = _projection_basis(PD, interps.ξ)
        dP = dP + (κ * dot(χ, θ̄)) * _dg(vol, ∇v) + (κ * dot(χ, dθ̄)) * _g(vol)
        Kv = Kv + _scatter_qp(cell.∇N_X, dP, cell.JxW)
    end
    return Kv
end

# The reduced-precision action has no projected form; the exact action is
# returned, and `_fp32_action_is_effective` reports that below.
@inline function stiffness_action_fp32(
    physics::ProjectedSolidMechanics, ref_fe, x_el, t, dt, u_el, u_el_old, v_el,
    states::FEC.ElementState, props_el,
)
    return FEC.stiffness_action(physics, ref_fe, x_el, t, dt, u_el, u_el_old, v_el, states, props_el)
end

_fp32_action_is_effective(::ProjectedSolidMechanics, props_el) = false

# --------------------------------------------------------------------------- #
# Diagonal kernels: the quadrature-point diagonals plus the diagonal of the
# coupling term κ D M⁻¹ Dᵀ, whose entry k is (D M⁻¹)[k, :] ⋅ D[k, :].
# --------------------------------------------------------------------------- #

# One pass over the element for the diagonal kernels.  `column` is 0 for the
# diagonal and K in 1:3 for column K of each node's 3×3 block.
@inline function _projected_block_entries(
    physics::ProjectedSolidMechanics, ref_fe, x_el, dt, u_el, states, props_el, ::Val{column},
) where {column}
    model = physics.constitutive_model
    PD = Val(projection_degree(physics))
    NC = _num_projection_functions(PD)
    T  = eltype(x_el)
    NDOF = 3 * RFE.num_cell_dofs(ref_fe)
    θ̄, Minv, _ = _project(physics, ref_fe, x_el, u_el)
    κ = CM.bulk_modulus(model, props_el)
    d = zero(SVector{NDOF, T})
    D = zero(SMatrix{NDOF, NC, T})
    for q in 1:RFE.num_cell_quadrature_points(ref_fe)
        interps = FEC._cell_interpolants(ref_fe, q)
        cell = FEC.map_interpolants(interps, x_el)
        ∇u = _gradient_at(physics, cell, u_el)
        state_old_q, state_new_q = FEC.state_variables(states, q)
        A = CM.isochoric_material_tangent(model, props_el, state_old_q, state_new_q, dt, ∇u, 0.0)
        vol = _volumetric(model, ∇u)
        χ = _projection_basis(PD, interps.ξ)
        p̄ = κ * dot(χ, θ̄)
        A_v = FEC.extract_stiffness(FEC.ThreeDimensional(), A) + p̄ * _dg_matrix(vol)
        if column == 0
            d = d + _diag_qp(cell.∇N_X, A_v, cell.JxW)
        else
            d = d + _block_col_qp(cell.∇N_X, A_v, cell.JxW, Val(column))
        end
        D = D + _scatter_qp(cell.∇N_X, _g(vol), cell.JxW) * χ'
    end
    # Bound once before the closure: a variable reassigned in the loop above
    # and captured by the closure would be boxed, which a GPU kernel cannot
    # compile.
    d_q = d
    D_e = D
    E = D_e * Minv
    return SVector{NDOF, T}(ntuple(Val(NDOF)) do k
        n = (k - 1) ÷ 3 + 1
        kc = column == 0 ? k : 3 * (n - 1) + column
        s = zero(T)
        for m in 1:NC
            s = s + E[k, m] * D_e[kc, m]
        end
        d_q[k] + κ * s
    end)
end

@inline function (::StiffnessDiagonal)(
    physics::ProjectedSolidMechanics, ref_fe, x_el, t, dt, u_el, u_el_old,
    states::FEC.ElementState, props_el,
)
    return _projected_block_entries(physics, ref_fe, x_el, dt, u_el, states, props_el, Val(0))
end

@inline function (::StiffnessBlockColumn{K})(
    physics::ProjectedSolidMechanics, ref_fe, x_el, t, dt, u_el, u_el_old,
    states::FEC.ElementState, props_el,
) where {K}
    return _projected_block_entries(physics, ref_fe, x_el, dt, u_el, states, props_el, Val(K))
end

@inline function (dg::NewmarkDiagonal)(
    physics::ProjectedSolidMechanics, ref_fe, x_el, t, dt, u_el, u_el_old,
    states::FEC.ElementState, props_el,
)
    d = _projected_block_entries(physics, ref_fe, x_el, dt, u_el, states, props_el, Val(0))
    NDOF = length(d)
    T = eltype(d)
    cρ = dg.c_M * props_el[1]
    m = zero(SVector{NDOF, T})
    for q in 1:RFE.num_cell_quadrature_points(ref_fe)
        cell = FEC.map_interpolants(FEC._cell_interpolants(ref_fe, q), x_el)
        (; N, JxW) = cell
        m = m + SVector{NDOF, T}(ntuple(Val(NDOF)) do k
            n = (k - 1) ÷ 3 + 1
            cρ * JxW * N[n] * N[n]
        end)
    end
    return d + m
end

# --------------------------------------------------------------------------- #
# Kernels the projection does not change: sums of the per-quadrature-point
# kernels of the same model.
# --------------------------------------------------------------------------- #

@inline function _sum_over_quadrature_points(
    kernel, physics::ProjectedSolidMechanics, ref_fe, x_el, t, dt, u_el, u_el_old,
    states::FEC.ElementState, props_el,
)
    inner = _inner(physics)
    state_old_q, state_new_q = FEC.state_variables(states, 1)
    acc = kernel(inner, FEC._cell_interpolants(ref_fe, 1), x_el, t, dt, u_el, u_el_old,
                 state_old_q, state_new_q, props_el)
    for q in 2:RFE.num_cell_quadrature_points(ref_fe)
        state_old_q, state_new_q = FEC.state_variables(states, q)
        acc = acc + kernel(inner, FEC._cell_interpolants(ref_fe, q), x_el, t, dt, u_el, u_el_old,
                           state_old_q, state_new_q, props_el)
    end
    return acc
end

# The consistent mass: the scalar matrix ρ N Nᵀ summed over the quadrature
# points, expanded to the three components once.
@inline function FEC.mass(
    physics::ProjectedSolidMechanics, ref_fe, x_el, t, dt, u_el, u_el_old,
    states::FEC.ElementState, props_el,
)
    cell = FEC.map_interpolants(FEC._cell_interpolants(ref_fe, 1), x_el)
    NN = (cell.JxW * props_el[1]) * (cell.N * cell.N')
    for q in 2:RFE.num_cell_quadrature_points(ref_fe)
        cell = FEC.map_interpolants(FEC._cell_interpolants(ref_fe, q), x_el)
        NN = NN + (cell.JxW * props_el[1]) * (cell.N * cell.N')
    end
    return _expand_mass(NN)
end

for kernel in (:lumped_mass,)
    @eval @inline function FEC.$kernel(
        physics::ProjectedSolidMechanics, ref_fe, x_el, t, dt, u_el, u_el_old,
        states::FEC.ElementState, props_el,
    )
        return _sum_over_quadrature_points(FEC.$kernel, physics, ref_fe, x_el, t, dt,
                                           u_el, u_el_old, states, props_el)
    end
end

@inline function _mass_action_element(physics::ProjectedSolidMechanics, ref_fe, x_el, v_el, props_el)
    T = eltype(x_el)
    NDOF = 3 * RFE.num_cell_dofs(ref_fe)
    Mv = zero(SVector{NDOF, T})
    for q in 1:RFE.num_cell_quadrature_points(ref_fe)
        cell = FEC.map_interpolants(FEC._cell_interpolants(ref_fe, q), x_el)
        Mv = Mv + _mass_qp_action(cell.N, cell.JxW, props_el[1], v_el)
    end
    return Mv
end

@inline function FEC.mass_action(
    physics::ProjectedSolidMechanics, ref_fe, x_el, t, dt, u_el, u_el_old, v_el,
    states::FEC.ElementState, props_el,
)
    return _mass_action_element(physics, ref_fe, x_el, v_el, props_el)
end

@inline function (action::NewmarkAction)(
    physics::ProjectedSolidMechanics, ref_fe, x_el, t, dt, u_el, u_el_old, v_el,
    states::FEC.ElementState, props_el,
)
    Kv = FEC.stiffness_action(physics, ref_fe, x_el, t, dt, u_el, u_el_old, v_el, states, props_el)
    return Kv + action.c_M * _mass_action_element(physics, ref_fe, x_el, v_el, props_el)
end

# The element length is the same at every quadrature point.
function element_char_length(
    physics::ProjectedSolidMechanics, ref_fe, x_el, t, dt, u_el, u_el_old,
    states::FEC.ElementState, props_el,
)
    state_old_q, state_new_q = FEC.state_variables(states, 1)
    h = element_char_length(_inner(physics), FEC._cell_interpolants(ref_fe, 1), x_el, t, dt,
                            u_el, u_el_old, state_old_q, state_new_q, props_el)
    return ntuple(_ -> h, Val(RFE.num_cell_quadrature_points(ref_fe)))
end

# Output: the deformation gradient and the Cauchy stress σ = J⁻¹ P Fᵀ with the
# projected pressure, at each quadrature point.
function quadrature_field_output(
    physics::ProjectedSolidMechanics, ref_fe, x_el, t, dt, u_el, u_el_old,
    states::FEC.ElementState, props_el,
)
    model = physics.constitutive_model
    PD = Val(projection_degree(physics))
    NQ = RFE.num_cell_quadrature_points(ref_fe)
    θ̄, _, _ = _project(physics, ref_fe, x_el, u_el)
    κ = CM.bulk_modulus(model, props_el)
    return ntuple(Val(NQ)) do q
        interps = FEC._cell_interpolants(ref_fe, q)
        cell = FEC.map_interpolants(interps, x_el)
        ∇u = _gradient_at(physics, cell, u_el)
        state_old_q, state_new_q = FEC.state_variables(states, q)
        state_scratch = similar(state_new_q)
        P_iso = CM.isochoric_pk1_stress(model, props_el, state_old_q, state_scratch, dt, ∇u, 0.0)
        vol = _volumetric(model, ∇u)
        p̄ = κ * dot(_projection_basis(PD, interps.ξ), θ̄)
        P = P_iso + p̄ * _g(vol)
        F = ∇u + one(∇u)
        σ = symmetric((1 / vol.J) * dot(P, transpose(F)))
        QuadratureFieldOutput(F, σ)
    end
end
