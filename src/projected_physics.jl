# Solid mechanics with a projected volumetric strain (the mean-dilatation
# formulation of research/tet15-p1/note.tex).  Two forms are implemented: the
# split form, for a material whose stored energy has an exact
# volumetric-isochoric split with a quadratic volumetric energy (note, section
# "Reduction to the mean-dilatation formulation"), and the general form, for
# any material (note, section "General materials", case 2).  Both are
# described below; the form is a type parameter of the physics, and the
# kernels dispatch on it.
#
# Split form
# ----------
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
#
# General form
# ------------
#
# For any material, of which only the stored energy W(F), the stress
# P(F) = ∂W/∂F and the tangent A(F) = ∂²W/∂F∂F are used, the volumetric
# variable θ(J) is a property of the element, not of the material: log J or
# J − 1, strictly increasing with θ(1) = 0 and θ'(1) = 1.  With
#
#     J̃ := θ⁻¹(P_h θ(J)),   s := (J̃/J)^{1/3},   F̃ := s F,   det F̃ = J̃,
#
# the element is the stationarity of Π[u] = ∫ W(F̃(u)) dV: the material is
# evaluated at the pointwise isochoric deformation J^{-1/3} F and at the
# projected volume J̃, and its internal variables are updated there.  With
# P̃ := P(F̃) and p̃ := (P̃ : F̃)/(3 J̃), the mean Cauchy stress at F̃, the stress
# that multiplies the gradient of the test function at a quadrature point is
#
#     P_q = s_q P̃_q − p̃_q J̃_q F_q⁻ᵀ + p̄_q θ'(J_q) J_q F_q⁻ᵀ,
#     p̄ = P_h( p̃ / θ'(J̃) ),
#
# the material's stress at F̃ scaled to the pointwise volume, with its mean
# part replaced by the projected mean stress.  The residual takes three passes
# over the quadrature points: θ̄ = M⁻¹ b as above; F̃_q, one call of the
# material at each point, and b̄_m = Σ_q JxW_q χ_m(ξ_q) p̃_q/θ'(J̃_q), from
# which p̄ = M⁻¹ b̄; and the scatter of P_q.  The stresses P̃_q of the second
# pass are kept for the third.
#
# The element matrix and the diagonal kernels of the general form are the
# closed form of "The element tangent of the general form in closed form"
# below: the quadrature-point matrices plus two corrections of rank at most
# the number of projection functions, with the material's tangent evaluated
# once per quadrature point.  The matrix-free action is obtained by
# forward-mode differentiation of the element residual along the direction:
# the nodal displacements carry dual numbers with one partial, so the
# dependence of J̃ and of p̄ on every quadrature point of the element, through
# the two projections, is differentiated exactly.  The material is not called
# with dual numbers: at a dual F̃ the value of P̃ is the material's stress at
# the value of F̃, and its derivative is A(F̃) : δF̃ with the material's own
# tangent.  For a material whose tangent is the derivative of its stress both
# are the derivative of the residual; for the J2 model in plastic flow, whose
# tangent (BOX 9.2 of Simo and Hughes) is the symmetric part of that
# derivative, they are the tangent the split form uses.
#
# For a material with the split and θ the material's own volumetric strain,
# W(F̃) = κ/2 (P_h θ)² + W_iso(F̄), and the general form reproduces the split
# form to rounding: energy, residual, tangent and internal variables.

# --------------------------------------------------------------------------- #
# The volumetric variable θ(J) of the element
# --------------------------------------------------------------------------- #

"""
    VolumetricVariable

The volumetric strain measure θ(J) whose L² projection the element computes:
`LogJ()` (θ = log J) or `JMinusOne()` (θ = J − 1).  `MaterialVolumetricVariable()`
stands for the measure of a material with the volumetric-isochoric split that
is neither of the two; the split form then evaluates θ through the material.
"""
abstract type VolumetricVariable end
struct LogJ <: VolumetricVariable end
struct JMinusOne <: VolumetricVariable end
struct MaterialVolumetricVariable <: VolumetricVariable end

@inline _θ(::LogJ, J)    = log(J)
@inline _θ1(::LogJ, J)   = one(J) / J
@inline _θ2(::LogJ, J)   = -one(J) / (J * J)
@inline _θinv(::LogJ, t) = exp(t)

@inline _θ(::JMinusOne, J)    = J - one(J)
@inline _θ1(::JMinusOne, J)   = one(J)
@inline _θ2(::JMinusOne, J)   = zero(J)
@inline _θinv(::JMinusOne, t) = t + one(t)

volumetric_variable_name(::LogJ) = "log J"
volumetric_variable_name(::JMinusOne) = "J - 1"
volumetric_variable_name(::MaterialVolumetricVariable) = "the material's own measure"

# The measure θ(J) of a material with the split, identified by its values at
# two Jacobians.
function _material_volumetric_variable(cm::CM.AbstractConstitutiveModel)
    Js = (0.6, 1.4)
    all(J -> isapprox(CM.volumetric_strain(cm, J), J - 1; rtol = 1e-12), Js) && return JMinusOne()
    all(J -> isapprox(CM.volumetric_strain(cm, J), log(J); rtol = 1e-12), Js) && return LogJ()
    return MaterialVolumetricVariable()
end

# --------------------------------------------------------------------------- #
# The physics
# --------------------------------------------------------------------------- #

"""
    SplitForm, GeneralForm

The two forms of the projected element.  `SplitForm`: the volumetric energy
κ/2 θ² of a material with the volumetric-isochoric split is evaluated at the
projected strain and the isochoric response at the pointwise deformation.
`GeneralForm`: any material is evaluated at the modified deformation gradient
F̃ = (J̃/J)^{1/3} F, J̃ = θ⁻¹(P_h θ(J)).  See the header of this file.
"""
abstract type VolumetricForm end
struct SplitForm <: VolumetricForm end
struct GeneralForm <: VolumetricForm end

volumetric_form_name(::SplitForm) = "split"
volumetric_form_name(::GeneralForm) = "general"

"""
    ProjectedSolidMechanics(cm, degree; form = :automatic, volumetric_strain = nothing)

Solid mechanics in the mean-dilatation formulation: the volumetric strain
θ(J) is replaced by its L² projection onto the element-wise polynomials of
degree `degree` (0 or 1) in the reference coordinates.

`form` is `:split`, `:general` or `:automatic`.  The split form requires the
volumetric-isochoric split of `cm`
(`ConstitutiveModels.has_volumetric_isochoric_split`) and evaluates the
volumetric energy κ/2 θ² at the projected strain, with θ the material's own
measure.  The general form accepts any material and evaluates it at the
deformation gradient whose volume is the projected one.  `:automatic`
selects the split form when `cm` has the split and the general form
otherwise.

`volumetric_strain` is `LogJ()`, `JMinusOne()` or `nothing`.  For the general
form it selects θ, and `nothing` means `LogJ()`.  For the split form θ is the
material's, and a value that differs from it is an error, since the
pressure κ θ̄ of the split form is the stationarity condition of κ/2 θ²
written in the material's θ.

The kernels are assembled by element (`FiniteElementContainers.assembly_granularity`
is `ByElement()`), since the projection couples the quadrature points of an
element.
"""
struct ProjectedSolidMechanics{Model <: CM.AbstractConstitutiveModel, NP, NS, PD,
                               Form <: VolumetricForm, VV <: VolumetricVariable} <: FEC.AbstractPhysics{3, NP, NS}
    constitutive_model::Model
end

const _SplitProjected   = ProjectedSolidMechanics{<:Any, <:Any, <:Any, <:Any, SplitForm}
const _GeneralProjected = ProjectedSolidMechanics{<:Any, <:Any, <:Any, <:Any, GeneralForm}

function ProjectedSolidMechanics(cm::CM.AbstractConstitutiveModel, degree::Int;
                                 form::Symbol = :automatic,
                                 volumetric_strain::Union{Nothing, VolumetricVariable} = nothing)
    degree in (0, 1) || error(
        "The projection degree must be 0 (element-wise constant) or 1 (element-wise " *
        "linear); got $degree.")
    form in (:automatic, :split, :general) || error(
        "The volumetric form must be :automatic, :split or :general; got :$form.")
    has_split = CM.has_volumetric_isochoric_split(cm)
    form == :split && !has_split && error(
        "The split form of the projected volumetric formulation requires a constitutive " *
        "model with an exact volumetric-isochoric split and a quadratic volumetric energy " *
        "(ConstitutiveModels.has_volumetric_isochoric_split); $(typeof(cm)) has none.  " *
        "Use the general form.")
    volumetric_strain isa MaterialVolumetricVariable && error(
        "The volumetric strain of the projected element must be log J or J - 1.")
    split = form == :split || (form == :automatic && has_split)
    if split
        own = _material_volumetric_variable(cm)
        if volumetric_strain !== nothing && volumetric_strain !== own
            error("The split form evaluates the volumetric energy κ/2 θ² in the volumetric " *
                  "strain of the material, θ = $(volumetric_variable_name(own)) for " *
                  "$(typeof(cm)); the volumetric strain " *
                  "$(volumetric_variable_name(volumetric_strain)) differs from it.  Remove " *
                  "the volumetric strain, set it to the material's, or select the general form.")
        end
        F, VV = SplitForm, typeof(own)
    else
        F, VV = GeneralForm, typeof(volumetric_strain === nothing ? LogJ() : volumetric_strain)
    end
    NP = CM.num_properties(cm)
    NS = CM.num_state_variables(cm)
    return ProjectedSolidMechanics{typeof(cm), NP, NS, degree, F, VV}(cm)
end

FEC.assembly_granularity(::ProjectedSolidMechanics) = FEC.ByElement()

projection_degree(::ProjectedSolidMechanics{Model, NP, NS, PD}) where {Model, NP, NS, PD} = PD
volumetric_form(::ProjectedSolidMechanics{M, NP, NS, PD, F}) where {M, NP, NS, PD, F} = F()
volumetric_variable(::ProjectedSolidMechanics{M, NP, NS, PD, F, VV}) where {M, NP, NS, PD, F, VV} = VV()

# θ(J) of the element: the material's measure in the split form, the
# element's own in the general form.
@inline _theta(physics::_SplitProjected, J) = CM.volumetric_strain(physics.constitutive_model, J)
@inline _theta(physics::_GeneralProjected, J) = _θ(volumetric_variable(physics), J)

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
    # The displacement may carry dual numbers (the general form's tangent);
    # the geometry does not.
    Tu = promote_type(T, eltype(u_el))
    M  = zero(SMatrix{NC, NC, T})
    b  = zero(SVector{NC, Tu})
    everted = false
    for q in 1:RFE.num_cell_quadrature_points(ref_fe)
        interps = FEC._cell_interpolants(ref_fe, q)
        cell = FEC.map_interpolants(interps, x_el)
        J = det(_gradient_at(physics, cell, u_el) + one(Tensor{2, 3, Tu, 9}))
        everted = everted | (J <= zero(J))
        χ = _projection_basis(PD, interps.ξ)
        M = M + cell.JxW * (χ * χ')
        # θ is evaluated at J = 1 where J ≤ 0, so that log J cannot raise an
        # error in device code; the kernels return NaN in that case.
        b = b + (cell.JxW * _theta(physics, J > zero(J) ? J : one(J))) * χ
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
    physics::_SplitProjected, ref_fe, x_el, t, dt, u_el, u_el_old,
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
    physics::_SplitProjected, ref_fe, x_el, t, dt, u_el, u_el_old,
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
    physics::_SplitProjected, ref_fe, x_el, t, dt, u_el, u_el_old,
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
    physics::_SplitProjected, ref_fe, x_el, t, dt, u_el, u_el_old, v_el,
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
    physics::_SplitProjected, ref_fe, x_el, dt, u_el, states, props_el, ::Val{column},
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
    physics::_SplitProjected, ref_fe, x_el, t, dt, u_el, u_el_old,
    states::FEC.ElementState, props_el,
)
    return _projected_block_entries(physics, ref_fe, x_el, dt, u_el, states, props_el, Val(0))
end

@inline function (::StiffnessBlockColumn{K})(
    physics::_SplitProjected, ref_fe, x_el, t, dt, u_el, u_el_old,
    states::FEC.ElementState, props_el,
) where {K}
    return _projected_block_entries(physics, ref_fe, x_el, dt, u_el, states, props_el, Val(K))
end

@inline function (dg::NewmarkDiagonal)(
    physics::_SplitProjected, ref_fe, x_el, t, dt, u_el, u_el_old,
    states::FEC.ElementState, props_el,
)
    d = _projected_block_entries(physics, ref_fe, x_el, dt, u_el, states, props_el, Val(0))
    return d + _mass_diagonal_element(ref_fe, x_el, dg.c_M * props_el[1], d)
end

# The diagonal of cρ times the consistent mass, in the layout of `d`.
@inline function _mass_diagonal_element(ref_fe, x_el, cρ, d::SVector{NDOF, T}) where {NDOF, T}
    m = zero(SVector{NDOF, T})
    for q in 1:RFE.num_cell_quadrature_points(ref_fe)
        cell = FEC.map_interpolants(FEC._cell_interpolants(ref_fe, q), x_el)
        (; N, JxW) = cell
        m = m + SVector{NDOF, T}(ntuple(Val(NDOF)) do k
            n = (k - 1) ÷ 3 + 1
            cρ * JxW * N[n] * N[n]
        end)
    end
    return m
end

# --------------------------------------------------------------------------- #
# General form: the material at F̃ = s F
# --------------------------------------------------------------------------- #

struct _ProjectedTag end

# The material's stress at ∇ũ = F̃ − I.  At a real argument this is the
# material's stress; at a dual argument the value is the stress at the value
# of ∇ũ and the partials are A(F̃) : δF̃ with the material's tangent, so that
# the material is never called with dual numbers and a model with internal
# variables writes real values into its state.
@inline function _material_pk1(model, props, state_old, state_new, dt,
                               ∇ũ::Tensor{2, 3, T, 9}) where {T <: AbstractFloat}
    return CM.pk1_stress(model, props, state_old, state_new, dt, ∇ũ, zero(T))
end

@inline function _material_pk1(model, props, state_old, state_new, dt,
                               ∇ũ::Tensor{2, 3, D, 9}) where {D <: ForwardDiff.Dual}
    V = ForwardDiff.valtype(D)
    N = ForwardDiff.npartials(D)
    ∇ũ0 = Tensor{2, 3, V, 9}(ntuple(i -> ForwardDiff.value(∇ũ.data[i]), Val(9)))
    P0 = CM.pk1_stress(model, props, state_old, state_new, dt, ∇ũ0, zero(V))
    A  = CM.material_tangent(model, props, state_old, state_new, dt, ∇ũ0, zero(V))
    A_v = FEC.extract_stiffness(FEC.ThreeDimensional(), A)
    dP = ntuple(Val(N)) do k
        A_v * SVector{9, V}(ntuple(i -> ForwardDiff.partials(∇ũ.data[i], k), Val(9)))
    end
    return Tensor{2, 3, D, 9}(ntuple(Val(9)) do i
        D(P0.data[i], ForwardDiff.Partials{N, V}(ntuple(k -> dP[k][i], Val(N))))
    end)
end

# The second pass at quadrature point q: F̃, the material's stress P̃ at F̃,
# the mean Cauchy stress p̃ = (P̃ : F̃)/(3 J̃), J̃ and s.  `scratch` is true for
# the output kernel, which must not change the internal variables.
@inline function _general_point(physics::_GeneralProjected, ref_fe, x_el, dt, u_el, states,
                                props_el, θ̄, q, ::Val{scratch}) where {scratch}
    vv = volumetric_variable(physics)
    PD = Val(projection_degree(physics))
    interps = FEC._cell_interpolants(ref_fe, q)
    cell = FEC.map_interpolants(interps, x_el)
    ∇u = _gradient_at(physics, cell, u_el)
    F  = ∇u + one(∇u)
    J  = det(F)
    χ  = _projection_basis(PD, interps.ξ)
    J̃  = _θinv(vv, dot(χ, θ̄))
    s  = cbrt(J̃ / J)
    F̃  = s * F
    state_old_q, state_new_q = FEC.state_variables(states, q)
    Z_new = scratch ? similar(state_new_q) : state_new_q
    P̃ = _material_pk1(physics.constitutive_model, props_el, state_old_q, Z_new, dt, F̃ - one(F̃))
    p̃ = (P̃ ⊡ F̃) / (3 * J̃)
    return P̃, p̃, J̃, s, χ
end

# The first two passes of the residual: θ̄ and M⁻¹, the quantities of the
# second pass at every quadrature point, p̄, whether the element is everted,
# and the displacement used (zero where the element is everted).  Generic in
# the element type of `u_el`, so that it runs on dual numbers.
@inline function _general_passes(physics::_GeneralProjected, ref_fe, x_el, dt, u_el, states,
                                 props_el, ::Val{scratch}) where {scratch}
    vv = volumetric_variable(physics)
    NQ = RFE.num_cell_quadrature_points(ref_fe)
    # Pass 1: θ̄ = P_h θ(J).
    θ̄, Minv, everted = _project(physics, ref_fe, x_el, u_el)
    # Pass 2: the material at F̃ and b̄ = Σ_q JxW_q χ_q p̃_q / θ'(J̃_q).  Where
    # the element is everted the material is called at F = I, so that no
    # model is asked to evaluate J ≤ 0; the caller returns NaN.
    u_safe = everted ? zero(u_el) : u_el
    θ̄_safe = everted ? zero(θ̄) : θ̄
    pts = ntuple(q -> _general_point(physics, ref_fe, x_el, dt, u_safe, states, props_el,
                                     θ̄_safe, q, Val(scratch)), Val(NQ))
    b̄ = zero(θ̄)
    for q in 1:NQ
        P̃, p̃, J̃, s, χ = pts[q]
        JxW = FEC.map_interpolants(FEC._cell_interpolants(ref_fe, q), x_el).JxW
        b̄ = b̄ + (JxW * p̃ / _θ1(vv, J̃)) * χ
    end
    p̄ = Minv * b̄
    return pts, θ̄_safe, Minv, p̄, everted, u_safe
end

# The three passes of the residual.  Returns the stresses P_q of the third
# pass, one per quadrature point, and whether the element is everted.
@inline function _general_stresses(physics::_GeneralProjected, ref_fe, x_el, dt, u_el, states,
                                   props_el, ::Val{scratch}) where {scratch}
    vv = volumetric_variable(physics)
    NQ = RFE.num_cell_quadrature_points(ref_fe)
    pts, _, _, p̄, everted, u_safe = _general_passes(physics, ref_fe, x_el, dt, u_el, states,
                                                     props_el, Val(scratch))
    # Pass 3: P_q = s P̃ − p̃ J̃ F⁻ᵀ + p̄_q θ'(J) J F⁻ᵀ.
    Ps = ntuple(Val(NQ)) do q
        P̃, p̃, J̃, s, χ = pts[q]
        cell = FEC.map_interpolants(FEC._cell_interpolants(ref_fe, q), x_el)
        F = _gradient_at(physics, cell, u_safe)
        F = F + one(F)
        J = det(F)
        FinvT = inv(F)'
        s * P̃ + (dot(χ, p̄) * _θ1(vv, J) * J - p̃ * J̃) * FinvT
    end
    return Ps, everted
end

@inline function _general_residual(physics::_GeneralProjected, ref_fe, x_el, dt, u_el, states, props_el)
    Tu = promote_type(eltype(x_el), eltype(u_el))
    NDOF = 3 * RFE.num_cell_dofs(ref_fe)
    Ps, everted = _general_stresses(physics, ref_fe, x_el, dt, u_el, states, props_el, Val(false))
    R = zero(SVector{NDOF, Tu})
    for q in 1:RFE.num_cell_quadrature_points(ref_fe)
        cell = FEC.map_interpolants(FEC._cell_interpolants(ref_fe, q), x_el)
        R = R + _scatter_qp(cell.∇N_X, Ps[q], convert(Tu, cell.JxW))
    end
    return everted ? Tu(NaN) * R : R
end

@inline function FEC.residual(
    physics::_GeneralProjected, ref_fe, x_el, t, dt, u_el, u_el_old,
    states::FEC.ElementState, props_el,
)
    return _general_residual(physics, ref_fe, x_el, dt, u_el, states, props_el)
end

@inline function FEC.energy(
    physics::_GeneralProjected, ref_fe, x_el, t, dt, u_el, u_el_old,
    states::FEC.ElementState, props_el,
)
    model = physics.constitutive_model
    vv = volumetric_variable(physics)
    PD = Val(projection_degree(physics))
    NQ = RFE.num_cell_quadrature_points(ref_fe)
    θ̄, _, everted = _project(physics, ref_fe, x_el, u_el)
    u_safe = everted ? zero(u_el) : u_el
    θ̄_safe = everted ? zero(θ̄) : θ̄
    return ntuple(Val(NQ)) do q
        interps = FEC._cell_interpolants(ref_fe, q)
        cell = FEC.map_interpolants(interps, x_el)
        ∇u = _gradient_at(physics, cell, u_safe)
        F  = ∇u + one(∇u)
        J̃  = _θinv(vv, dot(_projection_basis(PD, interps.ξ), θ̄_safe))
        F̃  = cbrt(J̃ / det(F)) * F
        state_old_q, state_new_q = FEC.state_variables(states, q)
        W = cell.JxW * CM.helmholtz_free_energy(model, props_el, state_old_q, state_new_q, dt,
                                                F̃ - one(F̃), zero(J̃))
        everted ? oftype(W, NaN) : W
    end
end

@inline function FEC.stiffness_action(
    physics::_GeneralProjected, ref_fe, x_el, t, dt, u_el, u_el_old, v_el,
    states::FEC.ElementState, props_el,
)
    T = eltype(u_el)
    NDOF = length(u_el)
    D = ForwardDiff.Dual{_ProjectedTag, T, 1}
    u_d = SVector{NDOF, D}(ntuple(k -> D(u_el[k], ForwardDiff.Partials{1, T}((v_el[k],))), Val(NDOF)))
    R_d = _general_residual(physics, ref_fe, x_el, dt, u_d, states, props_el)
    return SVector{NDOF, T}(ntuple(k -> ForwardDiff.partials(R_d[k], 1), Val(NDOF)))
end

# --------------------------------------------------------------------------- #
# The element tangent of the general form in closed form
# --------------------------------------------------------------------------- #
#
# At quadrature point q the stress of the third pass,
#
#     P_q = s P̃ + c F⁻ᵀ,   c = p̄_q θ'(J) J − p̃ J̃,
#
# depends on δF_q = G_qᵀ δu (G_q the discrete gradient), and on the element
# through dθ̄ and dp̄ only, so that
#
#     dP_q = C_q δF_q + E_q dθ̄ + H_q dp̄,   H_q = θ'(J) J F⁻ᵀ ⊗ χ_q.
#
# With dJ = J F⁻ᵀ : δF, dJ̃ = χ·dθ̄ / θ'(J̃), ds = s/3 (dJ̃/J̃ − dJ/J),
# dF̃ = ds F + s δF, dP̃ = A(F̃) : dF̃ with the material's tangent,
# dp̃ = (dP̃ : F̃ + P̃ : dF̃)/(3 J̃) − p̃ dJ̃/J̃, and d(F⁻ᵀ) = −F⁻ᵀ δFᵀ F⁻ᵀ, the
# matrices C_q (9×9) and E_q (9×NC) follow by collecting the terms.  The two
# projections give
#
#     dθ̄ = M⁻¹ Σ_r w_r θ'(J_r) χ_r (J F⁻ᵀ)_r : δF_r,
#     dp̄ = M⁻¹ Σ_r w_r χ_r d(p̃/θ'(J̃))_r,
#     d(p̃/θ'(J̃)) = q_f · δF + q_t · dθ̄,
#
# and the element tangent is
#
#     K = Σ_q w_q G_q C_q G_qᵀ + (Y_E + Y_H R) M⁻¹ Z_θᵀ + Y_H M⁻¹ Z_pᵀ
#
# with Y_E = Σ_q w_q G_q E_q, Y_H = Σ_q w_q θ'(J_q) J_q (G_q F_q⁻ᵀ) χ_qᵀ,
# R = M⁻¹ Σ_q w_q χ_q q_tᵀ, Z_θ = Σ_q w_q θ'(J_q) (G_q J_q F_q⁻ᵀ) χ_qᵀ and
# Z_p = Σ_q w_q (G_q q_f) χ_qᵀ: the quadrature-point matrices plus two
# corrections of rank at most NC.  It uses the material's tangent once per
# quadrature point, against once per quadrature point and node for the
# derivative by dual numbers, which it equals to rounding (test set
# "Projected volumetric formulation, general form").  The matrix-free action
# stays one dual pass, which uses the tangent once per quadrature point.

# d(F⁻ᵀ)/dF as a 9×9 matrix in the column-major vec order of `_scatter_qp`:
# entry [i + 3(j−1), l + 3(k−1)] = −F⁻¹[k,i] F⁻¹[j,l].
@inline function _dFinvT_matrix(Finv::Tensor{2, 3, T, 9}) where {T}
    return SMatrix{9, 9, T, 81}(ntuple(Val(81)) do lin
        c, r = divrem(lin - 1, 9)
        j, i = divrem(r, 3)
        k, l = divrem(c, 3)
        -Finv[k + 1, i + 1] * Finv[j + 1, l + 1]
    end)
end

@inline _tensor9(v::SVector{9, T}) where {T} = Tensor{2, 3, T, 9}(Tuple(v))

# The linearization of P_q and of p̃/θ'(J̃) at one quadrature point: C, E, the
# coefficient h = θ'(J) J of H = h F⁻ᵀ ⊗ χ, F⁻ᵀ and J F⁻ᵀ as 9-vectors,
# θ'(J), q_f and q_t.
@inline function _general_point_tangent(vv, F::Tensor{2, 3, T, 9}, χ, P̃, p̃, J̃, s, p̄,
                                        A_v) where {T}
    J     = det(F)
    Finv  = inv(F)
    Fv    = SVector{9, T}(F.data)
    P̃v    = SVector{9, T}(P̃.data)
    FinvT = SVector{9, T}(Finv'.data)
    gJ    = J * FinvT
    θ1J, θ2J = _θ1(vv, J), _θ2(vv, J)
    θ1t, θ2t = _θ1(vv, J̃), _θ2(vv, J̃)
    γ   = one(T) / θ1t
    a_f = (-s / (3 * J)) * gJ                    # ds along δF
    a_t = (s * γ / (3 * J̃)) * χ                  # ds along dθ̄
    DFf = Fv * a_f' + s * one(SMatrix{9, 9, T})   # dF̃ along δF
    DFt = Fv * a_t'                               # dF̃ along dθ̄
    w9  = (A_v' * (s * Fv) + P̃v) / (3 * J̃)
    e_f = DFf' * w9                               # dp̃ along δF
    e_t = DFt' * w9 - (p̃ * γ / J̃) * χ             # dp̃ along dθ̄
    q_f = e_f / θ1t
    q_t = e_t / θ1t - (p̃ * θ2t * γ / (θ1t * θ1t)) * χ
    p̄_q = dot(χ, p̄)
    c   = p̄_q * θ1J * J - p̃ * J̃
    dc_f = (p̄_q * (θ2J * J + θ1J)) * gJ - J̃ * e_f
    dc_t = -J̃ * e_t - (p̃ * γ) * χ
    C = P̃v * a_f' + s * (A_v * DFf) + FinvT * dc_f' + c * _dFinvT_matrix(Finv)
    E = P̃v * a_t' + s * (A_v * DFt) + FinvT * dc_t'
    return C, E, θ1J * J, FinvT, gJ, θ1J, q_f, q_t
end

# The parts of the element tangent.  `column` is −1 for the element matrix
# Σ_q w_q G_q C_q G_qᵀ, 0 for its diagonal and K in 1:3 for column K of each
# node's 3×3 diagonal block.  Returns that part, Y_E + Y_H R, Y_H, Z_θ, Z_p,
# M⁻¹ and whether the element is everted.
@inline function _general_tangent_parts(
    physics::_GeneralProjected, ref_fe, x_el, dt, u_el, states, props_el, ::Val{column},
) where {column}
    model = physics.constitutive_model
    vv = volumetric_variable(physics)
    PD = Val(projection_degree(physics))
    NC = _num_projection_functions(PD)
    T  = eltype(u_el)
    NDOF = length(u_el)
    NQ = RFE.num_cell_quadrature_points(ref_fe)
    pts, _, Minv, p̄, everted, u_safe = _general_passes(physics, ref_fe, x_el, dt, u_el, states,
                                                        props_el, Val(false))
    Kq = column < 0 ? zero(SMatrix{NDOF, NDOF, T}) : zero(SVector{NDOF, T})
    Y_E = zero(SMatrix{NDOF, NC, T})
    Y_H = zero(SMatrix{NDOF, NC, T})
    Z_θ = zero(SMatrix{NDOF, NC, T})
    Z_p = zero(SMatrix{NDOF, NC, T})
    Racc = zero(SMatrix{NC, NC, T})
    for q in 1:NQ
        P̃, p̃, J̃, s, χ = pts[q]
        cell = FEC.map_interpolants(FEC._cell_interpolants(ref_fe, q), x_el)
        w = cell.JxW
        F = _gradient_at(physics, cell, u_safe)
        F = F + one(F)
        state_old_q, state_new_q = FEC.state_variables(states, q)
        A = CM.material_tangent(model, props_el, state_old_q, state_new_q, dt, s * F - one(F), zero(T))
        A_v = FEC.extract_stiffness(FEC.ThreeDimensional(), A)
        C, E, h, FinvT, gJ, θ1J, q_f, q_t = _general_point_tangent(vv, F, χ, P̃, p̃, J̃, s, p̄, A_v)
        if column < 0
            G = FEC.discrete_gradient(FEC.ThreeDimensional(), cell.∇N_X)
            Kq = Kq + w * (G * C * G')
        elseif column == 0
            Kq = Kq + _diag_qp(cell.∇N_X, C, w)
        else
            Kq = Kq + _block_col_qp(cell.∇N_X, C, w, Val(column))
        end
        GE = hcat(ntuple(m -> _scatter_qp(cell.∇N_X, _tensor9(E[:, m]), w), Val(NC))...)
        Y_E = Y_E + GE
        Y_H = Y_H + _scatter_qp(cell.∇N_X, _tensor9(FinvT), w * h) * χ'
        Z_θ = Z_θ + _scatter_qp(cell.∇N_X, _tensor9(gJ), w * θ1J) * χ'
        Z_p = Z_p + _scatter_qp(cell.∇N_X, _tensor9(q_f), w) * χ'
        Racc = Racc + w * (χ * q_t')
    end
    R = Minv * Racc
    return Kq, Y_E + Y_H * R, Y_H, Z_θ, Z_p, Minv, everted
end

@inline function FEC.stiffness(
    physics::_GeneralProjected, ref_fe, x_el, t, dt, u_el, u_el_old,
    states::FEC.ElementState, props_el,
)
    T = eltype(u_el)
    Kq, Y_θ, Y_H, Z_θ, Z_p, Minv, everted = _general_tangent_parts(
        physics, ref_fe, x_el, dt, u_el, states, props_el, Val(-1))
    K = Kq + Y_θ * Minv * Z_θ' + Y_H * Minv * Z_p'
    return everted ? T(NaN) * K : K
end

# Diagonal (column = 0) or column `column` of every node's 3×3 diagonal block
# of the element tangent: the quadrature-point part plus the entries of the
# two corrections, (Y M⁻¹)[k, :] · Z[kc, :].
@inline function _general_block_entries(
    physics::_GeneralProjected, ref_fe, x_el, dt, u_el, states, props_el, ::Val{column},
) where {column}
    T = eltype(u_el)
    NDOF = length(u_el)
    d, Y_θ, Y_H, Z_θ, Z_p, Minv, everted = _general_tangent_parts(
        physics, ref_fe, x_el, dt, u_el, states, props_el, Val(column))
    # Bound once before the closure (see `_projected_block_entries`).
    d_q = d
    YθM = Y_θ * Minv
    YHM = Y_H * Minv
    Zθ, Zp = Z_θ, Z_p
    NC = size(Minv, 1)
    out = SVector{NDOF, T}(ntuple(Val(NDOF)) do k
        n = (k - 1) ÷ 3 + 1
        kc = column == 0 ? k : 3 * (n - 1) + column
        acc = zero(T)
        for m in 1:NC
            acc = acc + YθM[k, m] * Zθ[kc, m] + YHM[k, m] * Zp[kc, m]
        end
        d_q[k] + acc
    end)
    return everted ? T(NaN) * out : out
end

@inline function (::StiffnessDiagonal)(
    physics::_GeneralProjected, ref_fe, x_el, t, dt, u_el, u_el_old,
    states::FEC.ElementState, props_el,
)
    return _general_block_entries(physics, ref_fe, x_el, dt, u_el, states, props_el, Val(0))
end

@inline function (::StiffnessBlockColumn{K})(
    physics::_GeneralProjected, ref_fe, x_el, t, dt, u_el, u_el_old,
    states::FEC.ElementState, props_el,
) where {K}
    return _general_block_entries(physics, ref_fe, x_el, dt, u_el, states, props_el, Val(K))
end

@inline function (dg::NewmarkDiagonal)(
    physics::_GeneralProjected, ref_fe, x_el, t, dt, u_el, u_el_old,
    states::FEC.ElementState, props_el,
)
    d = _general_block_entries(physics, ref_fe, x_el, dt, u_el, states, props_el, Val(0))
    return d + _mass_diagonal_element(ref_fe, x_el, dg.c_M * props_el[1], d)
end

# Output: the deformation gradient and the Cauchy stress σ = J⁻¹ P Fᵀ of the
# stress P_q of the third pass, at each quadrature point.
function quadrature_field_output(
    physics::_GeneralProjected, ref_fe, x_el, t, dt, u_el, u_el_old,
    states::FEC.ElementState, props_el,
)
    NQ = RFE.num_cell_quadrature_points(ref_fe)
    Ps, _ = _general_stresses(physics, ref_fe, x_el, dt, u_el, states, props_el, Val(true))
    return ntuple(Val(NQ)) do q
        cell = FEC.map_interpolants(FEC._cell_interpolants(ref_fe, q), x_el)
        ∇u = _gradient_at(physics, cell, u_el)
        F = ∇u + one(∇u)
        σ = symmetric((1 / det(F)) * dot(Ps[q], transpose(F)))
        QuadratureFieldOutput(F, σ)
    end
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
    physics::_SplitProjected, ref_fe, x_el, t, dt, u_el, u_el_old,
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
