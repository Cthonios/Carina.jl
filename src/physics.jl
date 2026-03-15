# SolidMechanics physics kernel for 3D solid mechanics.
# Connects FiniteElementContainers.jl assembly machinery with
# ConstitutiveModels.jl constitutive models.

import FiniteElementContainers as FEC
import ConstitutiveModels as CM
using StaticArrays
using Tensors

# --------------------------------------------------------------------------- #
# SolidMechanics struct
# NP: number of material properties (from the constitutive model)
# --------------------------------------------------------------------------- #

struct SolidMechanics{Model <: CM.AbstractConstitutiveModel, NP, NS} <: FEC.AbstractPhysics{3, NP, NS}
    constitutive_model::Model
    density::Float64
end

"""
    SolidMechanics(cm, density)

Construct a `SolidMechanics` physics object wrapping constitutive model `cm`.
`density` is used for the mass matrix (dynamics).
"""
function SolidMechanics(
    cm::CM.AbstractConstitutiveModel{NP, NS},
    density::Float64 = 0.0,
) where {NP, NS}
    return SolidMechanics{typeof(cm), NP, NS}(cm, density)
end

# --------------------------------------------------------------------------- #
# FEC interface: properties
# --------------------------------------------------------------------------- #

"""
    create_solid_mechanics_properties(cm, material_inputs)

Compute a flat `SVector` of material properties (e.g. κ, μ for NeoHookean)
from a `Dict{String}` of material inputs (e.g. "Young's modulus", "Poisson's ratio").
The returned vector is passed as `props_el` to each physics kernel call.
"""
function create_solid_mechanics_properties(
    cm::CM.AbstractConstitutiveModel{NP, NS},
    material_inputs::Dict{String},
) where {NP, NS}
    props_vec = CM.initialize_props(cm, material_inputs)
    return SVector{NP, Float64}(props_vec)
end

# `create_properties` tells FEC how to allocate the per-block property storage.
# We return zeros here; the caller is responsible for filling in the actual values
# (see `create_solid_mechanics_properties`).
function FEC.create_properties(::SolidMechanics{Model, NP, NS}) where {Model, NP, NS}
    return SVector{NP, Float64}(zeros(NP))
end

# `create_initial_state` returns the initial per-quadrature-point state vector.
# For NS=0 (hyperelastic) the default FEC fallback handles this.
# For NS>0 (e.g. plasticity) we call CM.initialize_state.
function FEC.create_initial_state(physics::SolidMechanics{Model, NP, NS}) where {Model, NP, NS}
    return CM.initialize_state(physics.constitutive_model)
end

# --------------------------------------------------------------------------- #
# FEC interface: residual (internal force vector, per quadrature point)
# --------------------------------------------------------------------------- #

@inline function FEC.residual(
    physics::SolidMechanics,
    interps, x_el,
    t, dt,
    u_el, u_el_old,
    state_old_q, state_new_q,
    props_el,
)
    # Map reference interpolants to physical coordinates
    cell = FEC.map_interpolants(interps, x_el)
    (; ∇N_X, JxW) = cell

    # Displacement gradient at quadrature point: SMatrix{3,3} → Tensor{2,3}
    ∇u_q = FEC.interpolate_field_gradients(physics, cell, u_el)
    ∇u_q = FEC.modify_field_gradients(FEC.ThreeDimensional(), ∇u_q)

    # PK1 stress (analytical or AD-backed depending on the model)
    P_q = CM.pk1_stress(
        physics.constitutive_model, props_el, dt, ∇u_q, 0.0, state_old_q, state_new_q,
    )

    # Voigt-ordered stress vector and B-matrix, then internal force
    P_v = FEC.extract_stress(FEC.ThreeDimensional(), P_q)
    G   = FEC.discrete_gradient(FEC.ThreeDimensional(), ∇N_X)
    return JxW * G * P_v
end

# --------------------------------------------------------------------------- #
# FEC interface: stiffness (tangent stiffness matrix, per quadrature point)
# --------------------------------------------------------------------------- #

@inline function FEC.stiffness(
    physics::SolidMechanics,
    interps, x_el,
    t, dt,
    u_el, u_el_old,
    state_old_q, state_new_q,
    props_el,
)
    cell = FEC.map_interpolants(interps, x_el)
    (; ∇N_X, JxW) = cell

    ∇u_q = FEC.interpolate_field_gradients(physics, cell, u_el)
    ∇u_q = FEC.modify_field_gradients(FEC.ThreeDimensional(), ∇u_q)

    # Material tangent ∂P/∂F via AD of pk1_stress
    A_q = CM.material_tangent(
        physics.constitutive_model, props_el, dt, ∇u_q, 0.0, state_old_q, state_new_q,
    )

    A_v = FEC.extract_stiffness(FEC.ThreeDimensional(), A_q)
    G   = FEC.discrete_gradient(FEC.ThreeDimensional(), ∇N_X)
    return JxW * G * A_v * G'
end

# --------------------------------------------------------------------------- #
# FEC interface: stiffness_action (K·v per quadrature point, no matrix formed)
# --------------------------------------------------------------------------- #

@inline function FEC.stiffness_action(
    physics::SolidMechanics,
    interps, x_el,
    t, dt,
    u_el, u_el_old, v_el,
    state_old_q, state_new_q,
    props_el,
)
    cell = FEC.map_interpolants(interps, x_el)
    (; ∇N_X, JxW) = cell

    ∇u_q = FEC.interpolate_field_gradients(physics, cell, u_el)
    ∇u_q = FEC.modify_field_gradients(FEC.ThreeDimensional(), ∇u_q)

    A_q = CM.material_tangent(
        physics.constitutive_model, props_el, dt, ∇u_q, 0.0, state_old_q, state_new_q,
    )

    A_v = FEC.extract_stiffness(FEC.ThreeDimensional(), A_q)
    G   = FEC.discrete_gradient(FEC.ThreeDimensional(), ∇N_X)
    # K_q·v_el = JxW·G·A_v·(G'·v_el) — avoids forming K_q (24×24)
    return JxW * G * (A_v * (G' * v_el))
end

# --------------------------------------------------------------------------- #
# FEC interface: mass (consistent mass matrix, per quadrature point)
# --------------------------------------------------------------------------- #

@inline function FEC.mass(
    physics::SolidMechanics,
    interps, x_el,
    t, dt,
    u_el, u_el_old,
    state_old_q, state_new_q,
    props_el,
)
    cell = FEC.map_interpolants(interps, x_el)
    (; N, JxW) = cell

    # N_vec: shape function vector expanded for all DOF components
    N_vec = FEC.discrete_values(FEC.ThreeDimensional(), N)
    return JxW * physics.density * N_vec * N_vec'
end

# --------------------------------------------------------------------------- #
# FEC interface: mass_action (M·v per quadrature point, no matrix formed)
# --------------------------------------------------------------------------- #

@inline function FEC.mass_action(
    physics::SolidMechanics,
    interps, x_el,
    t, dt,
    u_el, u_el_old, v_el,
    state_old_q, state_new_q,
    props_el,
)
    cell = FEC.map_interpolants(interps, x_el)
    (; N, JxW) = cell

    N_vec = FEC.discrete_values(FEC.ThreeDimensional(), N)
    # M_q·v_el = JxW·ρ·(N_vec·v_el)·N_vec — avoids forming M_q (24×24)
    return JxW * physics.density * dot(N_vec, v_el) * N_vec
end
