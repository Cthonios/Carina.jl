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
# Energy kernel (generic): W_int = ∫ Ψ(F) dΩ
# --------------------------------------------------------------------------- #
#
# Returns JxW * Ψ at each quadrature point. FEC.assemble_scalar! sums over
# all QPs and elements to give the total strain energy W_int.

@inline function FEC.energy(
    physics::SolidMechanics,
    interps, x_el,
    t, dt,
    u_el, u_el_old,
    state_old_q, state_new_q,
    props_el,
)
    cell = FEC.map_interpolants(interps, x_el)
    (; JxW) = cell
    ∇u_q = FEC.interpolate_field_gradients(physics, cell, u_el)
    ∇u_q = FEC.modify_field_gradients(FEC.ThreeDimensional(), ∇u_q)
    W_q = CM.helmholtz_free_energy(
        physics.constitutive_model, props_el, dt, ∇u_q, 0.0, state_old_q, state_new_q,
    )
    return JxW * W_q
end

# --------------------------------------------------------------------------- #
# Material-point finite-difference tangent ∂P/∂∇u
# --------------------------------------------------------------------------- #
#
# Forward FD with adaptive perturbation (Sierra/SM approach):
#   h_j = max(sqrt(ε_mach) * |∇u_j|, μ * 1e-10)
# where μ is the shear modulus (approximated from props).
# Cost: 9 pk1_stress evaluations (one per ∇u component).

@inline function _fd_material_tangent(model, props, dt, ∇u, state_q)
    # Base stress
    s0 = zero(state_q)
    P0 = CM.pk1_stress(model, props, dt, ∇u, 0.0, state_q, s0)

    # Adaptive perturbation (Sierra/SM approach, Wallin 2001):
    # h_j = max(sqrt(ε_mach) * |∇u_j|, floor)
    # Floor must be small enough for the strain scale but large enough
    # to avoid FP cancellation. Use 1e-10 as absolute floor.
    sqrt_eps = 1.49e-8  # sqrt(machine epsilon)
    h_floor  = 1e-10

    A = MArray{Tuple{3,3,3,3},Float64,4,81}(undef)
    for k in 1:3, l in 1:3
        idx = 3 * (l - 1) + k   # column-major flat index
        val = ∇u.data[idx]
        h = max(sqrt_eps * abs(val), h_floor)
        ∇u_p = Tensor{2,3}((i,j) -> ∇u[i,j] + (i==k && j==l ? h : 0.0))
        s_p = zero(state_q)
        P_p = CM.pk1_stress(model, props, dt, ∇u_p, 0.0, state_q, s_p)
        for i in 1:3, j in 1:3
            A[i,j,k,l] = (P_p[i,j] - P0[i,j]) / h
        end
    end
    return Tensor{4,3,Float64,81}(ntuple(i -> A[i], Val(81)))
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

    # Use state_old_q as the starting state — same as the residual kernel.
    # The tangent must be the Jacobian of the same function the residual evaluates.
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
# Small-strain specializations for LinearElastic
#
# ConstitutiveModels.LinearElastic.pk1_stress computes P = J·σ·F⁻ᵀ, the
# geometrically-nonlinear push-forward of the Cauchy stress.  This makes the
# problem nonlinear even though the constitutive law is linear, and does NOT
# match Norma's infinitesimal-kinematics formulation (where K·U = F exactly
# and Newton converges in one iteration).
#
# These overloads bypass the push-forward:
#   residual:          f_int = ∫ G^T · σ_v dV,  σ = C:sym(∇u)
#   stiffness:         K_el  = ∫ G^T · C · G dV (constant; C evaluated at ∇u=0)
#   stiffness_action:  K·v   = ∫ G^T · C · (G·v) dV  (same constant C)
#
# At ∇u=0: J=1, σ=0, so the geometric terms in ∂P/∂∇u vanish and
# CM.material_tangent reduces to the small-strain C.  This is consistent with
# the residual and Newton converges in one iteration for any strain level.
# --------------------------------------------------------------------------- #

@inline function FEC.residual(
    physics::SolidMechanics{CM.LinearElastic, NP, NS},
    interps, x_el,
    t, dt,
    u_el, u_el_old,
    state_old_q, state_new_q,
    props_el,
) where {NP, NS}
    cell = FEC.map_interpolants(interps, x_el)
    (; ∇N_X, JxW) = cell

    ∇u_q = FEC.interpolate_field_gradients(physics, cell, u_el)
    ∇u_q = FEC.modify_field_gradients(FEC.ThreeDimensional(), ∇u_q)

    # Small-strain Cauchy stress σ = C:ε,  ε = sym(∇u)
    σ_q = CM.cauchy_stress(
        physics.constitutive_model, props_el, dt, ∇u_q, 0.0, state_old_q, state_new_q,
    )
    # SymmetricTensor{2,3} → Tensor{2,3} (column-major, matching FEC convention)
    T_el = eltype(σ_q)
    σ_full = Tensor{2, 3, T_el, 9}((
        σ_q[1, 1], σ_q[2, 1], σ_q[3, 1],
        σ_q[1, 2], σ_q[2, 2], σ_q[3, 2],
        σ_q[1, 3], σ_q[2, 3], σ_q[3, 3],
    ))
    P_v = FEC.extract_stress(FEC.ThreeDimensional(), σ_full)
    G   = FEC.discrete_gradient(FEC.ThreeDimensional(), ∇N_X)
    return JxW * G * P_v
end

@inline function FEC.energy(
    physics::SolidMechanics{CM.LinearElastic, NP, NS},
    interps, x_el,
    t, dt,
    u_el, u_el_old,
    state_old_q, state_new_q,
    props_el,
) where {NP, NS}
    cell = FEC.map_interpolants(interps, x_el)
    (; JxW) = cell
    ∇u_q = FEC.interpolate_field_gradients(physics, cell, u_el)
    ∇u_q = FEC.modify_field_gradients(FEC.ThreeDimensional(), ∇u_q)
    W_q = CM.helmholtz_free_energy(
        physics.constitutive_model, props_el, dt, ∇u_q, 0.0, state_old_q, state_new_q,
    )
    return JxW * W_q
end

@inline function FEC.stiffness(
    physics::SolidMechanics{CM.LinearElastic, NP, NS},
    interps, x_el,
    t, dt,
    u_el, u_el_old,
    state_old_q, state_new_q,
    props_el,
) where {NP, NS}
    cell = FEC.map_interpolants(interps, x_el)
    (; ∇N_X, JxW) = cell

    ∇u_q = FEC.interpolate_field_gradients(physics, cell, u_el)
    ∇u_q = FEC.modify_field_gradients(FEC.ThreeDimensional(), ∇u_q)

    # Constant small-strain tangent C = ∂P/∂∇u|_{∇u=0}
    A_q = CM.material_tangent(
        physics.constitutive_model, props_el, dt, zero(∇u_q), 0.0, state_old_q, state_new_q,
    )

    A_v = FEC.extract_stiffness(FEC.ThreeDimensional(), A_q)
    G   = FEC.discrete_gradient(FEC.ThreeDimensional(), ∇N_X)
    return JxW * G * A_v * G'
end

@inline function FEC.stiffness_action(
    physics::SolidMechanics{CM.LinearElastic, NP, NS},
    interps, x_el,
    t, dt,
    u_el, u_el_old, v_el,
    state_old_q, state_new_q,
    props_el,
) where {NP, NS}
    cell = FEC.map_interpolants(interps, x_el)
    (; ∇N_X, JxW) = cell

    ∇u_q = FEC.interpolate_field_gradients(physics, cell, u_el)
    ∇u_q = FEC.modify_field_gradients(FEC.ThreeDimensional(), ∇u_q)

    A_q = CM.material_tangent(
        physics.constitutive_model, props_el, dt, zero(∇u_q), 0.0, state_old_q, state_new_q,
    )

    A_v = FEC.extract_stiffness(FEC.ThreeDimensional(), A_q)
    G   = FEC.discrete_gradient(FEC.ThreeDimensional(), ∇N_X)
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

    # Build element mass matrix in interleaved DOF ordering:
    #   M_el[3*(n-1)+d, 3*(m-1)+d'] = δ(d,d') * N[n] * N[m]
    # i.e. kron(N*N', I_3).  The FEC assembly infrastructure expects
    # rows/cols in the same interleaved order as discrete_gradient, so
    # "N_vec * N_vec'" with a block-ordered N_vec would be wrong.
    N_nodes = size(N, 1)
    NDOF    = 3 * N_nodes
    tup = zeros(SVector{NDOF * NDOF, eltype(N)})
    for n in 1:N_nodes
        for m in 1:N_nodes
            Nnm = N[n] * N[m]
            for d in 1:3
                r = 3 * (n - 1) + d
                c = 3 * (m - 1) + d
                linear_idx = r + NDOF * (c - 1)   # column-major flat index
                tup = setindex(tup, Nnm, linear_idx)
            end
        end
    end
    M_el = SMatrix{NDOF, NDOF, eltype(N), NDOF * NDOF}(tup.data)
    return JxW * physics.density * M_el
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

    # Correct M·v in interleaved DOF ordering:
    #   (M·v)[3*(n-1)+d] = N[n] * Σ_m N[m] * v_el[3*(m-1)+d]
    # i.e. per-direction dot products, NOT a single dot over all DOFs.
    N_nodes = size(N, 1)
    s1 = sum(N[m] * v_el[3 * (m - 1) + 1] for m in 1:N_nodes)
    s2 = sum(N[m] * v_el[3 * (m - 1) + 2] for m in 1:N_nodes)
    s3 = sum(N[m] * v_el[3 * (m - 1) + 3] for m in 1:N_nodes)
    tup = zeros(SVector{3 * N_nodes, eltype(v_el)})
    for n in 1:N_nodes
        k = 3 * (n - 1)
        tup = setindex(tup, N[n] * s1, k + 1)
        tup = setindex(tup, N[n] * s2, k + 2)
        tup = setindex(tup, N[n] * s3, k + 3)
    end
    return JxW * physics.density * tup
end

# --------------------------------------------------------------------------- #
# Characteristic element length kernel (for stable time step estimation).
# Returns a scalar per QP — all QPs in an element give the same value.
# Uses current (deformed) coordinates: X + u.
# --------------------------------------------------------------------------- #

function element_char_length(
    physics::SolidMechanics, interps, x_el,
    t, dt, u_el, u_el_old,
    state_old_q, state_new_q, props_el,
)
    x_cur = x_el + u_el
    ndim = 3
    nnpe = length(x_cur) ÷ ndim
    T = eltype(x_cur)
    cx = cy = cz = zero(T)
    for i in 1:nnpe
        cx += x_cur[(i-1)*ndim + 1]
        cy += x_cur[(i-1)*ndim + 2]
        cz += x_cur[(i-1)*ndim + 3]
    end
    cx /= nnpe; cy /= nnpe; cz /= nnpe
    total = zero(T)
    for i in 1:nnpe
        dx = x_cur[(i-1)*ndim + 1] - cx
        dy = x_cur[(i-1)*ndim + 2] - cy
        dz = x_cur[(i-1)*ndim + 3] - cz
        total += sqrt(dx*dx + dy*dy + dz*dz)
    end
    return 2 * total / nnpe
end
