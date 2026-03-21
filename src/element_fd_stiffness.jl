# Element-level finite-difference stiffness assembly (Sierra/SM approach).
#
# Computes K_el by probing the FEC.residual kernel:
#   K_el[:, j] = (f_int(u_el + h·eⱼ) - f_int(u_el)) / h
#
# This goes through the EXACT same code path as the assembled residual,
# guaranteeing consistency. No analytical tangent needed.
#
# Compared to the material-point FD (∂P/∂∇u), this captures:
# - State handling (state_old/state_new mutation)
# - The full element kinematics chain (u_el → ∇u → F → σ → f_int)
# - All quadrature point contributions summed correctly

import FiniteElementContainers as FEC
import FiniteElementContainers: ReferenceFE, connectivity, _element_level_fields_flat,
    _element_level_properties, _cell_interpolants, _quadrature_level_state,
    num_cell_quadrature_points, block_view, function_space

import ConstitutiveModels as CM
using StaticArrays

# ---------------------------------------------------------------------------
# Compute element internal force by summing FEC.residual over all QPs
# ---------------------------------------------------------------------------

function _element_internal_force(
    physics, ref_fe, x_el, t, dt, u_el, u_el_old,
    state_old_block, state_new_block, props_el, e,
)
    nqp = num_cell_quadrature_points(ref_fe)
    f_el = zero(u_el)  # same SVector type
    for q in 1:nqp
        interps     = _cell_interpolants(ref_fe, q)
        state_old_q = _quadrature_level_state(state_old_block, q, e)
        # Use a scratch for state_new to prevent mutation of the real state
        state_scratch = similar(state_old_q)
        copyto!(state_scratch, state_old_q)  # start from state_old

        f_q = FEC.residual(physics, interps, x_el, t, dt,
                            u_el, u_el_old, state_old_q, state_scratch, props_el)
        f_el = f_el + f_q
    end
    return f_el
end

# ---------------------------------------------------------------------------
# Element-level FD stiffness: K_el via forward differences of residual
# ---------------------------------------------------------------------------

function _element_fd_stiffness(
    physics, ref_fe, x_el, t, dt, u_el, u_el_old,
    state_old_block, state_new_block, props_el, e,
)
    ndof = length(u_el)
    T_el = eltype(u_el)

    # Base internal force
    f0 = _element_internal_force(
        physics, ref_fe, x_el, t, dt, u_el, u_el_old,
        state_old_block, state_new_block, props_el, e,
    )

    # Adaptive perturbation (Sierra/SM approach)
    sqrt_eps = 1.49e-8
    h_floor  = 1e-10

    # Assemble K_el column by column (forward differences)
    K_data = MVector{ndof * ndof, T_el}(undef)

    for j in 1:ndof
        val_j = u_el[j]
        h = max(sqrt_eps * abs(val_j), h_floor)

        # Perturbed u_el
        u_el_p = setindex(u_el, val_j + h, j)

        f_p = _element_internal_force(
            physics, ref_fe, x_el, t, dt, u_el_p, u_el_old,
            state_old_block, state_new_block, props_el, e,
        )

        # Column j of K_el
        for i in 1:ndof
            K_data[(j-1)*ndof + i] = (f_p[i] - f0[i]) / h
        end
    end

    return SMatrix{ndof, ndof, T_el}(K_data)
end

# ---------------------------------------------------------------------------
# Full assembly: replace FEC.assemble_stiffness! with element-level FD
# ---------------------------------------------------------------------------

function assemble_stiffness_fd!(asm, p)
    storage = asm.stiffness_storage
    fill!(storage, zero(eltype(storage)))

    fspace = function_space(asm.dof)
    t  = FEC.current_time(p.times)
    dt = FEC.time_step(p.times)
    conns = fspace.elem_conns
    pattern = asm.matrix_pattern

    for (b, (block_physics, ref_fe, props)) in enumerate(zip(
        values(p.physics), values(fspace.ref_fes), values(p.properties),
    ))
        nelem   = conns.nelems[b]
        coffset = conns.offsets[b]
        state_old_b = block_view(p.state_old, b)
        state_new_b = block_view(p.state_new, b)
        bsi = pattern.block_start_indices[b]
        bels = pattern.block_el_level_sizes[b]

        for e in 1:nelem
            conn     = connectivity(ref_fe, conns.data, e, coffset)
            x_el     = _element_level_fields_flat(p.h1_coords, ref_fe, conn)
            u_el     = _element_level_fields_flat(p.h1_field,  ref_fe, conn)
            u_el_old = _element_level_fields_flat(p.h1_field_old, ref_fe, conn)
            props_el = _element_level_properties(props, e)

            K_el = _element_fd_stiffness(
                block_physics, ref_fe, x_el, t, dt, u_el, u_el_old,
                state_old_b, state_new_b, props_el, e,
            )

            # Assemble into global sparse matrix using FEC's pattern
            FEC._assemble_element!(storage, K_el, conn, e, bsi, bels)
        end
    end
end
