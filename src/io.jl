# Exodus output helpers for Carina simulations.
#
# Responsibilities:
#   _parse_output_spec        — parse YAML output: section → OutputSpec
#   _is_dynamic_integrator    — true for Newmark / central difference
#   _build_nodal_vars         — VectorFunction list for PostProcessor
#   _element_var_names        — element variable names to register upfront
#   _expand_unk_to_h1field    — expand n_unknown vector to full H1Field
#   write_output!             — write all fields for one output stop
#   _write_element_fields!    — per-block stress / F / IV loop

import FiniteElementContainers as FEC
import ConstitutiveModels as CM
import ReferenceFiniteElements as RFE
import Exodus
import Adapt
using Tensors
using StaticArrays

# ---------------------------------------------------------------------------
# OutputSpec parser
# ---------------------------------------------------------------------------

function _parse_output_spec(dict::Dict{String,Any})
    out = get(dict, "output", Dict{String,Any}())
    return OutputSpec(
        Bool(get(out, "velocity",              false)),
        Bool(get(out, "acceleration",          false)),
        Bool(get(out, "stress",                false)),
        Bool(get(out, "deformation gradient",  false)),
        Bool(get(out, "internal variables",    false)),
    )
end

# ---------------------------------------------------------------------------
# Helpers for integrator type
# ---------------------------------------------------------------------------

function _is_dynamic_integrator(dict::Dict{String,Any})
    ti  = get(dict, "time integrator", Dict{String,Any}())
    typ = lowercase(get(ti, "type", "quasi static"))
    return typ in ("central difference", "centraldifference", "cd",
                   "newmark", "newmark-beta", "newmark beta")
end

# ---------------------------------------------------------------------------
# VectorFunction list for PostProcessor
# ---------------------------------------------------------------------------

# Returns [displ_func, velo_func?, acce_func?] depending on OutputSpec.
function _build_nodal_vars(V, output_spec::OutputSpec, is_dynamic::Bool)
    vars = [FEC.VectorFunction(V, :displ)]
    if is_dynamic && output_spec.velocity
        push!(vars, FEC.VectorFunction(V, :velo))
    end
    if is_dynamic && output_spec.acceleration
        push!(vars, FEC.VectorFunction(V, :acce))
    end
    return vars
end

# ---------------------------------------------------------------------------
# Element variable names
# ---------------------------------------------------------------------------

# Compute all element variable names that must be declared before the first
# write.  Matches the naming convention used by FEC.write_field for
# SymmetricTensor / Tensor (ext_q ordering).
function _element_var_names(asm_cpu, physics::SolidMechanics,
                             output_spec::OutputSpec)
    any_el = output_spec.stress || output_spec.deformation_gradient ||
             output_spec.internal_variables
    any_el || return String[]

    fspace = FEC.function_space(asm_cpu.dof)
    nq_max = maximum(RFE.num_cell_quadrature_points(ref_fe)
                     for ref_fe in values(fspace.ref_fes))
    NS     = CM.num_state_variables(physics.constitutive_model)

    names = String[]

    if output_spec.stress
        for ext in ("xx", "yy", "zz", "yz", "xz", "xy")
            for q in 1:nq_max
                push!(names, "stress_$(ext)_$q")
            end
        end
    end

    if output_spec.deformation_gradient
        for ext in ("xx", "yy", "zz", "yz", "xz", "xy", "zy", "zx", "yx")
            for q in 1:nq_max
                push!(names, "F_$(ext)_$q")
            end
        end
    end

    if output_spec.internal_variables && NS > 0
        for iv in 1:NS
            for q in 1:nq_max
                push!(names, "iv_$(iv)_$q")
            end
        end
    end

    return names
end

# ---------------------------------------------------------------------------
# Velocity / acceleration field expansion
# ---------------------------------------------------------------------------

# Expand an n_unknown CPU vector (V or A from an integrator) to a full
# H1Field (3 × n_nodes), zeroing constrained DOFs.
function _expand_unk_to_h1field(unk_cpu::AbstractVector{Float64}, dof)
    field = FEC.create_field(dof)
    for (i, fd) in enumerate(dof.unknown_dofs)
        field.data[fd] = unk_cpu[i]
    end
    return field
end

# ---------------------------------------------------------------------------
# Per-integrator velocity / acceleration accessors
# ---------------------------------------------------------------------------

_has_velocity(::NewmarkIntegrator)            = true
_has_velocity(::CentralDifferenceIntegrator)  = true
_has_velocity(::QuasiStaticIntegrator)        = false
_has_velocity(::Any)                          = false

_has_acceleration(::NewmarkIntegrator)            = true
_has_acceleration(::CentralDifferenceIntegrator)  = true
_has_acceleration(::QuasiStaticIntegrator)        = false
_has_acceleration(::Any)                          = false

# Bring velocity vector to CPU (no-op if already CPU).
function _get_velocity_cpu(ig::NewmarkIntegrator, device::Symbol)
    device == :cpu && return Vector{Float64}(ig.V)
    return Vector{Float64}(Base.invokelatest(Adapt.adapt, Array, ig.V))
end
function _get_velocity_cpu(ig::CentralDifferenceIntegrator, device::Symbol)
    device == :cpu && return Vector{Float64}(ig.V)
    return Vector{Float64}(Base.invokelatest(Adapt.adapt, Array, ig.V))
end

function _get_acceleration_cpu(ig::NewmarkIntegrator, device::Symbol)
    device == :cpu && return Vector{Float64}(ig.A)
    return Vector{Float64}(Base.invokelatest(Adapt.adapt, Array, ig.A))
end
function _get_acceleration_cpu(ig::CentralDifferenceIntegrator, device::Symbol)
    device == :cpu && return Vector{Float64}(ig.A)
    return Vector{Float64}(Base.invokelatest(Adapt.adapt, Array, ig.A))
end

# ---------------------------------------------------------------------------
# Main output function
# ---------------------------------------------------------------------------

"""
    write_output!(sim::SingleDomainSimulation, step::Int)

Write all requested fields to the Exodus output database at time-step
index `step`.  Displacement is always written; velocity, acceleration,
stress, deformation gradient, and internal variables are written if
enabled in `sim.output_spec`.
"""
function write_output!(sim::SingleDomainSimulation, step::Int)
    (; params, params_cpu, asm_cpu, integrator,
       post_processor, controller, output_spec, device) = sim

    t = controller.time

    # --- displacement (always) ---
    h1_cpu = device != :cpu ?
        Base.invokelatest(Adapt.adapt, Array, params.h1_field) :
        Adapt.adapt(Array, params.h1_field)

    FEC.write_times(post_processor, step, t)
    FEC.write_field(post_processor, step,
                    ("displ_x", "displ_y", "displ_z"), h1_cpu)

    # --- velocity ---
    if output_spec.velocity && _has_velocity(integrator)
        V_cpu = _get_velocity_cpu(integrator, device)
        v_h1  = _expand_unk_to_h1field(V_cpu, asm_cpu.dof)
        FEC.write_field(post_processor, step,
                        ("velo_x", "velo_y", "velo_z"), v_h1)
    end

    # --- acceleration ---
    if output_spec.acceleration && _has_acceleration(integrator)
        A_cpu = _get_acceleration_cpu(integrator, device)
        a_h1  = _expand_unk_to_h1field(A_cpu, asm_cpu.dof)
        FEC.write_field(post_processor, step,
                        ("acce_x", "acce_y", "acce_z"), a_h1)
    end

    # --- element-level fields (CPU only, recompute from current state) ---
    if output_spec.stress || output_spec.deformation_gradient ||
       output_spec.internal_variables

        # For GPU runs, bring h1_field and state_new back to CPU.
        # h1_cpu already computed above.
        state_new_cpu = device != :cpu ?
            Base.invokelatest(Adapt.adapt, Array, params.state_new) :
            params.state_new

        _write_element_fields!(post_processor, params_cpu, h1_cpu,
                                state_new_cpu, asm_cpu, output_spec, step)
    end

    return nothing
end

# ---------------------------------------------------------------------------
# Element-level field writer (stress, F, internal variables)
# ---------------------------------------------------------------------------

function _write_element_fields!(pp, p_cpu, h1_field_cpu, state_new_cpu,
                                 asm_cpu, output_spec::OutputSpec, step::Int)
    fspace  = FEC.function_space(asm_cpu.dof)
    conns   = fspace.elem_conns
    I3      = one(Tensor{2,3,Float64})

    for (b, (block_name, ref_fe, block_physics, block_props)) in enumerate(zip(
            keys(fspace.ref_fes), values(fspace.ref_fes),
            values(p_cpu.physics), values(p_cpu.properties)
        ))

        nelem   = conns.nelems[b]
        coffset = conns.offsets[b]
        nquad   = RFE.num_cell_quadrature_points(ref_fe)
        NS      = CM.num_state_variables(block_physics.constitutive_model)

        # Allocate per-(quad,elem) output arrays.
        σ_mat   = output_spec.stress ?
            Matrix{SymmetricTensor{2,3,Float64,6}}(undef, nquad, nelem) : nothing
        F_mat   = output_spec.deformation_gradient ?
            Matrix{Tensor{2,3,Float64,9}}(undef, nquad, nelem) : nothing
        iv_data = (output_spec.internal_variables && NS > 0) ?
            zeros(Float64, NS, nquad, nelem) : nothing

        state_new_b = FEC.block_view(state_new_cpu, b)

        for e in 1:nelem
            conn     = FEC.connectivity(ref_fe, conns.data, e, coffset)
            x_el     = FEC._element_level_fields_flat(p_cpu.h1_coords,  ref_fe, conn)
            u_el     = FEC._element_level_fields_flat(h1_field_cpu,     ref_fe, conn)
            u_el_old = u_el   # use current displ for both (hyper-elastic post-proc)

            for q in 1:nquad
                interps     = ref_fe.cell_interps[q]
                # Use state_new for both old and new (converged end-of-step state).
                state_q     = view(state_new_b, :, q, e)

                cell  = FEC.map_interpolants(interps, x_el)
                ∇u_q  = FEC.interpolate_field_gradients(block_physics, cell, u_el)
                ∇u_q  = FEC.modify_field_gradients(FEC.ThreeDimensional(), ∇u_q)
                F_q   = I3 + ∇u_q

                if output_spec.stress
                    J_q = det(F_q)
                    P_q = CM.pk1_stress(block_physics.constitutive_model,
                                        block_props, 0.0, ∇u_q, 0.0,
                                        state_q, state_q)
                    σ_mat[q, e] = symmetric((1 / J_q) * P_q ⋅ transpose(F_q))
                end

                if output_spec.deformation_gradient
                    F_mat[q, e] = F_q
                end

                if output_spec.internal_variables && NS > 0
                    for iv in 1:NS
                        iv_data[iv, q, e] = state_q[iv]
                    end
                end
            end
        end

        block_str = String(block_name)

        if output_spec.stress
            FEC.write_field(pp, step, block_str, "stress", σ_mat)
        end

        if output_spec.deformation_gradient
            FEC.write_field(pp, step, block_str, "F", F_mat)
        end

        if output_spec.internal_variables && NS > 0
            for iv in 1:NS
                for q in 1:nquad
                    Exodus.write_values(
                        pp.field_output_db, Exodus.ElementVariable,
                        step, block_str, "iv_$(iv)_$q",
                        Vector{Float64}(iv_data[iv, q, :])
                    )
                end
            end
        end
    end

    return nothing
end
