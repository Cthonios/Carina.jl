# Exodus output helpers for Carina simulations.
#
# Responsibilities:
#   _parse_output_spec        — parse YAML output: section → OutputSpec
#   _is_dynamic_integrator    — true for Newmark / central difference
#   _build_nodal_vars         — VectorFunction list for PostProcessor
#   _element_var_names        — element variable names to register upfront
#   _full_dof_to_h1field    — expand n_unknown vector to full H1Field
#   write_output!             — write all fields for one output stop
#   _write_element_fields!    — per-block stress / F / IV loop
# ---------------------------------------------------------------------------
# OutputSpec parser
# ---------------------------------------------------------------------------

function _parse_output_spec(dict::Dict{String,Any})
    out = get(dict, "output", Dict{String,Any}())
    !isempty(out) && _validate_keys(out, _OUTPUT_KEYS, "output")
    rec_str = lowercase(get(out, "recovery", "lumped"))
    recovery = if rec_str == "lumped"
        :lumped
    elseif rec_str in ("consistent", "l2")
        :consistent
    elseif rec_str == "none"
        :none
    else
        :none
    end
    return OutputSpec(
        Bool(get(out, "velocity",              false)),
        Bool(get(out, "acceleration",          false)),
        Bool(get(out, "stress",                true)),
        Bool(get(out, "deformation gradient",  true)),
        Bool(get(out, "internal variables",    false)),
        recovery,
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
        for ext in ("(XX)", "(XY)", "(XZ)", "(YY)", "(YZ)", "(ZZ)")
            for q in 1:nq_max
                push!(names, "sigma_$(ext)_$q")
            end
        end
    end

    if output_spec.deformation_gradient
        for ext in ("(XX)", "(YX)", "(ZX)", "(XY)", "(YY)", "(ZY)", "(XZ)", "(YZ)", "(ZZ)")
            for q in 1:nq_max
                push!(names, "F_$(ext)_$q")
            end
        end
    end

    if output_spec.internal_variables && NS > 0
        iv_names = CM.state_variable_names(physics.constitutive_model)
        for name in iv_names
            for q in 1:nq_max
                push!(names, "$(name)_$q")
            end
        end
    end

    return names
end

# ---------------------------------------------------------------------------
# Recovered (nodal) variable names for L2 projection
# ---------------------------------------------------------------------------

function _recovered_nodal_var_names(physics::SolidMechanics, output_spec::OutputSpec)
    output_spec.recovery == :none && return String[]
    names = String[]
    if output_spec.stress
        for ext in ("(XX)", "(XY)", "(XZ)", "(YY)", "(YZ)", "(ZZ)")
            push!(names, "nodal_sigma_$(ext)")
        end
    end
    if output_spec.deformation_gradient
        for ext in ("(XX)", "(YX)", "(ZX)", "(XY)", "(YY)", "(ZY)", "(XZ)", "(YZ)", "(ZZ)")
            push!(names, "nodal_F_$(ext)")
        end
    end
    return names
end

# ---------------------------------------------------------------------------
# L2 projection recovery: project QP stress to nodes
# ---------------------------------------------------------------------------

"""
    _write_recovered_fields!(sim, step)

Assemble RHS of L2 projection b_i = Σ_e Σ_q N_i(ξ_q) σ_qp w_q |J|,
then compute σ_nodal = M_lumped⁻¹ · b and write as nodal variables.
"""
function _write_recovered_fields!(sim, step)
    (; params_cpu, asm_cpu, post_processor, output_spec, recovery_data) = sim
    recovery_data isa NoRecovery && return

    if !output_spec.stress
        return
    end

    fspace = FEC.function_space(asm_cpu.dof)
    n_nodes = size(params_cpu.coords, 2)

    # Assemble RHS for each stress component: b_i = Σ_e Σ_q N_i σ_comp w |J|
    # We recompute stress at QPs (same as _write_element_fields!) and weight by N_i * JxW
    stress_nodal = zeros(Float64, 6, n_nodes)  # 6 symmetric components

    conns = fspace.elem_conns
    t  = FEC.current_time(params_cpu.times)
    dt = FEC.time_step(params_cpu.times)

    for (b, (block_physics, ref_fe, props)) in enumerate(zip(
        values(params_cpu.physics), values(fspace.ref_fes), values(params_cpu.properties),
    ))
        nelem   = conns.nelems[b]
        coffset = conns.offsets[b]
        state_old_b = FEC.block_view(params_cpu.state_old, b)
        state_new_b = FEC.block_view(params_cpu.state_new, b)

        for e in 1:nelem
            conn = FEC.connectivity(ref_fe, conns.data, e, coffset)
            x_el = FEC._element_level_fields_flat(params_cpu.coords, ref_fe, conn)
            u_el = FEC._element_level_fields_flat(params_cpu.field, ref_fe, conn)
            u_el_old = FEC._element_level_fields_flat(params_cpu.field_old, ref_fe, conn)
            props_el = FEC._element_level_properties(props, e)

            for q in 1:RFE.num_cell_quadrature_points(ref_fe)
                interps = FEC._cell_interpolants(ref_fe, q)
                state_old_q = FEC._quadrature_level_state(state_old_b, q, e)
                state_new_q = FEC._quadrature_level_state(state_new_b, q, e)

                # Compute stress at this QP (same as quadrature_field_output)
                cell = FEC.map_interpolants(interps, x_el)
                ∇u_q = FEC.interpolate_field_gradients(block_physics, cell, u_el)
                ∇u_q = FEC.modify_field_gradients(FEC.ThreeDimensional(), ∇u_q)
                state_scratch = zero(state_new_q)
                σ_q = symmetric(CM.cauchy_stress(
                    block_physics.constitutive_model, props_el, dt, ∇u_q, 0.0, state_old_q, state_scratch,
                ))

                # Shape function values at this QP
                N = RFE.cell_shape_function_value(ref_fe, q)
                JxW = cell.JxW

                # Accumulate b_i += N_i * σ_comp * JxW for each node in element
                nnpe = RFE.num_cell_dofs(ref_fe)
                for i in 1:nnpe
                    node = conn[i]
                    NiJxW = N[i] * JxW
                    for c in 1:6
                        stress_nodal[c, node] += NiJxW * σ_q.data[c]
                    end
                end
            end
        end
    end

    # Apply lumped mass inverse: σ_nodal = M_lumped⁻¹ · b
    if recovery_data isa LumpedRecovery
        for node in 1:n_nodes
            inv_m = recovery_data.inv_m_lumped[node]
            for c in 1:6
                stress_nodal[c, node] *= inv_m
            end
        end
    end

    # Write as nodal variables
    exo = post_processor.field_output_db
    comp_names = ("(XX)", "(XY)", "(XZ)", "(YY)", "(YZ)", "(ZZ)")
    for (c, ext) in enumerate(comp_names)
        Exodus.write_values(exo, Exodus.NodalVariable, step,
                            "nodal_sigma_$(ext)", stress_nodal[c, :])
    end
end

# ---------------------------------------------------------------------------
# Velocity / acceleration field expansion
# ---------------------------------------------------------------------------

# Create an H1Field from a full-DOF CPU vector (e.g. integrator.V or .A).
function _full_dof_to_h1field(unk_cpu::AbstractVector{Float64}, dof)
    field = FEC.create_field(dof)
    # Scatter reduced (unknown-DOF) vector into full field at unknown positions.
    for (i, fd) in enumerate(dof.unknown_dofs)
        field.data[fd] = unk_cpu[i]
    end
    return field
end

# ---------------------------------------------------------------------------
# Per-integrator velocity / acceleration accessors
# ---------------------------------------------------------------------------

_has_velocity(::_DynamicIntegrator) = true
_has_velocity(::Any)               = false

_has_acceleration(::_DynamicIntegrator) = true
_has_acceleration(::Any)               = false

# Bring velocity/acceleration vector to CPU (no-op if already CPU).
function _get_velocity_cpu(ig::_DynamicIntegrator, device::Symbol)
    device == :cpu && return Vector{Float64}(ig.V)
    return Vector{Float64}(Adapt.adapt(Array, ig.V))
end

function _get_acceleration_cpu(ig::_DynamicIntegrator, device::Symbol)
    device == :cpu && return Vector{Float64}(ig.A)
    return Vector{Float64}(Adapt.adapt(Array, ig.A))
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
    h1_cpu = Adapt.adapt(Array, params.field)

    FEC.write_times(post_processor, step, t)
    FEC.write_field(post_processor, step,
                    ("displ_x", "displ_y", "displ_z"), h1_cpu)

    # --- velocity ---
    if output_spec.velocity && _has_velocity(integrator)
        V_cpu = _get_velocity_cpu(integrator, device)
        v_h1  = _full_dof_to_h1field(V_cpu, asm_cpu.dof)
        FEC.write_field(post_processor, step,
                        ("velo_x", "velo_y", "velo_z"), v_h1)
    end

    # --- acceleration ---
    if output_spec.acceleration && _has_acceleration(integrator)
        A_cpu = _get_acceleration_cpu(integrator, device)
        a_h1  = _full_dof_to_h1field(A_cpu, asm_cpu.dof)
        FEC.write_field(post_processor, step,
                        ("acce_x", "acce_y", "acce_z"), a_h1)
    end

    # --- element-level fields (CPU only, recompute from current state) ---
    if output_spec.stress || output_spec.deformation_gradient ||
       output_spec.internal_variables

        # For GPU runs, bring field and state_new back to CPU.
        # h1_cpu already computed above.
        state_old_cpu = Adapt.adapt(Array, params.state_old)
        state_new_cpu = Adapt.adapt(Array, params.state_new)

        _write_element_fields!(post_processor, params_cpu, h1_cpu,
                               state_old_cpu, state_new_cpu, asm_cpu, output_spec, step)
    end

    # --- recovered nodal fields (L2 projection, CPU only) ---
    _write_recovered_fields!(sim, step)

    return nothing
end

# ---------------------------------------------------------------------------
# Element-level field writer (stress, F, internal variables)
# ---------------------------------------------------------------------------

# Per-quadrature-point output: stress and deformation gradient.
# Always computed at output time from the converged state — no storage needed.
struct QuadratureFieldOutput
    deformation_gradient::Tensor{2, 3, Float64, 9}
    stress::SymmetricTensor{2, 3, Float64, 6}
end

function quadrature_field_output(
    physics::SolidMechanics,
    interps, x_el,
    t, dt,
    u_el, u_el_old,
    state_old_q, state_new_q,
    props_el,
)
    cell = FEC.map_interpolants(interps, x_el)
    ∇u_q = FEC.interpolate_field_gradients(physics, cell, u_el)
    ∇u_q = FEC.modify_field_gradients(FEC.ThreeDimensional(), ∇u_q)

    F_q = ∇u_q + one(∇u_q)

    # Compute Cauchy stress from the converged state (state_old after promotion).
    # Use a scratch buffer for state_new to prevent mutation.
    state_scratch = similar(state_new_q)
    σ_q = symmetric(CM.cauchy_stress(
        physics.constitutive_model, props_el, dt, ∇u_q, 0.0, state_old_q, state_scratch,
    ))

    return QuadratureFieldOutput(F_q, σ_q)
end

function _write_element_fields!(pp, p_cpu, field_cpu, state_old_cpu, state_new_cpu,
                                asm_cpu, output_spec::OutputSpec, step::Int)
    fspace = FEC.function_space(asm_cpu.dof)

    # Assemble per-QP stress and F via FEC's quadrature assembly
    q_outputs = StructArray{QuadratureFieldOutput}[]
    for (b, ref_fe) in enumerate(fspace.ref_fes)
        nquad = RFE.num_cell_quadrature_points(ref_fe)
        nelem = FEC.num_elements(fspace, b)
        push!(q_outputs, StructArray{QuadratureFieldOutput}(undef, nquad, nelem))
    end
    q_outputs = NamedTuple{keys(fspace.ref_fes)}(q_outputs)

    # Extract unknown DOFs from the full field for the assembly call
    # (non-condensed mode expects n_free-sized unknown vector).
    Uu = field_cpu.data[asm_cpu.dof.unknown_dofs]
    FEC.assemble_quadrature_quantity!(
        q_outputs, nothing, asm_cpu.dof,
        quadrature_field_output,
        Uu, p_cpu
    )

    # Write fields
    for (b, (block_name, vals)) in enumerate(zip(keys(fspace.ref_fes), values(q_outputs)))
        block_str = String(block_name)

        if output_spec.stress
            FEC.write_field(pp, step, block_str, "sigma", vals.stress)
        end

        if output_spec.deformation_gradient
            FEC.write_field(pp, step, block_str, "F", vals.deformation_gradient)
        end

        if output_spec.internal_variables
            state_new_b = FEC.block_view(state_new_cpu, b)
            NS = size(state_new_b, 1)   # (NS, nquad, nelem)
            nquad = size(state_new_b, 2)
            block_physics = p_cpu.physics[block_name]
            iv_names = CM.state_variable_names(block_physics.constitutive_model)
            for (iv, name) in enumerate(iv_names)
                for q in 1:nquad
                    Exodus.write_values(
                        pp.field_output_db, Exodus.ElementVariable,
                        step, block_str, "$(name)_$q",
                        Vector{Float64}(state_new_b[iv, q, :])
                    )
                end
            end
        end
    end
end
