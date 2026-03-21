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

# Create an H1Field from a full-DOF CPU vector (e.g. integrator.V or .A).
function _full_dof_to_h1field(full_cpu::AbstractVector{Float64}, dof)
    field = FEC.create_field(dof)
    copyto!(field.data, full_cpu)
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
    return Vector{Float64}(Base.invokelatest(Adapt.adapt, Array, ig.V))
end

function _get_acceleration_cpu(ig::_DynamicIntegrator, device::Symbol)
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
        state_old_cpu = device != :cpu ?
            Base.invokelatest(Adapt.adapt, Array, params.state_old) :
            params.state_old
        state_new_cpu = device != :cpu ?
            Base.invokelatest(Adapt.adapt, Array, params.state_new) :
            params.state_new

        _write_element_fields!(post_processor, params_cpu, h1_cpu,
                               state_old_cpu, state_new_cpu, asm_cpu, output_spec, step)
    end

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

    FEC.assemble_quadrature_quantity!(
        q_outputs, nothing, asm_cpu.dof,
        quadrature_field_output,
        field_cpu, p_cpu
    )

    # Write fields
    for (b, (block_name, vals)) in enumerate(zip(keys(fspace.ref_fes), values(q_outputs)))
        block_str = String(block_name)

        if output_spec.stress
            FEC.write_field(pp, step, block_str, "stress", vals.stress)
        end

        if output_spec.deformation_gradient
            FEC.write_field(pp, step, block_str, "F", vals.deformation_gradient)
        end

        if output_spec.internal_variables
            state_new_b = FEC.block_view(state_new_cpu, b)
            NS = size(state_new_b, 1)   # (NS, nquad, nelem)
            nquad = size(state_new_b, 2)
            nelem = size(state_new_b, 3)
            for iv in 1:NS
                for q in 1:nquad
                    Exodus.write_values(
                        pp.field_output_db, Exodus.ElementVariable,
                        step, block_str, "iv_$(iv)_$q",
                        Vector{Float64}(state_new_b[iv, q, :])
                    )
                end
            end
        end
    end
end
