# Material parsing: maps input strings to ConstitutiveModels.jl instances
# and assembles the property dict that CM's initialize_props expects.

import ConstitutiveModels as CM

# ---------------------------------------------------------------------------
# Model name → CM constructor
# ---------------------------------------------------------------------------

# Explicit dispatch (no Dict-of-closures) — the abstract `Function` lookup
# isn't trim-resolvable.  Returns `nothing` for unknown names.
const _MODEL_NAMES = (
    "neohookean", "neo_hookean", "linear_elastic", "hencky",
    "linear_elasto_plasticity",
    "saint_venant_kirchhoff", "svk",
    "seth_hill",
    "j2_plasticity", "finite_def_j2_plasticity",
)

function _model_ctor(name::String)
    name == "neohookean"               && return CM.NeoHookean()
    name == "neo_hookean"              && return CM.NeoHookean()
    name == "linear_elastic"           && return CM.LinearElastic()
    name == "hencky"                   && return CM.Hencky()
    name == "linear_elasto_plasticity" && return CM.LinearElastoPlasticity(
                                              CM.VonMisesYieldSurface(),
                                              CM.LinearIsotropicHardening())
    name == "saint_venant_kirchhoff"   && return CM.SaintVenantKirchhoff()
    name == "svk"                      && return CM.SaintVenantKirchhoff()
    name == "seth_hill"                && return CM.SethHill()
    name == "j2_plasticity"            && return CM.FiniteDefJ2Plasticity()
    name == "finite_def_j2_plasticity" && return CM.FiniteDefJ2Plasticity()
    return nothing
end

# ---------------------------------------------------------------------------
# Elastic-constant key aliases (input key → CM key)
# CM's canonical keys retain apostrophes / accents; the LHS is the TOML form.
# ---------------------------------------------------------------------------

const _ELASTIC_KEY_ALIASES = Dict{String, String}(
    "elastic_modulus"      => "Young's modulus",
    "youngs_modulus"       => "Young's modulus",
    "poissons_ratio"       => "Poisson's ratio",
    "bulk_modulus"         => "bulk modulus",
    "shear_modulus"        => "shear modulus",
    "lames_first_constant" => "Lamé's first constant",
    # Plasticity
    "yield_stress"         => "yield stress",
    "hardening_modulus"    => "hardening modulus",
    # Seth-Hill exponents
    "m"                    => "m",
    "n"                    => "n",
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    parse_material(model_name, model_dict) -> (cm, density, props_inputs)

Parse a material block.  Returns the ConstitutiveModel instance, the
density (Float64), and a `Dict{String,Any}` of elastic-constant inputs
ready to be passed to `create_solid_mechanics_properties`.
"""
function parse_material(model_name::String, model_dict::Dict)
    cm = _model_ctor(model_name)
    cm === nothing && error(
        "Unknown material model \"$model_name\". " *
        "Supported: $(join(_MODEL_NAMES, ", "))"
    )

    density = _f64(get(model_dict, "density", 0.0))
    iszero(density) && _carina_log(0, :warning, "No density specified for material \"$model_name\"; using 0.0.")

    # Build CM-compatible property dict (canonicalize key names)
    props_inputs = Dict{String, Any}()
    for (input_key, val) in model_dict
        input_key == "density" && continue
        cm_key = get(_ELASTIC_KEY_ALIASES, input_key, nothing)
        cm_key === nothing && _carina_log(0, :warning, "Unknown material property key \"$input_key\"; ignoring.")
        cm_key !== nothing && (props_inputs[cm_key] = _f64(val))
    end

    return cm, density, props_inputs
end
