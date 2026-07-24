# Material parsing: maps YAML strings to ConstitutiveModels.jl instances
# and assembles the property dict that CM's initialize_props expects.

import ConstitutiveModels as CM

# ---------------------------------------------------------------------------
# Model name → CM constructor
# ---------------------------------------------------------------------------
#
# The user-facing name is matched after a `lowercase(strip(...))` normalisation
# so YAML can spell the model as "linear elastic" / "neo-hookean" / "j2
# plasticity" without each variant needing its own switch arm.

# Single source of truth: accepted spellings → constructor.  `_MODEL_NAMES` is
# DERIVED from this table, so the "Supported: ..." list in the parse error can
# never drift from what `_model_ctor` actually accepts.  The previous version
# kept the two by hand as parallel `&& return` arms; a stray trailing comma
# folded the Saint-Venant-Kirchhoff arm into the preceding `return` as a tuple
# element, which made every SVK spelling unreachable *and* made
# `linear elasto plasticity` return a `Tuple` instead of a model.  Neither
# failure was visible until runtime, and the error message went on advertising
# "svk" as supported while rejecting it.  A table cannot desynchronise that way.
const _MODEL_TABLE = (
    (("neohookean", "neo-hookean", "neo hookean"),
     () -> CM.Hyperelastic(CM.NeoHookean())),
    (("linear elastic", "linearelastic"),
     () -> CM.LinearElastic()),
    (("hencky",),
     () -> CM.Hyperelastic(CM.Hencky())),
    (("linear elasto plasticity",),
     () -> CM.LinearElastoplastic(CM.VonMises(CM.LinearIsotropicHardening()))),
    (("saint venant kirchhoff", "saintvenant-kirchhoff", "saintvenantkirchhoff", "svk"),
     () -> CM.Hyperelastic(CM.SaintVenantKirchhoff())),
    (("seth-hill", "seth hill", "sethhill"),
     () -> CM.Hyperelastic(CM.SethHill())),
    (("j2 plasticity", "finitedefj2plasticity", "finite def j2 plasticity"),
     () -> CM.FiniteDefJ2Plasticity()),
)

const _MODEL_NAMES = Tuple(Iterators.flatten(aliases for (aliases, _) in _MODEL_TABLE))

function _model_ctor(name::String)
    key = lowercase(strip(name))
    for (aliases, ctor) in _MODEL_TABLE
        key in aliases && return ctor()
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Elastic-constant key aliases (YAML key → CM key)
# CM's canonical keys retain apostrophes / accents; the LHS is the YAML form
# the user is allowed to write.  Matched case-insensitively via `lowercase`.
# ---------------------------------------------------------------------------

const _ELASTIC_KEY_ALIASES = Dict{String, String}(
    "elastic modulus"        => "Young's modulus",
    "young's modulus"        => "Young's modulus",
    "youngs modulus"         => "Young's modulus",
    "poisson's ratio"        => "Poisson's ratio",
    "poissons ratio"         => "Poisson's ratio",
    "bulk modulus"           => "bulk modulus",
    "shear modulus"          => "shear modulus",
    "lame's first constant"  => "Lamé's first constant",
    "lames first constant"   => "Lamé's first constant",
    "lamé's first constant"  => "Lamé's first constant",
    # Plasticity
    "yield stress"           => "yield stress",
    "hardening modulus"      => "hardening modulus",
    # Seth-Hill exponents
    "m"                      => "m",
    "n"                      => "n",
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    parse_material(model_name, model_dict) -> (cm, density, props_inputs)

Parse YAML material block.  Returns the ConstitutiveModel instance, the
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
    for (yaml_key, val) in model_dict
        yaml_key == "density" && continue
        # `model` selects the constitutive model when the block's material label
        # is an arbitrary name; it is not a material property.
        lowercase(strip(yaml_key)) == "model" && continue
        cm_key = get(_ELASTIC_KEY_ALIASES, lowercase(yaml_key), nothing)
        cm_key === nothing && _carina_log(0, :warning, "Unknown material property key \"$yaml_key\"; ignoring.")
        cm_key !== nothing && (props_inputs[cm_key] = _f64(val))
    end

    # Density is a material property as far as ConstitutiveModels is concerned:
    # every model's `initialize_props` reads `inputs["density"]` and emits it as
    # props[1].  It is still returned separately because the input-validation and
    # logging paths want it before any property vector exists.
    props_inputs["density"] = density

    return cm, density, props_inputs
end
