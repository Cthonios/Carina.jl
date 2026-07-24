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

const _MODEL_NAMES = (
    "neohookean", "neo-hookean", "neo hookean",
    "linear elastic", "linearelastic",
    "hencky",
    "linear elasto plasticity",
    "saint venant kirchhoff", "saintvenant-kirchhoff", "saintvenantkirchhoff", "svk",
    "seth-hill", "seth hill", "sethhill",
    "j2 plasticity", "finitedefj2plasticity", "finite def j2 plasticity",
)

function _model_ctor(name::String)
    key = lowercase(strip(name))
    (key == "neohookean" || key == "neo-hookean" || key == "neo hookean")           && return CM.Hyperelastic(CM.NeoHookean())
    (key == "linear elastic" || key == "linearelastic")                              && return CM.LinearElastic()
    key == "hencky"                                                                  && return CM.Hyperelastic(CM.Hencky())
    key == "linear elasto plasticity"                                                && return CM.LinearElastoplastic(
                                                                                          CM.VonMises(CM.LinearIsotropicHardening())),
    (key == "saint venant kirchhoff" || key == "saintvenant-kirchhoff" ||
     key == "saintvenantkirchhoff"   || key == "svk")                                && return CM.Hyperelastic(CM.SaintVenantKirchhoff())
    (key == "seth-hill" || key == "seth hill" || key == "sethhill")                  && return CM.Hyperelastic(CM.SethHill())
    (key == "j2 plasticity" || key == "finitedefj2plasticity" ||
     key == "finite def j2 plasticity")                                              && return CM.FiniteDefJ2Plasticity()
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
        cm_key = get(_ELASTIC_KEY_ALIASES, lowercase(yaml_key), nothing)
        cm_key === nothing && _carina_log(0, :warning, "Unknown material property key \"$yaml_key\"; ignoring.")
        cm_key !== nothing && (props_inputs[cm_key] = _f64(val))
    end

    # quick fix to pack density into props_inputs for new CM interface
    props_inputs["density"] = density

    return cm, density, props_inputs
end
