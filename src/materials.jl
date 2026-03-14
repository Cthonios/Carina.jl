# Material parsing: maps YAML strings to ConstitutiveModels.jl instances
# and assembles the property dict that CM's initialize_props expects.

import ConstitutiveModels as CM

# ---------------------------------------------------------------------------
# Model name → CM constructor
# ---------------------------------------------------------------------------

const _MODEL_CONSTRUCTORS = Dict{String, Function}(
    "neohookean"              => () -> CM.NeoHookean(),
    "neo-hookean"             => () -> CM.NeoHookean(),
    "neo hookean"             => () -> CM.NeoHookean(),
    "linear elastic"          => () -> CM.LinearElastic(),
    "linearelastic"           => () -> CM.LinearElastic(),
    "hencky"                  => () -> CM.Hencky(),
    "linear elasto plasticity"=> () -> CM.LinearElastoPlasticity(
                                    CM.VonMisesYieldSurface(),
                                    CM.LinearIsotropicHardening()),
    "saint venant kirchhoff"  => () -> CM.SaintVenantKirchhoff(),
    "saintvenant-kirchhoff"   => () -> CM.SaintVenantKirchhoff(),
    "saintvenantkirchhoff"    => () -> CM.SaintVenantKirchhoff(),
    "svk"                     => () -> CM.SaintVenantKirchhoff(),
    "seth-hill"               => () -> CM.SethHill(),
    "seth hill"               => () -> CM.SethHill(),
    "sethhill"                => () -> CM.SethHill(),
    "j2 plasticity"           => () -> CM.FiniteDefJ2Plasticity(),
    "finitedefj2plasticity"   => () -> CM.FiniteDefJ2Plasticity(),
    "finite def j2 plasticity"=> () -> CM.FiniteDefJ2Plasticity(),
)

# ---------------------------------------------------------------------------
# Elastic-constant key aliases  (YAML key → CM key)
# ---------------------------------------------------------------------------

const _ELASTIC_KEY_ALIASES = Dict{String, String}(
    "elastic modulus"    => "Young's modulus",
    "young's modulus"    => "Young's modulus",
    "youngs modulus"     => "Young's modulus",
    "poisson's ratio"    => "Poisson's ratio",
    "poissons ratio"     => "Poisson's ratio",
    "bulk modulus"       => "bulk modulus",
    "shear modulus"      => "shear modulus",
    "lame's first constant"  => "Lamé's first constant",
    "lames first constant"   => "Lamé's first constant",
    "lamé's first constant"  => "Lamé's first constant",
    # Plasticity
    "yield stress"        => "yield stress",
    "hardening modulus"   => "hardening modulus",
    # Seth-Hill exponents
    "m"                   => "m",
    "n"                   => "n",
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
    key = lowercase(strip(model_name))
    ctor = get(_MODEL_CONSTRUCTORS, key, nothing)
    ctor === nothing && error(
        "Unknown material model \"$model_name\". " *
        "Supported: $(join(keys(_MODEL_CONSTRUCTORS), ", "))"
    )
    cm = ctor()

    density = Float64(get(model_dict, "density", 0.0))
    density == 0.0 && @warn "No density specified for material \"$model_name\"; using 0.0."

    # Build CM-compatible property dict (canonicalize key names)
    props_inputs = Dict{String, Any}()
    for (yaml_key, val) in model_dict
        yaml_key == "density" && continue
        cm_key = get(_ELASTIC_KEY_ALIASES, lowercase(yaml_key), nothing)
        cm_key === nothing && @warn "Unknown material property key \"$yaml_key\"; ignoring."
        cm_key !== nothing && (props_inputs[cm_key] = Float64(val))
    end

    return cm, density, props_inputs
end
