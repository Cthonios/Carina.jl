#!/usr/bin/env julia
# One-shot TOML → YAML converter for Carina example inputs.
#
# Reverses the YAML → TOML migration: snake_case keys become space-
# separated, the few keys that carry apostrophes / accents in the YAML
# schema (Young's, Poisson's, Lamé's) are restored, and model-name
# selectors stored under `type:` or inside `blocks:` get the same
# snake → space transform applied to their string values.
#
# Usage:
#   julia --project=. tools/toml_to_yaml.jl examples/path/to/input.toml
#       writes examples/path/to/input.yaml
#   julia --project=. tools/toml_to_yaml.jl --all
#       converts every examples/**/*.toml in place, deleting the toml.

using TOML
using YAML

# Keys that need apostrophe / accent restoration, not just space substitution.
# Snake-case form on the left, YAML-canonical form on the right.
const _KEY_RESTORATIONS = Dict(
    "poissons_ratio"        => "Poisson's ratio",
    "youngs_modulus"        => "Young's modulus",
    "lames_first_constant"  => "Lamé's first constant",
    "elastic_modulus"       => "elastic modulus",
)

unsnake(s::AbstractString) = replace(string(s), "_" => " ")

function rename_key(k::AbstractString)
    haskey(_KEY_RESTORATIONS, k) && return _KEY_RESTORATIONS[k]
    return unsnake(k)
end

# `enclosing_key` is the key of the dict/list that contains the value we're
# processing — used to detect when a string value is a selector (e.g. all
# strings inside a `blocks:` table are material-name selectors).
function transform(d::AbstractDict, enclosing_key::AbstractString = "")
    out = Dict{String, Any}()
    for (k, v) in d
        new_k = rename_key(string(k))
        out[new_k] = transform_value(v, new_k, enclosing_key)
    end
    return out
end

function transform(v::AbstractVector, enclosing_key::AbstractString = "")
    return Any[transform_value(x, "", enclosing_key) for x in v]
end

function transform_value(v::AbstractDict, local_key::AbstractString, enclosing_key::AbstractString)
    return transform(v, local_key)
end

function transform_value(v::AbstractVector, local_key::AbstractString, enclosing_key::AbstractString)
    return transform(v, local_key)
end

function transform_value(v::AbstractString, local_key::AbstractString, enclosing_key::AbstractString)
    if local_key == "type" || enclosing_key == "blocks"
        return unsnake(v)
    end
    return v
end

transform_value(v, local_key::AbstractString, enclosing_key::AbstractString) = v

function convert_file(toml_path::AbstractString; delete_toml::Bool = false)
    endswith(toml_path, ".toml") || error("not a .toml: $toml_path")
    yaml_path = replace(toml_path, r"\.toml$" => ".yaml")

    raw = TOML.parsefile(toml_path)
    transformed = transform(raw)

    YAML.write_file(yaml_path, transformed)

    println("wrote ", yaml_path)
    if delete_toml
        rm(toml_path)
        println("removed ", toml_path)
    end
    return yaml_path
end

function convert_all()
    files = String[]
    for (root, _, fs) in walkdir("examples")
        for f in fs
            endswith(f, ".toml") && push!(files, joinpath(root, f))
        end
    end
    sort!(files)
    for f in files
        convert_file(f; delete_toml = true)
    end
    println("\nConverted ", length(files), " files.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) == 1 && ARGS[1] == "--all"
        convert_all()
    elseif length(ARGS) == 1
        convert_file(ARGS[1])
    else
        println("usage: julia tools/toml_to_yaml.jl <file.toml> | --all")
        exit(1)
    end
end
