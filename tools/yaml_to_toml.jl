#!/usr/bin/env julia
# One-shot YAML -> TOML converter for Carina example inputs.
#
# Walks the dict tree produced by YAML.load_file, lowercases every key,
# replaces spaces with underscores, and drops apostrophes.  String values
# get the same transform only when they are model-name selectors, namely:
#   - any value under a `type:` key (e.g. type: "solid mechanics" -> "solid_mechanics")
#   - any value inside a `blocks:` table (block name -> material name selector)
# Other string values (expressions, file paths) are left alone.
#
# Usage:
#   julia --project=. tools/yaml_to_toml.jl examples/path/to/input.yaml
#       writes examples/path/to/input.toml
#   julia --project=. tools/yaml_to_toml.jl --all
#       converts every examples/**/*.yaml in place, deleting the yaml.

using YAML
using TOML

const ALIAS_DROP = Dict(
    "γ" => "gamma",
    "β" => "beta",
    "α" => "alpha",
)

snake(s::AbstractString) = replace(lowercase(replace(s, "'" => "")), " " => "_")

function rename_key(k::AbstractString)
    haskey(ALIAS_DROP, k) && return ALIAS_DROP[k]
    return snake(k)
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
        return snake(v)
    end
    return v
end

transform_value(v, local_key::AbstractString, enclosing_key::AbstractString) = v

function convert_file(yaml_path::AbstractString; delete_yaml::Bool = false)
    endswith(yaml_path, ".yaml") || error("not a .yaml: $yaml_path")
    toml_path = replace(yaml_path, r"\.yaml$" => ".toml")

    raw = YAML.load_file(yaml_path; dicttype=Dict{String,Any})
    transformed = transform(raw)

    open(toml_path, "w") do io
        TOML.print(io, transformed; sorted = false)
    end

    println("wrote ", toml_path)
    if delete_yaml
        rm(yaml_path)
        println("removed ", yaml_path)
    end
    return toml_path
end

function convert_all()
    files = String[]
    for (root, _, fs) in walkdir("examples")
        for f in fs
            endswith(f, ".yaml") && push!(files, joinpath(root, f))
        end
    end
    sort!(files)
    for f in files
        convert_file(f; delete_yaml = true)
    end
    println("\nConverted ", length(files), " files.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) == 1 && ARGS[1] == "--all"
        convert_all()
    elseif length(ARGS) == 1
        convert_file(ARGS[1])
    else
        println("usage: julia tools/yaml_to_toml.jl <file.yaml> | --all")
        exit(1)
    end
end
