#!/usr/bin/env julia
# Normalize YAML files to Norma-style minimal quoting.
#
# Strips unnecessary "..." from keys and values:
#   - keys are always unquoted (YAML allows bare keys with spaces, apostrophes)
#   - values are unquoted UNLESS:
#       * the key is `function` (expression strings must stay strings)
#       * unquoting would change the parse (number, bool, null)
#
# Usage:
#   julia --project=. tools/normalize_yaml.jl path/to/file.yaml [more.yaml ...]
#   julia --project=. tools/normalize_yaml.jl --all          # all examples/**/*.yaml

# A value is safe to unquote if YAML would still parse it as the same string.
function _parses_as_string(s::AbstractString)::Bool
    tryparse(Float64, s) === nothing || return false
    tryparse(Int,     s) === nothing || return false
    s in ("true", "false", "True", "False", "TRUE", "FALSE")          && return false
    s in ("null", "Null", "NULL", "~", "")                            && return false
    # Conservative: refuse anything that starts with a YAML indicator.
    occursin(r"^[\-?:,\[\]{}#&*!|>'\"%@`]", s) && return false
    return true
end

# Match: indent (group 1) + optional list dash (group 2) + key chunk (group 3,
# possibly quoted) + colon + optional whitespace + optional value chunk
# (group 4 = quoted form like `"..."`, group 5 = bare form).
const _LINE_RE = r"""
    ^(\s*)                              # 1: indent
    (-\s+)?                             # 2: optional list dash
    (\"[^\"]+\"|[^:\s][^:]*?)           # 3: key (quoted or bare)
    \s*:                                # colon
    (?:                                 # optional value
        \s+
        (?:(\"[^\"]*\")|(.+?))          # 4: quoted value | 5: bare value
    )?
    \s*$
"""x

# Returns the unquoted form if the value can be safely unquoted under `key`,
# else the original value text.
function _maybe_unquote_value(key::AbstractString, qvalue::AbstractString)::String
    inner = qvalue[2:end-1]                # strip the surrounding ""
    key == "function" && return qvalue     # expressions must remain strings
    _parses_as_string(inner) || return qvalue
    return inner
end

function normalize_line(line::AbstractString)::String
    m = match(_LINE_RE, line)
    m === nothing && return line

    indent, dash, key_chunk, qvalue, bvalue = m.captures
    dash = dash === nothing ? "" : dash

    key = startswith(key_chunk, '"') && endswith(key_chunk, '"') ?
          key_chunk[2:end-1] : key_chunk

    if qvalue !== nothing
        new_value = _maybe_unquote_value(key, qvalue)
        return string(indent, dash, key, ": ", new_value)
    elseif bvalue !== nothing
        return string(indent, dash, key, ": ", bvalue)
    else
        return string(indent, dash, key, ":")
    end
end

function normalize_file(path::AbstractString)
    src = read(path, String)
    out = IOBuffer()
    for line in eachline(IOBuffer(src))
        println(out, normalize_line(line))
    end
    new_src = String(take!(out))
    if new_src != src
        write(path, new_src)
        println("normalized ", path)
    else
        println("unchanged  ", path)
    end
end

function normalize_all()
    files = String[]
    for (root, _, fs) in walkdir("examples")
        for f in fs
            endswith(f, ".yaml") && push!(files, joinpath(root, f))
        end
    end
    sort!(files)
    for f in files
        normalize_file(f)
    end
    println("\nProcessed ", length(files), " files.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) == 1 && ARGS[1] == "--all"
        normalize_all()
    elseif !isempty(ARGS)
        for f in ARGS
            normalize_file(f)
        end
    else
        println("usage: julia --project=. tools/normalize_yaml.jl --all")
        println("       julia --project=. tools/normalize_yaml.jl path/to/file.yaml ...")
        exit(1)
    end
end
