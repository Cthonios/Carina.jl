#!/usr/bin/env julia
# Reorder keys in Carina YAML files to mirror Norma's canonical ordering:
#   - top level: type, mesh I/O, output interval, output, model,
#                time integrator, initial conditions, boundary conditions, solver
#   - any nested mapping with a `type:` key: hoist `type:` to the top
#
# Works on the raw text so existing quoting decisions (function: "0.0") are
# preserved exactly.  Assumes 2-space indentation and no inline comments or
# blank lines mid-mapping (true for all current Carina examples).
#
# Usage:
#   julia --project=. tools/reorder_yaml.jl path/to/file.yaml [more.yaml ...]
#   julia --project=. tools/reorder_yaml.jl --all

const _TOP_ORDER = [
    "type",
    "input mesh file",
    "output mesh file",
    "output interval",
    "output",
    "model",
    "time integrator",
    "initial conditions",
    "boundary conditions",
    "solver",
]

# Number of leading spaces on `line`; -1 if blank or comment.
function _indent_of(line::AbstractString)::Int
    s = lstrip(line)
    (isempty(s) || startswith(s, '#')) && return -1
    return length(line) - length(s)
end

# Extract the bare key text from a line like `  key: value`, `  "key": value`,
# `  - key: value`, or `  - "key": value`.  Returns "" if no `key:` found.
function _key_of(line::AbstractString)::String
    s = lstrip(line)
    startswith(s, "- ") && (s = lstrip(s[3:end]))
    colon = findfirst(':', s)
    colon === nothing && return ""
    k = strip(s[1:prevind(s, colon)])
    if startswith(k, '"') && endswith(k, '"') && length(k) >= 2
        k = k[2:prevind(k, lastindex(k))]
    end
    return String(k)
end

# Detect whether the body of a block at `indent` is a nested mapping (vs. a list
# or a scalar value on the key line).  We look at the first non-blank line after
# the key line.  A nested mapping is "lines deeper than `indent` that don't
# start with `- `".
function _body_is_mapping(body::Vector{<:AbstractString}, indent::Int)::Bool
    for line in body
        i = _indent_of(line)
        i < 0 && continue
        i <= indent && return false
        s = lstrip(line)
        return !startswith(s, "- ")
    end
    return false
end

# Split lines into blocks at `base_indent`.  Each block is the key line at
# `base_indent` plus every following line until the next key line at
# `base_indent`.  Lines at indent < base_indent end the parse.
function _split_blocks(lines::Vector{<:AbstractString}, base_indent::Int)
    blocks = Tuple{String, Vector{String}}[]
    n = length(lines)
    i = 1
    while i <= n
        ind = _indent_of(lines[i])
        if ind < 0
            i += 1
            continue
        end
        ind < base_indent && break          # left this scope
        ind > base_indent && error("unexpected deeper indent at line $i: $(lines[i])")
        # ind == base_indent: a key line at this level
        key = _key_of(lines[i])
        j = i + 1
        while j <= n
            ind2 = _indent_of(lines[j])
            if ind2 >= 0 && ind2 <= base_indent
                break
            end
            j += 1
        end
        push!(blocks, (key, String.(lines[i:j-1])))
        i = j
    end
    return blocks
end

function _stable_sort_by(blocks, order::Vector{String})
    rank(k) = begin
        idx = findfirst(==(k), order)
        idx === nothing ? length(order) + 1 : idx
    end
    return sort(blocks; by = b -> rank(b[1]), alg = Base.MergeSort)
end

function _hoist_type(blocks)
    ti = findfirst(b -> b[1] == "type", blocks)
    ti === nothing && return blocks
    return [blocks[ti]; blocks[1:ti-1]; blocks[ti+1:end]]
end

# Recursively reorder a block's body (the lines after its key line) by:
#   - splitting into sub-blocks at the next indent level
#   - hoisting `type:` first (if present)
#   - recursing into each sub-block's body
# Returns the new list of lines for the body.
function _reorder_body(body::Vector{String}, base_indent::Int)::Vector{String}
    _body_is_mapping(body, base_indent) || return body
    sub_indent = base_indent + 2
    subs = _split_blocks(body, sub_indent)
    subs = _hoist_type(subs)
    out = String[]
    for (_, sub_lines) in subs
        if length(sub_lines) > 1
            inner = _reorder_body(sub_lines[2:end], sub_indent)
            append!(out, [sub_lines[1]; inner])
        else
            append!(out, sub_lines)
        end
    end
    return out
end

function reorder_file(path::AbstractString)
    src   = read(path, String)
    lines = collect(eachline(IOBuffer(src)))

    # Top-level blocks at indent 0
    top = _split_blocks(lines, 0)
    top = _stable_sort_by(top, _TOP_ORDER)

    out = String[]
    for (_, body) in top
        # body[1] is the key line at indent 0; recurse into body[2:end] at indent 2
        if length(body) > 1
            inner = _reorder_body(body[2:end], 0)
            append!(out, [body[1]; inner])
        else
            append!(out, body)
        end
    end
    new_src = join(out, "\n") * "\n"
    if new_src != src
        write(path, new_src)
        println("reordered ", path)
    else
        println("unchanged ", path)
    end
end

function reorder_all()
    files = String[]
    for (root, _, fs) in walkdir("examples")
        for f in fs
            endswith(f, ".yaml") && push!(files, joinpath(root, f))
        end
    end
    sort!(files)
    for f in files
        reorder_file(f)
    end
    println("\nProcessed ", length(files), " files.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) == 1 && ARGS[1] == "--all"
        reorder_all()
    elseif !isempty(ARGS)
        for f in ARGS
            reorder_file(f)
        end
    else
        println("usage: julia --project=. tools/reorder_yaml.jl --all")
        println("       julia --project=. tools/reorder_yaml.jl path/to/file.yaml ...")
        exit(1)
    end
end
