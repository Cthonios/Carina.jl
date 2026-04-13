# Structured console output for Carina.jl.
#
# Format mirrors Norma.jl's norma_log API:
#   [KEYWORD]  message           (level 0, flush left)
#       [KEYWORD]  message       (level 4, one indent = time step)
#           [KEYWORD]  message   (level 8, two indents = Newton iteration)
#
# Colors mirror Norma.jl's NORMA_COLORS so the two tools share a palette.
# Carina-specific additions:
#   :carina  — magenta (analogous to Norma's :norma)
#   :device  — light_blue (makes GPU vs CPU immediately visible at startup)

import Printf

@inline _use_color() =
    get(ENV, "CARINA_NO_COLOR", "false") != "true" &&
    (get(ENV, "FORCE_COLOR", "") != "" ||
     (isdefined(Base, :have_color) && Base.have_color === true) ||
     stdout isa Base.TTY)

const _COLORS = Dict{Symbol, Symbol}(
    :acceleration => :blue,
    :advance      => :green,
    :carina       => :magenta,
    :device       => :light_blue,
    :done         => :green,
    :equilibrium  => :blue,
    :linesearch   => :cyan,
    :output       => :cyan,
    :recover      => :yellow,
    :setup        => :magenta,
    :solve        => :cyan,
    :stop         => :blue,
    :time         => :light_cyan,
    :warning      => :yellow,
)

const CARINA_WRITE_LOG_FILE = Ref(true)
const CARINA_LOG_FILE = Ref{Union{IOStream,Nothing}}(nothing)

function open_log_file(input_file::AbstractString)
    CARINA_WRITE_LOG_FILE[] || return nothing
    CARINA_LOG_FILE[] === nothing || return nothing  # outermost run() owns the file
    path = first(splitext(input_file)) * ".log"
    CARINA_LOG_FILE[] = open(path, "w")
    return nothing
end

function close_log_file()
    io = CARINA_LOG_FILE[]
    io === nothing && return nothing
    close(io)
    CARINA_LOG_FILE[] = nothing
    return nothing
end

function _carina_log(level::Int, keyword::Symbol, msg::AbstractString)
    indent    = " "^level
    kw_str    = uppercase(string(keyword))
    kw_str    = kw_str[1:min(end, 7)]
    bracketed = "[" * kw_str * "]"
    padded    = rpad(bracketed, 9)
    prefix    = indent * padded * " "
    if _use_color()
        color = get(_COLORS, keyword, :default)
        printstyled(prefix; color=color, bold=true)
    else
        print(prefix)
    end
    println(msg)

    io = CARINA_LOG_FILE[]
    if io !== nothing
        println(io, prefix, msg)
        flush(io)
    end
end

function _carina_logf(level::Int, keyword::Symbol, fmt::AbstractString, args...)
    _carina_log(level, keyword, Printf.format(Printf.Format(fmt), args...))
end

"""
Format seconds as a human-readable wall time string.
  < 60s:     "12.34s"
  < 3600s:   "2m 34.56s"
  < 86400s:  "1h 23m 45.67s"
  ≥ 86400s:  "1d 2h 3m 4.56s"
"""
function format_time(seconds::Float64)
    if seconds < 60.0
        return Printf.@sprintf("%.2fs", seconds)
    end
    d = floor(Int, seconds / 86400);  seconds -= d * 86400
    h = floor(Int, seconds / 3600);   seconds -= h * 3600
    m = floor(Int, seconds / 60);     seconds -= m * 60
    s = Printf.@sprintf("%.2fs", seconds)
    parts = String[]
    d > 0 && push!(parts, "$(d)d")
    h > 0 && push!(parts, "$(h)h")
    m > 0 && push!(parts, "$(m)m")
    push!(parts, s)
    return join(parts, " ")
end

function _status_str(converged::Bool)
    if _use_color()
        converged ? "\e[32m[DONE]\e[39m" : "\e[33m[WAIT]\e[39m"
    else
        converged ? "[DONE]" : "[WAIT]"
    end
end

function _cg_status_str(converged::Bool)
    if _use_color()
        converged ? "\e[32m[CONV]\e[39m" : "\e[31m[STALL]\e[39m"
    else
        converged ? "[CONV]" : "[STALL]"
    end
end
