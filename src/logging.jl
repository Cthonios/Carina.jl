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
    get(ENV, "NO_COLOR", "") == "" &&
    get(ENV, "CI", "") == "" &&
    (get(ENV, "FORCE_COLOR", "") != "" || stdout isa Base.TTY)

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

# Trim-safe support: precompute indents, 9-char labels, and ANSI escape codes
# so the hot path avoids `repeat`/`rpad`/`printstyled` (none of which survive
# `juliac --trim`).

const _INDENTS = ("", "    ", "        ", "            ", "                ")
@inline function _indent(level::Int)
    @inbounds for k in 0:length(_INDENTS) - 1
        4k == level && return _INDENTS[k + 1]
    end
    return ""
end

# 9-char-wide bracketed labels: bracketed name padded to a 9-char column.
const _LABELS = (
    (:acceleration, "[ACCELER]"),
    (:advance,      "[ADVANCE]"),
    (:carina,       "[CARINA] "),
    (:device,       "[DEVICE] "),
    (:done,         "[DONE]   "),
    (:equilibrium,  "[EQUILIB]"),
    (:function,     "[FUNCTIO]"),
    (:linesearch,   "[LINESEA]"),
    (:output,       "[OUTPUT] "),
    (:recover,      "[RECOVER]"),
    (:setup,        "[SETUP]  "),
    (:size,         "[SIZE]   "),
    (:solve,        "[SOLVE]  "),
    (:stop,         "[STOP]   "),
    (:time,         "[TIME]   "),
    (:warning,      "[WARNING]"),
)
@inline function _label(keyword::Symbol)
    @inbounds for (k, v) in _LABELS
        k === keyword && return v
    end
    return "[" * uppercase(string(keyword)) * "]"
end

# ANSI escape codes mirroring the entries in _COLORS.
const _ANSI = (
    (:blue,       "\e[34m"),
    (:green,      "\e[32m"),
    (:magenta,    "\e[35m"),
    (:cyan,       "\e[36m"),
    (:yellow,     "\e[33m"),
    (:red,        "\e[31m"),
    (:light_blue, "\e[94m"),
    (:light_cyan, "\e[96m"),
    (:default,    "\e[39m"),
)
@inline function _ansi(color::Symbol)
    @inbounds for (k, v) in _ANSI
        k === color && return v
    end
    return "\e[39m"
end

const _ANSI_BOLD  = "\e[1m"
const _ANSI_RESET = "\e[22;39m"

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
    prefix = _indent(level) * _label(keyword) * " "
    if _use_color()
        color = get(_COLORS, keyword, :default)
        print(Core.stdout, _ANSI_BOLD, _ansi(color), prefix, _ANSI_RESET)
    else
        print(Core.stdout, prefix)
    end
    println(Core.stdout, msg)

    io = CARINA_LOG_FILE[]
    if io !== nothing
        println(io, prefix, replace(msg, r"\e\[[0-9;]*m" => ""))
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

# ---------------------------------------------------------------------------
# Phase progress + timing instrumentation
# ---------------------------------------------------------------------------
# Setup proceeds through several FEC calls that produce no console output, so a
# slow phase looks like a hang.  `@carina_phase` announces each phase before
# running it; `@carina_timed` is a quieter variant for fine-grained sub-phases.
# Both also log wall time afterwards when CARINA_TIMING is set.

_carina_timing_on() = get(ENV, "CARINA_TIMING", "") in ("1", "true", "yes", "on")

"""
    @carina_phase "Doing the thing" expr

Announce a setup phase (`[SETUP] Doing the thing...`) before evaluating `expr`,
so a long silent phase never looks like a hang.  When `CARINA_TIMING` is set,
also log the phase wall time afterwards.  Returns the value of `expr`.
"""
macro carina_phase(label, expr)
    quote
        _carina_log(0, :setup, string($(esc(label)), "..."))
        if _carina_timing_on()
            local _t0 = time()
            local _result = $(esc(expr))
            _carina_logf(0, :time, "  %-26s %s",
                         $(esc(label)), format_time(time() - _t0))
            _result
        else
            $(esc(expr))
        end
    end
end

"""
    @carina_timed "label" expr

Evaluate `expr` and return its value.  When `CARINA_TIMING` is set, log the
wall time under `label`; otherwise a zero-overhead pass-through that prints
nothing — used for fine-grained sub-phase diagnostics.
"""
macro carina_timed(label, expr)
    quote
        if _carina_timing_on()
            local _t0 = time()
            local _result = $(esc(expr))
            _carina_logf(0, :time, "  %-26s %s",
                         $(esc(label)), format_time(time() - _t0))
            _result
        else
            $(esc(expr))
        end
    end
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
