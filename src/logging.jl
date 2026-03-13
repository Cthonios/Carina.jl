# Structured console output for Carina.jl.
#
# Format mirrors Norma.jl's norma_log API:
#   [KEYWORD]  message           (level 0, flush left)
#       [KEYWORD]  message       (level 4, one indent = time step)
#           [KEYWORD]  message   (level 8, two indents = Newton iteration)
#
# Color notes vs Norma.jl:
#   :carina  — magenta (matches :norma in Norma)
#   :device  — light_blue (new: makes GPU vs CPU immediately visible)
#   :advance — green (same as Norma)
#   :solve   — cyan (same as Norma)
#   :stop    — blue (same as Norma)
#   :output  — cyan (same as Norma)
#   :done    — light_green (brighter than Norma's :green for clear end-of-run signal)
#   :time    — light_cyan (same as Norma)
#   :warn    — yellow (same as Norma's :warning; shortened keyword)

import Printf

@inline _use_color() =
    get(ENV, "CARINA_NO_COLOR", "false") != "true" &&
    (get(ENV, "FORCE_COLOR", "") != "" ||
     (isdefined(Base, :have_color) && Base.have_color === true) ||
     stdout isa Base.TTY)

const _COLORS = Dict{Symbol, Symbol}(
    :carina  => :magenta,
    :setup   => :magenta,
    :device  => :light_blue,
    :advance => :green,
    :solve   => :cyan,
    :stop    => :blue,
    :output  => :cyan,
    :done    => :light_green,
    :time    => :light_cyan,
    :warn    => :yellow,
)

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
end

function _carina_logf(level::Int, keyword::Symbol, fmt::AbstractString, args...)
    _carina_log(level, keyword, Printf.format(Printf.Format(fmt), args...))
end

function _status_str(converged::Bool)
    if _use_color()
        converged ? "\e[32m[DONE]\e[39m" : "\e[33m[WAIT]\e[39m"
    else
        converged ? "[DONE]" : "[WAIT]"
    end
end
