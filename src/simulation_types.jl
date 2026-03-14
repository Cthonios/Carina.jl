# Simulation container types for YAML-driven single-domain simulations.

import FiniteElementContainers as FEC

# ---------------------------------------------------------------------------
# TimeController
# ---------------------------------------------------------------------------

"""
    TimeController

Drives the coarse (output) time grid. Stops are uniformly spaced:
    t[k] = initial_time + k * control_step,   k = 0, 1, ..., num_stops-1
The integrator subcycles between consecutive stops; output is written
at every `output_interval` stops.
"""
mutable struct TimeController
    initial_time  ::Float64
    final_time    ::Float64
    control_step  ::Float64   # Δt_c: spacing between stops
    time          ::Float64   # target time of current stop
    prev_time     ::Float64   # time at previous stop
    num_stops     ::Int       # total stops = round((t_f − t_0)/Δt_c) + 1
    stop          ::Int       # current stop index (0-based)
end

# ---------------------------------------------------------------------------
# OutputSpec — controls which fields are written beyond displacement
# ---------------------------------------------------------------------------

"""
    OutputSpec

Controls which optional fields are written to the Exodus output file.
Parsed from the `output:` section of the input YAML.
"""
struct OutputSpec
    velocity            ::Bool
    acceleration        ::Bool
    stress              ::Bool
    deformation_gradient::Bool
    internal_variables  ::Bool
end

OutputSpec() = OutputSpec(false, false, false, false, false)

# ---------------------------------------------------------------------------
# Single-domain simulation
# ---------------------------------------------------------------------------

"""
    SingleDomainSimulation

Holds all objects needed to run one domain.
"""
struct SingleDomainSimulation{Params, ParamsCPU, Asm, Integrator, PP}
    params          ::Params       # device params (GPU or CPU)
    params_cpu      ::ParamsCPU    # always-CPU params (coords, physics, props)
    asm_cpu         ::Asm          # always-CPU assembler (DOF manager, ref_fes)
    integrator      ::Integrator
    post_processor  ::PP
    controller      ::TimeController
    output_interval ::Int
    device          ::Symbol       # :cpu, :rocm, or :cuda
    output_spec     ::OutputSpec
end
