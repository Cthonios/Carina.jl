# Carina: Copyright 2026 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software. This software
# is released under the BSD license detailed in the file LICENSE.md in the
# top-level Carina.jl directory.

# Simulation container types for YAML-driven single-domain simulations.

import FiniteElementContainers as FEC

# ---------------------------------------------------------------------------
# TimeController
# ---------------------------------------------------------------------------

"""
    TimeController

Drives the coarse (output) time grid. Stops are uniformly spaced,
    t[k] = initial_time + k * control_step,   k = 0, 1, ..., num_stops-1,
except that the last stop is clamped to `final_time` when `control_step`
(the `output interval`) does not divide the time span.  The integrator
subcycles between consecutive stops; output is written at every stop.
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
    recovery            ::Symbol   # :none, :lumped, :consistent
end

OutputSpec() = OutputSpec(false, false, false, false, false, :none)

# ---------------------------------------------------------------------------
# Recovery data for L2 projection of QP fields to nodes
# ---------------------------------------------------------------------------

abstract type AbstractRecoveryData end

struct NoRecovery <: AbstractRecoveryData end

# Nodal recovery of quadrature-point fields (stress, internal variables).
# The projection space is the continuous piecewise-linear space on the
# vertices of the elements: for elements whose nodes are all vertices (TETRA4,
# HEX8) that is the element's own basis; for elements with edge, face or
# interior nodes (TETRA10, TETRA15) the P1 space on the four vertices, with
# the other nodes interpolated from the vertices afterwards.  The full basis
# of those elements is not usable: its row-sum lumped mass is zero or
# negative at vertices, and its consistent mass integrated with the element's
# own quadrature rule (degree 5 for the fifteen-node tetrahedron, whose basis
# products reach degree 8) is singular on the interior bubble (found on the
# punch benchmark, 2026-10-15: recovered stresses of 1e12 Pa against element
# values of 1e7 Pa, negative recovered plastic strains, and a failed Cholesky
# factorization).  The P1 mass is integrated exactly by every rule of degree
# two or more, and its row sums are positive.
#   vidx   : node → index of its vertex unknown, 0 for a node that is not a vertex
#   interp : n_nodes × n_v, nodal values from the vertex values (identity rows
#            for vertices, the vertex shape functions at the node for the others)
struct LumpedRecovery <: AbstractRecoveryData
    inv_m_lumped::Vector{Float64}   # 1/M_lumped per vertex unknown (n_v)
    vidx::Vector{Int}
    interp::SparseArrays.SparseMatrixCSC{Float64, Int}
end

struct ConsistentRecovery <: AbstractRecoveryData
    M_factor::Any   # Cholesky factorization of the vertex P1 mass matrix (n_v × n_v)
    vidx::Vector{Int}
    interp::SparseArrays.SparseMatrixCSC{Float64, Int}
end

# ---------------------------------------------------------------------------
# Point loads (Neumann BCs on node sets — applied as direct forces)
# ---------------------------------------------------------------------------

struct PointLoad
    unk_idx ::Int           # index into the unknown DOF vector (0 = constrained, skip)
    node    ::Int           # node index (for coordinate lookup)
    func    ::FEC.Expressions.ScalarExpressionFunction{Float64}  # (coords, t) → scalar force value
end

# ---------------------------------------------------------------------------
# Single-domain simulation
# ---------------------------------------------------------------------------

"""
    SingleDomainSimulation

Holds all objects needed to run one domain.
"""
struct SingleDomainSimulation{Params, ParamsCPU, Asm, Integrator, PP, Bkd, RD <: AbstractRecoveryData}
    params          ::Params       # device params (GPU or CPU)
    params_cpu      ::ParamsCPU    # always-CPU params (coords, physics, props)
    asm_cpu         ::Asm          # always-CPU assembler (DOF manager, ref_fes)
    integrator      ::Integrator
    post_processor  ::PP
    controller      ::TimeController
    backend         ::Bkd          # KernelAbstractions backend (KA.CPU / CUDABackend / ROCBackend)
    output_spec     ::OutputSpec
    recovery_data   ::RD           # L2 projection data (NoRecovery, LumpedRecovery, or ConsistentRecovery)
end
