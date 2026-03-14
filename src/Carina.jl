module Carina

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

import FiniteElementContainers as FEC
import ConstitutiveModels as CM
using LinearAlgebra
using StaticArrays
using Tensors
import Krylov
import LinearOperators
import SparseArrays
import YAML

# ---------------------------------------------------------------------------
# Submodules / source files
# ---------------------------------------------------------------------------

include("logging.jl")
include("simulation_types.jl")
include("physics.jl")
include("solvers.jl")
include("integrators.jl")
include("materials.jl")
include("io.jl")
include("simulation.jl")

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

# Physics
export SolidMechanics
export create_solid_mechanics_properties

# Integrators
export QuasiStaticIntegrator
export CentralDifferenceIntegrator
export NewmarkIntegrator

# Simulation
export SingleDomainSimulation
export TimeController
export create_simulation
export evolve!

# Entry point
export run

# Re-export frequently-used FEC and CM aliases so callers need only `using Carina`
export FEC
export CM

end # module Carina

# Allow direct invocation: julia --project=. src/Carina.jl input.yaml
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) != 1
        println(stderr, "Usage: julia --project=. src/Carina.jl <input.yaml>")
        exit(1)
    end
    Carina.run(ARGS[1])
end
