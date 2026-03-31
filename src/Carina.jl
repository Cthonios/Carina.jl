module Carina

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
import Adapt
import AMDGPU
import ConstitutiveModels as CM
import CUDA
import Exodus
import FiniteElementContainers as FEC
import Krylov
import LinearOperators
import ReferenceFiniteElements as RFE
import SparseArrays
import YAML
using LinearAlgebra
using StaticArrays
using StructArrays
using Tensors

# ---------------------------------------------------------------------------
# Submodules / source files
# ---------------------------------------------------------------------------

include("logging.jl")
include("simulation_types.jl")
include("physics.jl")
include("solvers.jl")
include("integrators.jl")
include("linear_solvers.jl")
include("nonlinear_solvers.jl")
include("materials.jl")
include("yaml_parsing.jl")
include("initialization.jl")
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
export QuasiStaticLBFGSIntegrator
export CentralDifferenceIntegrator
export NewmarkIntegrator

# Simulation
export SingleDomainSimulation
export TimeController
export create_simulation
export evolve!

# Entry point
export run
export best_device

# Re-export frequently-used FEC and CM aliases so callers need only `using Carina`
export FEC
export CM

end # module Carina

# Allow direct invocation: julia --project=. src/Carina.jl input.yaml [--device cpu|rocm|cuda]
if abspath(PROGRAM_FILE) == @__FILE__
    function _parse_cli(args)
        yaml_file = nothing
        device    = nothing
        i = 1
        while i <= length(args)
            if args[i] == "--device" && i < length(args)
                device = args[i + 1]; i += 2
            elseif yaml_file === nothing && !startswith(args[i], "-")
                yaml_file = args[i]; i += 1
            else
                println(stderr, "Usage: carina <input.yaml> [--device cpu|rocm|cuda]")
                exit(1)
            end
        end
        if yaml_file === nothing
            println(stderr, "Usage: carina <input.yaml> [--device cpu|rocm|cuda]")
            exit(1)
        end
        return yaml_file, device
    end
    yaml_file, device = _parse_cli(ARGS)
    Carina.run(yaml_file; device=device)
end
