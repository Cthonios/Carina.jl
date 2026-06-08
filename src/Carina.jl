module Carina

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
import Adapt
import ConstitutiveModels as CM
import Exodus
import FiniteElementContainers as FEC
import KernelAbstractions as KA
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
include("status_tests.jl")
include("simulation_types.jl")
include("physics.jl")
include("solvers.jl")
include("integrators.jl")
include("linear_solvers.jl")
include("nonlinear_solvers.jl")
include("materials.jl")
include("input_parsing.jl")
include("initialization.jl")
include("io.jl")
include("simulation.jl")

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

# Status tests
export SolverStatus, Unconverged, Converged, Failed
export SolverInfo, AbstractStatusTest, check, reset!
export AbsResidualTest, RelResidualTest, AbsUpdateTest, RelUpdateTest
export MaxIterationsTest, MinIterationsTest, FiniteValueTest, DivergenceTest, StagnationTest
export ModelFlagTest, ComboAndTest, ComboOrTest
export default_nonlinear_status_test, default_linear_status_test

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

# Re-export frequently-used FEC and CM aliases so callers need only `using Carina`
export FEC
export CM

end # module Carina
