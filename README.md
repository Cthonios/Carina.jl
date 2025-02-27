# Carina.jl

**Carina.jl** is a finite element framework designed for **coupling and multiphysics simulations**, primarily in **solid mechanics and heat conduction**. As the **spiritual successor to Norma.jl**, Carina.jl introduces **GPU acceleration** for **HPC environments**. It leverages **KernelAbstractions.jl** to support execution on **CPUs, NVIDIA GPUs, and AMD GPUs**, while maintaining a flexible and extensible architecture.

## Features
- **GPU-Accelerated Computation** via `KernelAbstractions.jl`
- **Finite Element Framework** powered by [ReferenceFiniteElements.jl](https://github.com/Cthonios/ReferenceFiniteElements.jl)
- **Efficient Data Management** using [FiniteElementContainers.jl](https://github.com/Cthonios/FiniteElementContainers.jl)
- **Material Models & Constitutive Laws** via [ConstitutiveModels.jl](https://github.com/Cthonios/ConstitutiveModels.jl)
- **Coupling & Multiphysics** support for solid mechanics and heat conduction
- **Designed for Extensibility & Experimentation**
- **Scalable Parallel Execution** for HPC applications

## Installation
To install `Carina.jl`, run the following command in Julia:

```julia
using Pkg
Pkg.add(url="https://github.com/Cthonios/Carina.jl")
