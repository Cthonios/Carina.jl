# Carina.jl

**Carina.jl** is a finite element framework for **coupling and multiphysics
simulations**, primarily in **solid mechanics and heat conduction**.  As the
spiritual successor to [Norma.jl](https://github.com/sandialabs/Norma.jl),
Carina introduces **GPU acceleration** via
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl),
targeting CPUs, NVIDIA GPUs, and AMD GPUs from a single code base.

---

## Simulations

### Torsion wave — explicit central difference on AMD GPU

A free 1 m × 50 mm × 50 mm neo-Hookean elastic bar is given an initial
angular velocity field that launches torsion waves from every cross-section
simultaneously.  The waves superimpose, and the bar cycles through two
complete twist-and-return sequences before the animation loops.

| Displacement magnitude | Velocity magnitude |
|:---:|:---:|
| ![Displacement magnitude](docs/torsion_disp.gif) | ![Velocity magnitude](docs/torsion_velo.gif) |

*Neo-Hookean solid (E = 1 GPa, ν = 0.25, ρ = 1000 kg/m³).
160,000 hexahedral elements (~177,000 nodes).
Explicit central difference, Δt = 500 ns, ~14,000 steps, 5.95 ms simulated.
Wall time: 227 s on an AMD Radeon RX 7600 (ROCm).
Left: displacement magnitude [0, 71 mm].
Right: velocity magnitude [0, 141 m/s].
Rainbow Uniform colormap; geometry warped by actual displacements.*

---

### Sphere torsion — HHT-α implicit dynamics on AMD GPU

A free unit neo-Hookean sphere is given an initial angular velocity on its
surface that twists the top and bottom hemispheres in opposite directions,
launching torsional waves into the interior.  HHT-α algorithmic damping
suppresses spurious high-frequency modes while preserving the dominant
torsional dynamics.  The sphere undergoes progressively larger torsional
deformation as wave energy concentrates in the fundamental torsional mode.

| Displacement magnitude | Velocity magnitude |
|:---:|:---:|
| ![Displacement magnitude](docs/sphere_disp.gif) | ![Velocity magnitude](docs/sphere_velo.gif) |

*Neo-Hookean solid (E = 10 kPa, ν = 0.33, ρ = 1000 kg/m³).
864 hexahedral elements (997 nodes).
HHT-α implicit Newmark (α = −0.1, β = 0.3025, γ = 0.60) with L-BFGS nonlinear solver,
Δt = 50 ms, 800 steps, 40 s simulated.
Wall time: 207 s on an AMD Radeon RX 7600 (ROCm).
Left: displacement magnitude [0, 0.62 m].
Right: velocity magnitude [0, 0.50 m/s].
Rainbow Uniform colormap; geometry warped by actual displacements.*

---

## Features

- **GPU-accelerated** kernels via `KernelAbstractions.jl` — runs on CPU, NVIDIA, and AMD GPUs
- **Finite element framework** powered by [ReferenceFiniteElements.jl](https://github.com/Cthonios/ReferenceFiniteElements.jl) and [FiniteElementContainers.jl](https://github.com/Cthonios/FiniteElementContainers.jl)
- **Material models** via [ConstitutiveModels.jl](https://github.com/Cthonios/ConstitutiveModels.jl) (neo-Hookean, linear elastic, …)
- **Time integrators**: quasi-static Newton, implicit Newmark-β / HHT-α, explicit central difference, L-BFGS
- **Exodus I/O**: mesh input and field output compatible with ParaView
- **YAML-driven** problem setup — no recompilation needed
- Designed for extensibility toward **coupling and multiphysics**

---

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/Cthonios/Carina.jl")
```

Or clone and activate locally:

```bash
git clone https://github.com/Cthonios/Carina.jl
cd Carina.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

---

## Running

```bash
# Command-line
julia --project=. src/Carina.jl input.yaml

# Self-activating CLI wrapper
bin/carina input.yaml

# GPU (ROCm)
julia --project=. src/Carina.jl input.yaml   # with device: rocm in YAML
```

Select the device in the YAML input:

```yaml
device: cpu     # default
device: cuda    # NVIDIA
device: rocm    # AMD
```

---

## Testing

```bash
julia --project=. test/runtests.jl           # full suite
julia --project=. test/runtests.jl --quick   # fast subset
julia --project=. test/runtests.jl --filter torsion
julia --project=. test/runtests.jl --list    # show all tests
julia --project=. test/runtests.jl 1 3 5     # by index
```
