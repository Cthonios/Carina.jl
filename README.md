[![CI](https://github.com/Cthonios/Carina.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/Cthonios/Carina.jl/actions/workflows/ci.yml)
[![docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://cthonios.github.io/Carina.jl/dev/)
[![License: BSD 3-Clause](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](LICENSE)

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
| ![Displacement magnitude](docs/src/assets/torsion_disp.gif) | ![Velocity magnitude](docs/src/assets/torsion_velo.gif) |

*Neo-Hookean solid (E = 1 GPa, ν = 0.25, ρ = 1000 kg/m³).
160,000 hexahedral elements (~177,000 nodes).
Explicit central difference, Δt = 500 ns, ~14,000 steps, 5.95 ms simulated.
Wall time: 227 s on an AMD Radeon RX 7600 (ROCm).*

### Sphere torsion — implicit Newmark on CPU

A free unit neo-Hookean sphere is given an initial angular velocity field
that twists the top and bottom hemispheres in opposite directions,
launching torsional waves into the interior.

| Displacement magnitude | Velocity magnitude |
|:---:|:---:|
| ![Displacement magnitude](docs/src/assets/sphere_disp.gif) | ![Velocity magnitude](docs/src/assets/sphere_velo.gif) |

*Neo-Hookean solid (E = 10 kPa, ν = 0.33, ρ = 1000 kg/m³).
864 hexahedral elements (997 nodes).
Implicit Newmark-β (β = 0.49, γ = 0.9) with CG + Jacobi preconditioner,
Δt = 10 ms, 400 steps, 4.0 s simulated.*

### Elastic wave propagation — clamped beam

A 1 m linear elastic beam clamped at both ends is given a Gaussian
displacement pulse at the midpoint.  The pulse splits into two
counter-propagating waves that reflect off the boundaries and reform the
mirror image of the initial condition at t = L/c = 1 ms.  The computed
solution (red) is overlaid on the closed-form analytical solution (black).

![Clamped wave](docs/src/assets/clamped_wave.gif)

*Linear elastic (E = 1 GPa, ν = 0, ρ = 1000 kg/m³); wave speed c = 1000 m/s.
1000 hexahedral elements (4004 nodes).
Explicit central difference, Δt = 100 ns, 10,000 steps, 1 ms simulated.
Analytical solution: Mota, Tezaur & Phlipot, IJNME 123:5036–5071, 2022, eq. 28.*

---

## Features

- **GPU-accelerated** kernels via `KernelAbstractions.jl` — CPU, NVIDIA, and AMD from one code base
- **Finite element framework** powered by [ReferenceFiniteElements.jl](https://github.com/Cthonios/ReferenceFiniteElements.jl) and [FiniteElementContainers.jl](https://github.com/Cthonios/FiniteElementContainers.jl)
- **Material models** via [ConstitutiveModels.jl](https://github.com/Cthonios/ConstitutiveModels.jl) — neo-Hookean, linear elastic, Hencky, Saint Venant–Kirchhoff, Seth–Hill, and J2 plasticity
- **Time integrators** — quasi-static Newton, implicit Newmark-β / HHT-α, explicit central difference
- **Solvers** — Newton, nonlinear CG, steepest descent; direct, CG, and L-BFGS linear solves
- **Preconditioners** — Jacobi, Chebyshev, incomplete Cholesky, and smoothed-aggregation AMG
- **Exodus I/O** — mesh input and field output compatible with ParaView
- **YAML-driven** problem setup — no recompilation needed

See [Features](docs/src/features.md) for the full list, including what is
planned rather than implemented.

---

## Installation

Carina tracks the `main` branches of three sibling packages via relative paths,
so all four repositories must be cloned side by side:

```bash
git clone git@github.com:Cthonios/ConstitutiveModels.jl.git
git clone git@github.com:Cthonios/FiniteElementContainers.jl.git
git clone git@github.com:Cthonios/ReferenceFiniteElements.jl.git
git clone git@github.com:Cthonios/Carina.jl.git
cd Carina.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Full details, including GPU setup, in [Installation](docs/src/installation.md).

---

## Running

```bash
# Self-activating CLI wrapper (recommended)
bin/carina input.yaml

# With GPU
bin/carina input.yaml --device rocm
bin/carina input.yaml --device cuda

# Multi-threaded (CPU)
bin/carina input.yaml --threads 8

# Or directly with julia (CPU only)
julia --project=. src/Carina.jl input.yaml
```

The device may also be set in the YAML input:

```yaml
device: cpu     # default; also cuda, rocm, or auto
```

See [Running Carina](docs/src/running.md).

---

## Testing

```bash
julia --project=. test/runtests.jl           # full suite
julia --project=. test/runtests.jl --quick   # fast subset
julia --project=. test/runtests.jl --filter torsion
julia --project=. test/runtests.jl --list    # show all tests
julia --project=. test/runtests.jl 1 3 5     # by index
```

---

## Documentation

Full documentation lives in [`docs/`](docs/) and is built with
[Documenter.jl](https://documenter.juliadocs.org/):

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate(); include("docs/make.jl")'
# then open docs/build/index.html
```

- [Installation](docs/src/installation.md)
- [Running Carina](docs/src/running.md)
- [Features](docs/src/features.md)
- [Testing](docs/src/testing.md)
- [Examples](docs/src/examples.md)
- [Troubleshooting](docs/src/troubleshooting.md)
- **[Input File Reference](docs/src/reference/index.md)** — every section and key of the YAML input file

---

## License

BSD 3-Clause.  See [LICENSE](LICENSE).
