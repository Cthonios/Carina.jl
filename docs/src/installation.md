# Installation

Carina depends on three sibling packages that track their `main` branches
through relative paths declared in `Project.toml`:

- [ConstitutiveModels.jl](https://github.com/Cthonios/ConstitutiveModels.jl)
- [FiniteElementContainers.jl](https://github.com/Cthonios/FiniteElementContainers.jl)
- [ReferenceFiniteElements.jl](https://github.com/Cthonios/ReferenceFiniteElements.jl)

Because the paths are relative, all four repositories must be cloned as
**siblings in the same parent directory**:

```bash
git clone git@github.com:Cthonios/ConstitutiveModels.jl.git
git clone git@github.com:Cthonios/FiniteElementContainers.jl.git
git clone git@github.com:Cthonios/ReferenceFiniteElements.jl.git
git clone git@github.com:Cthonios/Carina.jl.git
cd Carina.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The resulting layout:

```
parent/
├── Carina.jl/
├── ConstitutiveModels.jl/
├── FiniteElementContainers.jl/
└── ReferenceFiniteElements.jl/
```

Changes to any sibling repository are picked up immediately after restarting
Julia — no reinstallation step.

## Keeping the siblings current

The sibling packages move independently, and a change to one can require a
matching change in Carina's `Project.toml` compat bounds. If resolution starts
failing after a pull, update all four and re-resolve:

```bash
for r in ConstitutiveModels.jl FiniteElementContainers.jl ReferenceFiniteElements.jl; do
  git -C ../$r pull --ff-only
done
julia --project=. -e 'using Pkg; Pkg.update(); Pkg.resolve()'
```

A resolution failure typically reads:

```
ERROR: Unsatisfiable requirements detected for package Exodus [f57ae99e]:
 ├─restricted to versions 0.14 by Carina
 └─restricted to versions 0.15 by FiniteElementContainers — no versions left
```

which means a sibling has advanced past Carina's compat bound for a shared
dependency. See [Troubleshooting](troubleshooting.md).

## GPU support

GPU execution is optional and requires the vendor package for your hardware:

| Device | Package | `device` value |
|---|---|---|
| AMD | [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) (ROCm) | `rocm` |
| NVIDIA | [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) | `cuda` |

Carina loads the backend through package extensions, so neither dependency is
required for CPU-only use. Install whichever applies:

```bash
julia --project=. -e 'using Pkg; Pkg.add("AMDGPU")'   # AMD
julia --project=. -e 'using Pkg; Pkg.add("CUDA")'     # NVIDIA
```

Verify the device is visible before running a large job:

```bash
julia --project=. -e 'using AMDGPU; AMDGPU.versioninfo()'
julia --project=. -e 'using CUDA; CUDA.versioninfo()'
```

## Building the documentation

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate(); include("docs/make.jl")'
```

Then open `docs/build/index.html`.
