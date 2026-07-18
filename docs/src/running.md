# Running Carina

## The CLI wrapper

```bash
bin/carina input.yaml
```

`bin/carina` is a self-activating shell wrapper. On first use it instantiates
its own launcher environment (`bin/Project.toml`) — a one-time step that prints:

```
carina: first run — instantiating launcher environment (one-time)...
```

```
Usage: carina <input.yaml> [--device cpu|cuda|rocm|auto] [--threads N]
```

| Flag | Values | Description |
|---|---|---|
| `--device` | `cpu`, `cuda`, `rocm`, `auto` | Compute backend. |
| `--threads`, `-t` | integer | Julia threads; passed through as `julia -t N`. |

### Why the launcher is a separate environment

The launcher environment — not the Carina library — owns the GPU vendor
packages (CUDA.jl, AMDGPU.jl). It selects a KernelAbstractions backend and
hands it to `Carina.run`. This is what keeps the library itself free of any
vendor dependency, so Carina can be used on a machine with neither CUDA nor
ROCm installed.

## Selecting a device

Three ways, in order of precedence:

1. The `--device` command-line flag
2. The `device:` key in the YAML input
3. Default: `cpu`

```bash
bin/carina input.yaml --device rocm
```

```yaml
device: rocm    # cpu (default), cuda, rocm, or auto
```

| Value | Backend |
|---|---|
| `cpu` | `KernelAbstractions.CPU()` |
| `cuda` | NVIDIA via CUDA.jl |
| `rocm` | AMD via AMDGPU.jl |
| `auto` | Best available, preferring ROCm, then CUDA, then CPU |

`cuda` and `rocm` are strict — if the requested device is not functional the
run aborts rather than silently falling back:

```
--device rocm: no functional AMD GPU found.
```

Use `auto` when you want a fallback. An unrecognised value is an error:

```
Unknown device "gpu". Expected cpu, cuda, rocm, or auto.
```

## CPU threading

```bash
bin/carina input.yaml --threads 12
```

Threading comes from FiniteElementContainers, which partitions element loops
across `Threads.nthreads()` tasks. What that actually covers:

**Multithreaded** — the internal-force assembly, which is the dominant
per-step cost, including the per-element constitutive evaluation that runs
inside the element loop. At setup, the mass assembly and the stable-time-step
reduction.

**Serial** — the nodal vector updates in the predictor, corrector, and
acceleration update; the free-to-full DOF scatter; norms and state copies.
These are `O(n_dof)` streaming operations on a single core.

Since the assembly dominates memory traffic, `--threads` speeds up the bulk of
the work, but the serial vector bookkeeping is a real Amdahl tail. Do not
expect linear scaling to high core counts.

## Running directly with Julia

```bash
julia --project=. src/Carina.jl input.yaml
```

This bypasses the launcher environment and therefore has no access to the GPU
vendor packages — it is CPU-only. Use `bin/carina` for GPU runs.

Interactively:

```julia
using Pkg; Pkg.activate(".")
using Carina
Carina.run("input.yaml")
```

`Carina.run` also accepts an explicit backend, which is how the launcher calls
it:

```julia
import KernelAbstractions as KA
Carina.run("input.yaml"; backend = KA.CPU())
```

## Reading the log

A typical startup:

```
[SETUP]   Input:  /path/to/cube.g
[SETUP]   Output: /path/to/cube.e
[SETUP]   Mesh:    177023 nodes, 160000 elements
[SETUP]   DOFs:    530523 total, 530523 free, 0 constrained
[SETUP]   Stable Δt = 2.05e-06 (CFL = 0.90)
[SETUP]   Setup complete (20.82s)
```

Then one line per output stop:

```
[STOP]    [1/20, 5%] : Time = 1.72e-04 : |U|_max = 1.91e-02 : wall = 9.69s
```

The `wall` figure is the time for that interval, including the Exodus write. On
short intervals the I/O can dominate — when benchmarking, use a large `output
interval` so the write cost is amortized.

Prefixes worth recognising:

| Prefix | Meaning |
|---|---|
| `[SETUP]` | Problem construction |
| `[STOP]` | An output frame was written |
| `[SOLVE]` | Nonlinear iteration detail |
| `[LINESEA]` | Line-search backtracking |
| `[WARNING]` | Non-fatal; the run continues with a default |
| `[TIME]` | Wall-clock totals |

Warnings are easy to scroll past and are the first thing to check when results
look wrong — see [Troubleshooting](troubleshooting.md).
