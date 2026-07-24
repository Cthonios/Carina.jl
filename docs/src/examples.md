# Examples

Runnable inputs live under `examples/`, grouped by time integrator. Each
directory contains a YAML input and its mesh, so any of them can be run
directly:

```bash
bin/carina examples/mechanics/quasistatic/cube/cube.yaml
```

Paths inside an input file resolve relative to that file, so examples work from
any working directory.

## Quasi-static

| Example | Demonstrates |
|---|---|
| `quasistatic/cube/cube.yaml` | Baseline Newton + direct solve |
| `quasistatic/cube/cube_gpu.yaml` | Same problem on GPU (CG + Jacobi) |
| `quasistatic/cube/cube_lbfgs.yaml` | Matrix-free L-BFGS linear solve |
| `quasistatic/cube/cube_nlcg.yaml` | Nonlinear CG, no linear solver |
| `quasistatic/cube/cube_sd.yaml` | Steepest descent |
| `quasistatic/cube-linear-elastic/cube.yaml` | Linear elastic material and its caching path |
| `quasistatic/cube-neumann-bc/cube.yaml` | Surface traction |
| `quasistatic/cube-gravity/cube.yaml` | Body force |
| `quasistatic/two-block/two-block.yaml` | Two element blocks with different materials |
| `quasistatic/tension-specimen/tension-specimen.yaml` | Tension specimen, elastic |
| `quasistatic/tension-specimen-j2/tension-specimen.yaml` | Finite-deformation J2 plasticity |
| `quasistatic/tension-specimen-j2/tension-specimen-dt001.yaml` | The same, with a smaller step for path accuracy |

## Implicit dynamics (Newmark)

| Example | Demonstrates |
|---|---|
| `implicit-dynamic/cube/cube.yaml` | Baseline Newmark |
| `implicit-dynamic/cube/cube_direct.yaml` | Direct linear solve |
| `implicit-dynamic/cube/cube_gpu.yaml` | GPU |
| `implicit-dynamic/cube/cube_lbfgs.yaml` | L-BFGS |
| `implicit-dynamic/cube/cube_nlcg.yaml` | Nonlinear CG |
| `implicit-dynamic/cube-rigid-body/cube.yaml` | Unconstrained rigid-body motion |
| `implicit-dynamic/cantilever/cantilever.yaml` | Cantilever vibration |
| `implicit-dynamic/clamped/clamped.yaml` | Clamped wave, verified against theory |
| `implicit-dynamic/clamped-bc/clamped-bc.yaml` | Wave driven by a boundary pulse |
| `implicit-dynamic/sphere/sphere_implicit.yaml` | Large-deformation sphere torsion |
| `implicit-dynamic/sphere/sphere_lbfgs.yaml` | The same with L-BFGS |
| `implicit-dynamic/torsion/torsion_lbfgs.yaml` | Torsion bar |
| `implicit-dynamic/tension-specimen/tension-specimen.yaml` | Dynamic tension |

## Explicit dynamics (central difference)

| Example | Demonstrates |
|---|---|
| `explicit-dynamic/cube/cube.yaml` | Baseline explicit run |
| `explicit-dynamic/cube/cube_gpu.yaml` | GPU explicit — the matrix-free path |
| `explicit-dynamic/cantilever/cantilever.yaml` | Cantilever |
| `explicit-dynamic/clamped/clamped.yaml` | Clamped wave with an initial pulse |
| `explicit-dynamic/clamped-bc/clamped-bc.yaml` | Boundary-driven pulse |
| `explicit-dynamic/sphere/sphere_explicit.yaml` | Sphere |
| `explicit-dynamic/torsion/torsion_explicit.yaml` | Torsion bar — the front-page animation |

## Suggested reading order

If you are new to the input format, work through these four:

1. **`quasistatic/cube/cube.yaml`** — the smallest complete input: mesh,
   material, integrator, one Dirichlet condition, one solver.
2. **`quasistatic/cube-neumann-bc/cube.yaml`** — adds loading, and shows the
   traction sign convention.
3. **`implicit-dynamic/cube/cube.yaml`** — adds mass, initial conditions, and
   the termination-criteria block.
4. **`explicit-dynamic/torsion/torsion_explicit.yaml`** — the explicit path:
   CFL-driven step, output-interval subcycling, and no solver section at all.

## Running on GPU

Examples ending in `_gpu.yaml` set `device:` in the input. Any example can be
moved to a GPU from the command line instead:

```bash
bin/carina examples/mechanics/explicit-dynamic/torsion/torsion_explicit.yaml --device rocm
```

The explicit examples benefit most — that path is matrix-free and scales to
millions of degrees of freedom. Note that `direct` linear solves and the `ic`
and `amg` preconditioners are CPU-only, so implicit examples using them will
fail on GPU; see [Solvers](reference/solvers.md).
