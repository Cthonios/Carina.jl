# Troubleshooting

Most Carina input problems are *silent* rather than loud — a mistyped section is
skipped, not rejected. This page is ordered by how often that bites.

## The simulation runs but nothing is constrained

**Symptom.** The body drifts or rotates freely, a quasi-static solve fails
immediately on a singular system, or displacements are orders of magnitude too
large.

**Cause.** Most often a mesh-name mismatch: a `side set` or `node set` that
does not exist in the Exodus file constrains nothing.

Capitalisation of `dirichlet:` / `neumann:` is *not* a cause on current
versions — those keys are matched case-insensitively and a genuine typo warns.
On older versions they were matched exactly, so `Dirichlet:` was skipped in
silence; if you are running an older build, check that first.

**Check.** Confirm the log's DOF line shows constrained DOFs:

```
[SETUP]   DOFs:    530523 total, 529000 free, 1523 constrained
```

`0 constrained` on a problem that should have boundary conditions is the tell.

## An initial condition had no effect

**Cause.** Same class of problem. The `initial conditions` section, its
sub-keys, and `displacement` / `velocity` entries are all unvalidated, so
`velocities:`, `initial condition:`, or `nodeset:` fail silently.

**Also check the integrator.** Quasi-static ignores velocity and traveling-wave
conditions, warning as it does so:

```
[WARNING] Initial velocity ICs ignored for non-Newmark integrator.
```

Traveling-wave entries *are* validated, so they will tell you what is wrong.

## Convergence is far slower than expected

**Cause.** Usually no preconditioner where one was intended — either the
`preconditioner` section was omitted, or `type: none` was left in place. A
misspelled `type` is now a hard error rather than a silent fallthrough, so it
will not cause this.

**Check.** Compare CG iteration counts against a known-good configuration. An
unpreconditioned FEM solve typically takes hundreds to thousands of iterations
where Jacobi takes tens. On large CPU problems, `amg` flattens the count
further; see [Solvers](reference/solvers.md).

## A material property seems not to apply

**Cause.** Unknown property keys are dropped with a warning:

```
[WARNING] Unknown material property key "elastic_modulus"; ignoring.
```

Carina's keys use **spaces**, not underscores: `elastic modulus`,
`Poisson's ratio`, `yield stress`.

## The wrong material is being used

**Cause.** Carina applies a single material to the entire mesh and reads only
the first entry of `blocks`. With more than one block listed, which material
wins is hash order, not file order.

**Fix.** List one block, or give all blocks the same material. See
[Materials](reference/materials.md).

## `beta` and `gamma` are being ignored

**Cause.** A nonzero `alpha` overrides both — β = (1−α)²/4 and γ = (1−2α)/2.

**Fix.** Remove `alpha` (or set it to `0.0`) to control β and γ directly. See
[Time integrators](reference/time-integrators.md).

## The solver converges after one iteration

**Cause.** A `minimum iterations` test is a *convergence* test, not a floor. In
a `converge when any` block it forces convergence as soon as the count is
reached.

**Fix.** Put it in an `all` group, where it does what you meant:

```yaml
    converge when all:
      - minimum iterations: 2
      - any:
          - absolute residual: 1.0e-08
          - relative residual: 1.0e-12
```

## An explicit run blows up

**Cause.** The step exceeds the stability limit. Carina warns when it can tell:

```
[WARNING] Δt = 5.00e-06 exceeds stable Δt = 2.05e-06 — using stable step.
```

**Fix.** Set `CFL` (0.9 is a reasonable default) and let Carina choose the step
rather than hand-tuning `time step`. On large-deformation problems, where
element geometry degrades as the run proceeds, also set `stable time step
interval` so the estimate is refreshed.

Note that a zero or missing `density` makes the stable-step estimate
degenerate — check for the no-density warning.

## Startup errors

| Message | Fix |
|---|---|
| `KeyError: "final time"` / `"time step"` | These two are required and indexed directly, so they error rather than warn. |
| `Adaptive time stepping requires all four: ...` | Supply all of `minimum time step`, `maximum time step`, `decrease factor`, `increase factor`, or none. |
| `decrease factor must be < 1.0` | Decrease shrinks the step; increase grows it. |
| `Material block "X" listed in blocks but no property dict found.` | The name in `blocks` is a reference; add a sibling key with the properties. |
| `Missing required [solver] section.` | Implicit integrators need one. Explicit does not. |
| `Simulation type "multi" not yet supported.` | Only `single` is implemented. |
| `Failed to establish initial equilibrium.` | The `initial equilibrium: true` solve diverged — relax tolerances or start from a better state. |

## GPU

**`--device rocm: no functional AMD GPU found.`** — `cuda` and `rocm` are
strict and will not fall back. Use `--device auto` if you want a CPU fallback.
Verify the device independently with `AMDGPU.versioninfo()` or
`CUDA.versioninfo()`.

**`solver.linear_solver.type = "direct" is CPU-only.`** — the direct solver and
the `ic` and `amg` preconditioners all need an assembled sparse matrix. On GPU
use `cg` with `jacobi` or `chebyshev`, or `lbfgs`.

**A `BoundsError` deep in a kernel, part way through a long run.** This is the
signature of GPU memory exhaustion rather than an indexing bug. Julia's garbage
collector triggers on host memory pressure, not device pressure, so device
temporaries can accumulate between output syncs until VRAM runs out; the
resulting asynchronous fault surfaces at the next bounds-checked kernel. Reduce
the problem size, shorten the output interval, or check for a per-step
allocation in a modified integrator path.

**Running with plain `julia` gives no GPU.** `julia --project=. src/Carina.jl`
bypasses the launcher environment that owns the vendor packages. Use
`bin/carina` for GPU runs.

## Dependency resolution fails

```
ERROR: Unsatisfiable requirements detected for package Exodus [f57ae99e]:
 ├─restricted to versions 0.14 by Carina
 └─restricted to versions 0.15 by FiniteElementContainers — no versions left
```

**Cause.** A sibling package advanced past Carina's compat bound for a shared
dependency. This shows up first in the nightly CI job, which clones the
siblings at `main`.

**Fix.** Update all siblings and re-resolve; if the conflict persists, widen the
relevant `[compat]` entry in `Carina/Project.toml`:

```bash
for r in ConstitutiveModels.jl FiniteElementContainers.jl ReferenceFiniteElements.jl; do
  git -C ../$r pull --ff-only
done
julia --project=. -e 'using Pkg; Pkg.update(); Pkg.resolve()'
```

Local runs can pass while CI fails, because your sibling checkouts may be older
than `main`. Pulling them is the fastest way to reproduce a CI-only failure.

## Benchmark numbers look wrong

Per-interval wall time includes the Exodus write. With a short `output
interval` the I/O dominates and compresses any comparison toward parity. Use a
large interval — a few hundred steps per frame — when measuring compute
throughput.
