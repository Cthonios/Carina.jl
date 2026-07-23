# Troubleshooting

Carina aims to fail loudly: every input section is key-validated, every value
drawn from a fixed set is checked against it, and every mesh name is verified
against the mesh file. So start by reading the log from the top — the answer is
usually a `[WARNING]` line you scrolled past. This page covers what is left.

!!! note "If you are on an older build"
    Much of this validation is recent. Older versions silently skipped a
    mistyped section rather than rejecting it, which is why the entries below
    still describe the silent symptom alongside the message you should now see.

## The simulation runs but nothing is constrained

**Symptom.** The body drifts or rotates freely, a quasi-static solve fails
immediately on a singular system, or displacements are orders of magnitude too
large.

**Cause.** A `dirichlet:` list that never reached the parser. On current
versions this is hard to do: `dirichlet` / `neumann` are matched
case-insensitively, a genuine typo warns, and a `side set` or `node set` that
is absent from the Exodus file aborts with the available names. On older
versions all three failed silently.

**Check.** Confirm the log's DOF line shows constrained DOFs:

```
[SETUP]   DOFs:    530523 total, 529000 free, 1523 constrained
```

`0 constrained` on a problem that should have boundary conditions is the tell.

## An initial condition had no effect

**Cause.** On current versions a misspelled list key warns
(`Unknown key "velocities" in initial conditions`), a misspelled entry key warns
and then aborts on the missing required field, and an unknown `node set` aborts
with the mesh's node-set names. All three were silent on older versions, which
made the run start from rest and look like a physics result.

**Also check the integrator.** Quasi-static ignores velocity and traveling-wave
conditions, warning as it does so:

```
[WARNING] Initial velocity ICs ignored for non-Newmark integrator.
```

This warning is the one remaining case where an initial condition is legitimately
dropped, so it is worth grepping for.

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

A zero or missing `density` would make the stable-step estimate degenerate, but
it cannot reach this point: a dynamic run with no density is rejected at
startup.

## Startup errors

| Message | Fix |
|---|---|
| `KeyError: "final time"` / `"time step"` | These two are required and indexed directly, so they error rather than warn. |
| `Adaptive time stepping requires all four: ...` | Supply all of `minimum time step`, `maximum time step`, `decrease factor`, `increase factor`, or none. |
| `decrease factor must be < 1.0` | Decrease shrinks the step; increase grows it. |
| `Material model "X" is assigned to block "Y" ... but [model.material] has no "X" property dict.` | The name in `blocks` is a reference; add a sibling key with the properties. The message lists the property dicts you did write. |
| `... refers to element block / node set / side set "X", which is not in the mesh.` | A mesh-name typo. The message suggests the closest match and lists what the mesh actually contains. |
| `[model.material.blocks] lists N blocks ...` | One material per simulation; name a single block. |
| `Unknown model.type = "X".` | Only `solid mechanics` exists. |
| `Unknown solver.type = "X".` | Note that `lbfgs` is a *linear* solver — set it under `linear solver`. |
| `... is missing required key "X".` | A boundary condition, body force, or initial condition entry is incomplete. The message names the section and entry index. |
| `... has density 0.0, but a dynamic time integrator requires a mass matrix.` | Set `density`, or check it for a typo — an unrecognised property key is dropped with a warning. |
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
