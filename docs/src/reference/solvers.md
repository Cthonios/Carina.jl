# Solvers

The `solver` section configures the nonlinear solve, the linear solve nested
inside it, and the convergence criteria. It is required for `quasi static` and
`newmark`, and **ignored entirely** by `central difference`, which is explicit.

```yaml
solver:
  type: newton
  linear solver:
    type: cg
    preconditioner:
      type: jacobi
  termination:
    converge when any:
      - absolute residual: 1.0e-08
      - relative residual: 1.0e-12
    fail when any:
      - maximum iterations: 16
```

## Nonlinear solvers

| `type` | Aliases | Needs a linear solver? | GPU | Description |
|---|---|---|---|---|
| `newton` | `hessian minimizer` | yes | via linear solver | Newton–Raphson. Quadratic convergence near the solution. **The default choice.** |
| `nonlinear cg` | `nlcg`, `conjugate gradient` | no | yes | Nonlinear CG. Matrix-free, linear convergence. |
| `steepest descent` | `gradient descent`, `sd` | no | yes | Preconditioned steepest descent. Matrix-free, slowest, most robust. |

`type` is required; there is no default. `hessian minimizer` is accepted for
compatibility and builds a plain `NewtonSolver`.

For `newton`, a `linear solver` sub-section is **required**. For the two
matrix-free solvers it is optional and defaults to `type: none`.

### Keys common to all nonlinear solvers

| Key | Default | Description |
|---|---|---|
| `minimum iterations` | `0` | Lower bound on iterations. |
| `maximum iterations` | `20` | Upper bound. **Overridden** by a `maximum iterations` test in `termination` if one is present. |
| `absolute tolerance` | `1e-10` | Used only when `termination` is absent. |
| `relative tolerance` | `1e-14` | Used only when `termination` is absent. |
| `use line search` | **`true`** | Armijo backtracking on ½‖R‖². Applies to `newton` only. |
| `line search backtrack factor` | `0.5` | Step reduction per backtrack. |
| `line search decrease factor` | `1e-4` | Armijo sufficient-decrease parameter. |
| `line search maximum iterations` | `10` | Maximum backtracking steps. |

!!! note "Line search defaults to ON"
    At large Δt or load increments the predictor can take element-inverting
    full steps; Armijo backtracking guards against that. It costs roughly one
    extra residual evaluation per iteration when the full step is already good
    (α = 1 is accepted immediately). Set `use line search: false` to disable.

    `use line search` reaches `newton` only. NLCG and steepest descent always
    run their own line search, but they do honour the three `line search *`
    tuning keys.

### Nonlinear CG only

| Key | Default | Description |
|---|---|---|
| `orthogonality tolerance` | `0.5` | Restart when consecutive gradients lose orthogonality. |
| `restart interval` | `0` | Force a restart every N iterations. `0` disables. |
| `preconditioner` | none | Presence of this key (any value) enables the built-in Jacobi preconditioner. |

Steepest descent accepts `preconditioner` with the same meaning. For both
solvers the key acts as a flag — its *contents* are not read, and the
preconditioner built is always the Jacobi one.

## Termination criteria

Carina accepts three syntaxes. Prefer the first.

### Preferred: `converge when` / `fail when`

```yaml
  termination:
    converge when any:
      - absolute residual: 1.0e-08
      - relative residual: 1.0e-12
    fail when any:
      - maximum iterations: 16
      - divergence: 1.0e6
```

Four block keys are recognised, each taking a **list**:

| Block key | Combines with |
|---|---|
| `converge when any` | OR |
| `converge when all` | AND |
| `fail when any` | OR |
| `fail when all` | AND |

Each list item is a single-key mapping of `test name: value`. Groups nest via
`any:` / `all:` inside a list:

```yaml
  termination:
    converge when all:
      - minimum iterations: 2
      - any:
          - absolute residual: 1.0e-08
          - relative residual: 1.0e-12
```

### Legacy: typed list

Still supported. Each entry carries an explicit `type`, with the value under
`tolerance`, `value`, `threshold`, or `window` depending on the test, and
`combo`/`tests` for nesting:

```yaml
  termination:
    - type: combo
      combo: and
      tests:
        - type: absolute residual
          tolerance: 1.0e-06
        - type: relative update
          tolerance: 1.0e-12
    - type: maximum iterations
      value: 16
```

### Oldest: flat tolerances

If `termination` is absent entirely, Carina builds
`OR(absolute residual, relative residual, finite value)` from the flat
`absolute tolerance` (default `1e-10`) and `relative tolerance` (default
`1e-14`) keys.

### Available tests

| Test name | Aliases | Value means | Signals |
|---|---|---|---|
| `absolute residual` | `abs_residual` | tolerance | Converged when ‖R‖ < tol |
| `relative residual` | `rel_residual` | tolerance | Converged when ‖R‖/‖R₀‖ < tol |
| `absolute update` | `abs_update` | tolerance | Converged when ‖ΔU‖ < tol |
| `relative update` | `rel_update` | tolerance | Converged when ‖ΔU‖/‖U‖ < tol |
| `maximum iterations` | `max iterations` | iteration count | **Failed** when iter ≥ value |
| `minimum iterations` | `min iterations` | iteration count | **Converged** when iter ≥ value |
| `finite value` | `nan check` | (ignored) | **Failed** when ‖R‖ is not finite |
| `divergence` | — | threshold | **Failed** when ‖R‖ > threshold·‖R₀‖ |
| `stagnation` | — | window | **Failed** on insufficient residual reduction |

Points worth knowing:

- **`minimum iterations` is a convergence test, not a floor.** It reports
  *Converged* once the iteration count is reached. Placed in a `converge when
  any` block it will force convergence on that iteration. It is only
  meaningful inside an `all` group, where it prevents premature convergence.
- **Relative tests normalise differently.** `relative residual` divides by the
  **initial** residual ‖R₀‖; `relative update` divides by the **current**
  solution norm ‖U‖. Each is inert while its denominator is zero — a
  `relative update` test cannot converge on the first iteration from a zero
  initial guess.
- **`divergence` compares against ‖R₀‖**, not the previous iterate, so it
  detects net growth from the start rather than a single bad step.
- **`stagnation` uses its value twice**: as the lookback distance *and* as the
  number of consecutive stagnant iterations required to fail. With the default
  window of 5 it needs at least 11 iterations before it can trigger. Its
  internal ratio floor is 0.95 — i.e. failure when ‖R_k‖/‖R_{k−window}‖ > 0.95
  — settable only through the legacy syntax's `tolerance` key.
- **A `finite value` test is appended automatically** to every termination
  tree, in all three syntaxes. You do not need to add one.
- **`maximum iterations` in the tree wins.** If the tree contains a
  `maximum iterations` test, its value replaces the flat `maximum iterations`
  key as the solver loop bound.
- **In an OR group, Converged beats Failed.** If one sub-test converges while
  another fails on the same iteration, the result is Converged. An AND group
  short-circuits on the first Failed.

## Linear solvers

Configured under `solver.linear solver`.

| `type` | Aliases | CPU | GPU | Description |
|---|---|---|---|---|
| `direct` | — | yes | **no** | Sparse LU. Robust, no tuning. |
| `iterative` | `cg`, `krylov`, `minres`, `conjugate gradient` | yes | yes | Conjugate gradient. All aliases produce CG — the stiffness is SPD. |
| `lbfgs` | — | yes | yes | L-BFGS quasi-Newton. Matrix-free. |
| `none` | — | yes | yes | No linear solve; for NLCG / steepest descent. |

Selecting `direct` on a GPU backend is a hard error:

```
solver.linear_solver.type = "direct" is CPU-only.
```

### Iterative keys

| Key | Default | Description |
|---|---|---|
| `maximum iterations` | `1000` | CG iteration cap. |
| `tolerance` | `1e-8` | Relative residual tolerance. |
| `preconditioner` | none | See below. |

### L-BFGS keys

| Key | Default | Description |
|---|---|---|
| `history size` | `10` | Stored gradient pairs. More is a better inverse-Hessian approximation at higher memory cost. |

The L-BFGS path always builds a Jacobi preconditioner regardless of any
`preconditioner` sub-section.

!!! note "`assembled` is accepted but computed"
    `assembled` is a recognised key in the linear-solver section, but its value
    is not read — Carina sets it from the backend (`true` on CPU, `false` on
    GPU). Setting it in YAML has no effect.

## Preconditioners

Configured under `solver.linear solver.preconditioner`.

| `type` | Aliases | CPU | GPU | Cost per iteration | Description |
|---|---|---|---|---|---|
| `jacobi` | — | yes | yes | one vector scale | Diagonal scaling. Cheap, weak, always available. |
| `ic` | `incomplete cholesky`, `ildl`, `incomplete ldlt` | yes | **no** | one triangular solve | Incomplete LDLᵀ. Strong for ill-conditioned systems. |
| `chebyshev` | `chebyshev polynomial` | yes | yes | k matvecs | Polynomial preconditioner; needs only matvecs, so it works on GPU. |
| `amg` | `algebraic multigrid`, `multigrid` | yes | **no** | one V-cycle | Smoothed-aggregation AMG with rigid-body-mode near-nullspace. |
| *(omitted)* | — | yes | yes | none | No preconditioning. |

### Chebyshev keys

| Key | Default | Description |
|---|---|---|
| `degree` | `5` | Polynomial degree. Higher is stronger but costs `degree` matvecs per iteration. |

### AMG

```yaml
  linear solver:
    type: cg
    tolerance: 1.0e-8
    maximum iterations: 500
    preconditioner:
      type: amg
```

AMG is the only preconditioner whose CG iteration count is nearly independent
of the conditioning that defeats Jacobi — on a 530k-DOF torsion problem it
holds 5–17 CG iterations across a range of Δt where Jacobi grows from 30 to
442.

It requires the CPU assembled path:

```
preconditioner.type = "amg" requires the CPU assembled path (GPU AMG not yet implemented).
```

Two behaviours worth knowing. The near-nullspace (six rigid-body modes) is
rebuilt from the **current** nodal coordinates `X + u` at every hierarchy
build, not frozen at the reference configuration — a frozen reference
nullspace degrades badly once the body rotates substantially. And the
hierarchy setup is expensive (seconds at 500k DOF), so it is **lagged**: built
once, then rebuilt only when the effective-mass coefficient changes (a Δt
change) or when CG iteration growth flags it as stale.

AMG targets SPD tangents — quasi-static and moderate-Δt dynamics. At very
large Δt on violently dynamic problems the Newmark predictor can overshoot
into near-inverted configurations whose tangent is indefinite, which breaks CG
regardless of preconditioner; explicit integration is the right tool there.

An unrecognised `type` is a hard error listing the supported values:

```
Unknown preconditioner.type = "jacobbi". Supported: "jacobi", "ic" (aliases
"incomplete cholesky", "ildl", "incomplete ldlt"), "chebyshev", "amg"
(aliases "algebraic multigrid", "multigrid"), "none".
```

Omitting the `preconditioner` section entirely, or writing `type: none`, means
no preconditioning — that remains a valid, unremarkable choice.

## Choosing a combination

### Integrator and nonlinear solver

| | Newton | Nonlinear CG | Steepest descent |
|---|:---:|:---:|:---:|
| Quasi-static | **recommended** | ok | fallback |
| Newmark | **recommended** | ok | fallback |
| Central difference | n/a — explicit | n/a | n/a |

### Newton, by device

| | `direct` | CG | CG + Jacobi | CG + IC | CG + Chebyshev | CG + AMG | L-BFGS |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CPU** | **best (small–medium)** | weak | ok | good | ok | **best (large)** | good |
| **GPU** | unavailable | weak | **good** | unavailable | good | unavailable | good |

Practical guidance:

- **CPU, small to medium** — `direct`. No tuning, always converges.
- **CPU, large** — `cg` + `amg`, or `cg` + `ic` as a simpler alternative.
- **GPU** — `cg` + `jacobi` or `cg` + `chebyshev`; `lbfgs` is a good
  matrix-free option.
- **Never run plain `cg` with no preconditioner** on a real mesh. It is valid
  and very slow.

### Combinations that fail or mislead

| Combination | Result |
|---|---|
| GPU + `direct` | Hard error at startup. |
| GPU + `ic` preconditioner | Requires an assembled matrix; unavailable. |
| GPU + `amg` preconditioner | Hard error at startup. |
| `central difference` + any `solver` block | Silently ignored; explicit has no nonlinear solve. |
| NLCG / SD + `linear solver` | Ignored by design; these are matrix-free. |
| Large Δt + `j2 plasticity` | Path-dependent; large steps miss the yield surface. Keep line search on and steps small. |
