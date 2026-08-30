# Solvers

The `solver` section configures the nonlinear solve, the linear solve nested
inside it, and the convergence criteria. It is required for `quasi static` and
`newmark`, and **ignored entirely** by `central difference`, which is explicit.

!!! tip "Looking for the mathematics?"
    This page documents *what to write*. For why each method works — Newton and
    its globalization, Eisenstat–Walker forcing terms, the Chebyshev
    recurrence, smoothed-aggregation multigrid — see the theory manual in
    `theory/`, built with `make`.

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
| `newton` | `newton raphson`, `newton-raphson`, `hessian minimizer` | yes | via linear solver | Newton–Raphson. Quadratic convergence near the solution. **The default choice.** |
| `nonlinear cg` | `nlcg`, `conjugate gradient` | no | yes | Nonlinear CG. Matrix-free, linear convergence. |
| `steepest descent` | `gradient descent`, `sd` | no | yes | Preconditioned steepest descent. Matrix-free, slowest, most robust. |

`type` is required; there is no default. `hessian minimizer` is accepted for
Norma compatibility and builds a plain `NewtonSolver`.

Anything else aborts with the list above. In particular `lbfgs` is **not** a
nonlinear solver type — it is a [linear solver](#Linear-solvers), selected with
`solver.linear solver.type: lbfgs`. That mistake used to fall through to Newton
silently, producing a run that looked fine but was not the algorithm requested:

```
ERROR: Unknown solver.type = "lbfgs". Supported: "newton" (aliases ...).
       L-BFGS is a linear solver: set solver.linear solver.type = "lbfgs".
```

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
    run their own line search, but they do honor the three `line search *`
    tuning keys.

### Nonlinear CG only

| Key | Default | Description |
|---|---|---|
| `orthogonality tolerance` | `0.5` | Restart when consecutive gradients lose orthogonality. |
| `restart interval` | `0` | Force a restart every N iterations. `0` disables. |
| `preconditioner` | none | `type: jacobi` (the default when the section is present) or `type: none`. |

Steepest descent accepts `preconditioner` with the same meaning. Jacobi is
the only preconditioner implemented at this level; asking for anything else
(`chebyshev`, `ic`, `amg`) is a hard error rather than silently running
Jacobi.

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

Four block keys are recognized, each taking a **list**:

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

`combo` must be `and` or `or`; the default when it is omitted is `or`. Any other
value is an error — it previously fell through to `or`, which inverted the
meaning of the group without saying so.

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
- **Relative tests normalize differently.** `relative residual` divides by the
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
| `preconditioner` | none | See [Preconditioners](#Preconditioners). |
| `forcing term` | `fixed` | Per-iteration tolerance for inexact Newton. See [below](#Inexact-Newton-forcing-terms). |
| `assembled` | CPU `true`, GPU `false` | Assembled matrix vs matrix-free operator. |

### Inexact Newton forcing terms

By default every Newton iteration solves its linear system to `tolerance`. That
is more accuracy than the early iterations can use: the Newton step they produce
is about to be discarded by the next correction. On a 530k-DOF Newmark run, CG
spent 181, 205 and 193 iterations on the three solves of a step while ‖R‖ fell
`1.6e4 → 1.3e2 → 7.8e-2 → 4.6e-8`. Only the last one needed eight digits.

A **forcing term** replaces the fixed tolerance with a per-iteration ηₖ, chosen
from how fast the *nonlinear* residual is actually falling:

```yaml
  linear solver:
    type: cg
    tolerance: 1.0e-8
    preconditioner:
      type: jacobi
    forcing term:
      type: eisenstat-walker
```

| `type` | Aliases | Description |
|---|---|---|
| `fixed` | `none`, `constant` | Every iteration solves to `tolerance`. **The default.** |
| `eisenstat-walker` | `eisenstat walker`, `ew`, `adaptive` | Choice 2 of Eisenstat & Walker (1996). |

#### Eisenstat–Walker keys

| Key | Default | Range | Description |
|---|---|---|---|
| `maximum` | `0.2` | (0, 1) | η_max — the loosest tolerance any solve may use. |
| `initial` | = `maximum` | (0, `maximum`] | η for the first Newton iteration of a step, where no ratio exists yet. |
| `gamma` | `1.0` | (0, 1] | Scale factor γ. |
| `exponent` | `1.618…` | (1, 2] | Exponent α; the default is the golden ratio, as in the paper. |
| `safety factor` | `0.5` | ≥ 0 | Over-solve guard strength. `0` disables it. |

Every bound is enforced. Out-of-range values and unknown `type`s are hard
errors rather than silent fallbacks — a typo here would otherwise cost exactly
the speedup the block was added to get, with nothing in the log to say the
setting never took effect.

#### How ηₖ is chosen

The base rule is
`ηₖ = γ (‖Rₖ‖ / ‖Rₖ₋₁‖)^α`,
guarded three ways:

1. **Eisenstat–Walker's own safeguard** stops η falling faster than the observed
   convergence rate justifies: `η ← max(η, γ·η_prev^α)` whenever that term
   exceeds `0.1`.
2. **Kelley's over-solve guard** stops the *final* solve being worked past the
   point the nonlinear test will reward: `η ← max(η, safety·τ/‖Rₖ‖)`, where τ is
   the residual norm your `termination` block is actually aiming at. Carina
   reads τ from the termination tree itself, so a deck's own `converge when`
   thresholds are honored rather than the flat tolerance keys.
3. **A clamp to `[tolerance, maximum]`.**

!!! note "A forcing term can only loosen a solve, never tighten one"
    The lower clamp at your own `tolerance` is what makes this safe to switch
    on. The final Newton iterations still run at exactly the tolerance you
    asked for, so the converged answer is unchanged and an A/B against `fixed`
    compares like with like. Measured across CPU and A100 runs in both implicit
    regimes, `|U|_max` agrees with the fixed-tolerance baseline to every
    printed digit at every output stop.

When a solve is loosened, the log says so:

```
        [SOLVE]       inexact Newton: η = 2.00e-01
        [SOLVE]       CG: 16 iters : |r|_CG = 1.07e-01 : [CONV]
```

#### What to expect

A forcing term trades linear work for nonlinear work: fewer CG iterations, more
Newton iterations. **It therefore pays in proportion to how much of your step
is the linear solve** — most on the GPU matrix-free paths, where the step
essentially *is* the stiffness matvec; least on assembled CPU runs where
residual assembly and the line search already dominate.

Measured with the defaults (see `benchmark/evidence/inexact_newton.txt`):

| Configuration | CG iterations | Per-step wall |
|---|---|---|
| CPU quasi-static, J2 specimen, 57k DOF | 2.47× fewer | 1.30× |
| A100 Newmark, 530k DOF, CG + Jacobi | 2.27× fewer | **1.45×** |
| A100 quasi-static, 530k DOF, CG + AMG | 2.05× fewer | **1.59×** |

The quasi-static AMG gain is on top of the ~2× AMG already has over Jacobi.

!!! tip "Why `maximum` defaults to 0.2 and not the paper's 0.9"
    Total CG work turns out to be nearly *invariant* in `maximum` — 63.1k /
    62.9k / 63.6k iterations at 0.1 / 0.2 / 0.5 on the CPU sweep. What the knob
    really controls is how many Newton iterations you spend buying that work,
    and there the spread is large (413 vs 547).

    The reason is safeguard 1 above, which activates when `γ·η^α > 0.1`. At the
    default α, `0.2^1.618 = 0.076` sits just *below* that threshold, so the
    residual ratio drives η and the method adapts as intended. `0.5^1.618 =
    0.326` sits well above it, pinning η high for several iterations. So 0.2 is
    the largest round value that keeps the safeguard out of the way of the
    adaptivity it is meant to protect — not a fitted constant. Two problems on
    two architectures selected it independently.

    Raise it only if you have measured your own problem.

!!! note "Interaction with AMG"
    None you need to configure, but worth knowing: the AMG hierarchy staleness
    detector compares CG iteration counts, which are not comparable across
    tolerances. Counts from loosened solves are rescaled by the digits of
    residual reduction requested before the detector sees them, so lagged
    rebuilds keep working. With no forcing term the rescaling is exactly the
    identity.

#### References

- S. C. Eisenstat and H. F. Walker, *Choosing the forcing terms in an inexact
  Newton method*, SIAM Journal on Scientific Computing **17**(1):16–32, 1996.
  The ηₖ rule and safeguard 1.
- C. T. Kelley, *Iterative Methods for Linear and Nonlinear Equations*, SIAM,
  1995, §6.3. The over-solve guard, safeguard 2.

### L-BFGS keys

| Key | Default | Description |
|---|---|---|
| `history size` | `10` | Stored gradient pairs. More is a better inverse-Hessian approximation at higher memory cost. |

The L-BFGS path always builds a Jacobi preconditioner regardless of any
`preconditioner` sub-section.

!!! warning "L-BFGS does not work for quasi-static"
    Use it for implicit **dynamics** only. On quasi-static problems it stalls
    roughly seven orders short of tolerance and the step fails — on CPU and GPU
    alike, so this is not a device limitation and no amount of `history size`
    fixes it.

    L-BFGS models the inverse tangent from its last few secant pairs. Implicit
    dynamics adds the mass shift `c_M = 1/(βΔt²)`, which at small Δt leaves the
    effective operator strongly diagonally dominant and easy to model at low
    rank — there L-BFGS is the fastest option Carina has on GPU. Quasi-static
    has no such term: the same system takes 787 AMG iterations to solve, and a
    rank-10 model cannot represent it. Use `cg` + `amg` for quasi-static.

!!! note "`assembled` defaults from the backend"
    Setting `assembled: false` on the CPU forces the matrix-free operator path
    (the same operators the GPU runs); `assembled: true` on a GPU backend is a
    hard error — there is no device sparse matrix to assemble.

## Preconditioners

Configured under `solver.linear solver.preconditioner`.

| `type` | Aliases | CPU | GPU | Cost per iteration | Description |
|---|---|---|---|---|---|
| `jacobi` | — | yes | yes | one vector scale | Diagonal scaling. Cheap, weak, always available. |
| `ic` | `incomplete cholesky`, `ildl`, `incomplete ldlt` | yes | **no** | one triangular solve | Incomplete LDLᵀ. Strong for ill-conditioned systems. |
| `chebyshev` | `chebyshev polynomial` | yes | yes | k matvecs | Polynomial preconditioner; needs only matvecs, so it works on GPU. |
| `amg` | `algebraic multigrid`, `multigrid` | yes | yes | one V-cycle | Smoothed-aggregation AMG with rigid-body-mode near-nullspace. On GPU the hierarchy is built on the host and the V-cycle applies on the device. |
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

**It runs on both CPU and GPU.**  On GPU the hierarchy is built on the host
(the sparse pattern lives there anyway) and converted to device CSR; the
V(2,2)-cycle then applies entirely on the device through KernelAbstractions
kernels, with the fine level smoothed through the matrix-free stiffness action
so the fine matrix is never formed on the device.  It is the fastest
quasi-static option Carina has — see `benchmark_report.md`.

Two behaviors worth knowing. The near-nullspace (six rigid-body modes) is
rebuilt from the **current** nodal coordinates `X + u` at every hierarchy
build, not frozen at the reference configuration — a frozen reference
nullspace degrades badly once the body rotates substantially. And the
hierarchy setup is expensive (seconds at 500k DOF), so it is **lagged**: built
once, then rebuilt only when the effective-mass coefficient changes (a Δt
change) or when CG iteration growth flags it as stale.

AMG combines well with an [Eisenstat–Walker forcing
term](#Inexact-Newton-forcing-terms): the two are independent, and on a
530k-DOF quasi-static torsion run on an A100 the forcing term takes a further
1.59× off the per-step wall on top of what AMG already gives over Jacobi.

AMG targets SPD tangents — quasi-static and moderate-Δt dynamics. At very
large Δt on violently dynamic problems the Newmark predictor can overshoot
into near-inverted configurations whose tangent is indefinite, which breaks CG
regardless of preconditioner; explicit integration is the right tool there.

An unrecognized `type` is a hard error listing the supported values:

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
| **CPU** | **best (small–medium)** | weak | ok | good | ok | **best (large)** | dynamics only\* |
| **GPU** | unavailable | weak | good | unavailable | ok | **best** | dynamics only\* |

\* L-BFGS is the fastest option for implicit dynamics on GPU, but **fails on
quasi-static** on either device — see the warning above.

Practical guidance:

- **CPU, small to medium** — `direct`. No tuning, always converges.
- **CPU, large** — `cg` + `amg`, or `cg` + `ic` as a simpler alternative.
- **GPU, quasi-static** — `cg` + `amg`. It is the only GPU option that both
  converges its linear systems and beats the CPU direct solver at scale, and
  at 1.57M DOF it is the fastest option on either device.
- **GPU, implicit dynamics at small Δt** — `cg` + `jacobi`. The mass term
  conditions the system, so AMG's iteration reduction does not repay its
  per-application cost; see `benchmark_report.md`.
- **Never run plain `cg` with no preconditioner** on a real mesh. It is valid
  and very slow.
- **Add a forcing term to any Newton + CG combination**, on either device.
  `forcing term: {type: eisenstat-walker}` is the cheapest speedup available
  here: it needs no extra memory, changes no kernel, and cannot make the
  converged answer less accurate than your `tolerance` already asked for. It
  helps most where CG dominates the step — the GPU matrix-free paths — and
  least where residual assembly and the line search already do.

### Combinations that fail or mislead

| Combination | Result |
|---|---|
| GPU + `direct` | Hard error at startup. |
| GPU + `ic` preconditioner | Requires an assembled matrix; unavailable. |
| `central difference` + any `solver` block | Silently ignored; explicit has no nonlinear solve. |
| NLCG / SD + `linear solver` | Ignored by design; these are matrix-free. |
| Large Δt + `j2 plasticity` | Path-dependent; large steps miss the yield surface. Keep line search on and steps small. |
