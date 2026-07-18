# Time integrators

The required top-level `time integrator` section selects the time-stepping
scheme and its parameters.

```yaml
time integrator:
  type: quasi static
  initial time: 0.0
  final time:   1.0
  time step:    0.1
```

## Keys common to all integrators

| Key | Required | Default | Description |
|---|---|---|---|
| `type` | no | `quasi static` | Scheme — see below. |
| `initial time` | no | `0.0` | Start time `t₀`. |
| `final time` | **yes** | — | End time `t_f`. Missing key raises a `KeyError`. |
| `time step` | **yes** | — | Nominal step `Δt`. Missing key raises a `KeyError`. |

Unlike most sections, `final time` and `time step` are indexed directly rather
than fetched with a default, so omitting either produces a bare `KeyError`
rather than a friendly message.

## `type` values

| Value | Aliases | Kind |
|---|---|---|
| `quasi static` | `quasistatic`, `static` | Rate-independent equilibrium; no inertia |
| `newmark` | `newmark-beta` | Implicit dynamics (Newmark-β / HHT-α) |
| `central difference` | `centraldifference`, `cd` | Explicit dynamics |

The value is lowercased before matching. An unrecognised type is a hard error.

## Adaptive time stepping

Available to `quasi static` and `newmark` (parsed for `central difference` as
well, though an explicit run is governed by stability rather than convergence).

All four keys must appear together, or none of them:

```yaml
  minimum time step: 0.001
  maximum time step: 0.1
  decrease factor:   0.5
  increase factor:   1.5
```

| Key | Constraint | Description |
|---|---|---|
| `minimum time step` | ≤ `maximum time step` | Floor on Δt. |
| `maximum time step` | — | Ceiling on Δt. |
| `decrease factor` | **must be < 1.0** | Δt multiplier after a failed solve. |
| `increase factor` | **must be > 1.0** | Δt multiplier after a successful solve. |

Supplying some but not all four is an error:

```
Adaptive time stepping requires all four: minimum time step, maximum time step,
decrease factor, increase factor.
```

Each constraint is checked individually, so `decrease factor: 1.5` fails with
`decrease factor must be < 1.0`. When the keys are omitted, Δt is fixed:
`min_dt = max_dt = time step` and both factors are `1.0`.

## Quasi-static

For problems with no inertia. Requires a `solver` section.

```yaml
time integrator:
  type: quasi static
  initial time: 0.0
  final time:   1.0
  time step:    0.1
  initial equilibrium: false
```

| Key | Default | Description |
|---|---|---|
| `initial equilibrium` | `false` | Solve `R = 0` at `t₀` before advancing. |

If `initial equilibrium: true` and that first solve does not converge, the run
aborts with `Failed to establish initial equilibrium.`

Velocity and traveling-wave initial conditions are **ignored** by this
integrator (with a warning) — see [Initial conditions](initial-conditions.md).

## Newmark (implicit dynamics)

Second-order implicit integration. Requires a `solver` section.

```yaml
time integrator:
  type: newmark
  initial time: 0.0
  final time:   0.01
  time step:    1.0e-4
  beta:  0.25
  gamma: 0.5
```

| Key | Aliases | Default | Description |
|---|---|---|---|
| `beta` | `β` | `0.25` | Displacement-update parameter. |
| `gamma` | `γ` | `0.5` | Velocity-update parameter. `0.5` = no numerical damping. |
| `alpha` | — | `0.0` | HHT-α parameter. See the warning below. |

`beta` and `gamma` accept either the ASCII or the Unicode spelling; the ASCII
key is checked first and wins if both are present.

!!! warning "`alpha` overrides `beta` and `gamma`"
    When `alpha` is nonzero, β and γ are **derived** from it and any
    user-supplied `beta`/`gamma` are silently discarded:

    ```
    β = (1 − α)² / 4        γ = (1 − 2α) / 2
    ```

    So `alpha: -0.1` always yields β = 0.3025, γ = 0.6 regardless of what you
    wrote. To set β and γ independently, leave `alpha` at its default of `0.0`.

Common parameter choices:

| Scheme | `beta` | `gamma` | `alpha` | Damping |
|---|---|---|---|---|
| Trapezoidal rule (average acceleration) | 0.25 | 0.5 | 0.0 | none |
| HHT-α, mild | — | — | −0.05 | light |
| HHT-α, moderate | — | — | −0.1 | moderate |

## Central difference (explicit dynamics)

Conditionally stable explicit integration. Uses **no** nonlinear solver — the
`solver` section is not consulted. This is the matrix-free path: Carina opts
the assembler into matrix-free mode, skipping sparse preallocation entirely.

```yaml
time integrator:
  type: central difference
  initial time: 0.0
  final time:   0.001
  time step:    1.0e-7
  gamma: 0.5
  CFL:   0.9
  stable time step interval: 0
```

| Key | Aliases | Default | Description |
|---|---|---|---|
| `gamma` | `γ` | `0.5` | Velocity-update parameter. `0.5` = standard central difference. |
| `CFL` | `cfl` | `0.0` | If > 0, compute a stable Δt and cap `time step` with it. |
| `stable time step interval` | — | `0` | Steps between recomputations of the stable Δt. `0` = compute once at startup only. |

### The CFL cap

When `CFL > 0`, Carina computes a stable step from the element characteristic
length and the material dilatational wave speed,

```
Δt_stable = CFL · min(h) / c_p,     c_p = √(M / ρ)
```

with `M` the p-wave modulus, and then takes `Δt = min(Δt_stable, time step)`.
The estimate is logged at startup:

```
[SETUP]   Stable Δt = 2.05e-06 (CFL = 0.90)
```

If your requested `time step` exceeds the stable step, Carina warns and uses
the stable one — it will not run an unstable explicit simulation just because
you asked:

```
[WARNING] Δt = 5.00e-06 exceeds stable Δt = 2.05e-06 — using stable step.
```

Setting `stable time step interval: N` with `N > 0` recomputes the estimate
every `N` steps, which matters when large deformation changes element
geometry enough to shrink the stable step mid-run. Leaving it at `0` computes
the estimate once, from the undeformed configuration.

Note that the stable-step computation needs a nonzero `density`; see
[Materials](materials.md).
