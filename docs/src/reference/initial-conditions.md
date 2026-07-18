# Initial conditions

The optional top-level `initial conditions` section sets the displacement and
velocity fields at `t₀`. It holds three lists: `displacement`, `velocity`, and
`traveling wave`.

```yaml
initial conditions:
  displacement:
    - node set: nsall
      component: z
      function: "0.001 * x"
  velocity:
    - node set: nsall
      component: x
      function: "100.0 * sin(pi * y)"
```

Each of the three keys must hold a **list**, even for a single entry. A scalar
or mapping raises e.g. `initial_conditions.displacement must be a list.`

!!! warning "Misspelled keys fail silently"
    The `initial conditions` section is not key-validated, and neither are
    `displacement` / `velocity` entries. A typo in the section key (`initial
    condition:`), in a list key (`velocities:`), or in an entry key
    (`nodeset:`) produces **no warning** — the initial condition is simply
    never applied and the run starts from rest.

    Traveling-wave entries are the exception: they are fully validated.

    If a simulation starts from an unexpected state, check these spellings
    first.

## Displacement and velocity

| Key | Required | Description |
|---|---|---|
| `node set` | yes | Exodus node set name. |
| `component` | yes | `x`, `y`, or `z`. |
| `function` | yes | Expression in `x`, `y`, `z`, `t`. See [Function expressions](functions.md). |

Only **free** degrees of freedom receive initial values; DOFs constrained by a
Dirichlet condition are skipped, and their values come from the boundary
condition instead.

### Integrator support

| Integrator | `displacement` | `velocity` | `traveling wave` |
|---|:---:|:---:|:---:|
| `quasi static` | applied | **ignored** | **ignored** |
| `newmark` | applied | applied | applied |
| `central difference` | applied | applied | applied |

Velocity and traveling-wave conditions are meaningful only to the dynamic
integrators. Quasi-static runs drop them with a warning rather than an error:

```
[WARNING] Initial velocity ICs ignored for non-Newmark integrator.
```

The message says "non-Newmark"; it applies to any non-dynamic integrator, which
in practice means quasi-static.

## Traveling wave

A traveling-wave condition sets a displacement profile *and* the velocity field
consistent with it propagating along an axis, without you having to write the
derivative by hand.

```yaml
initial conditions:
  traveling wave:
    - node set: nsall
      component: z
      displacement: "a=1.0e-3; s=0.1; a*exp(-(x-0.5)^2/s^2)"
      direction: x
      wave speed: 1000.0
```

| Key | Required | Description |
|---|---|---|
| `node set` | **yes** | Exodus node set name. |
| `component` | **yes** | `x`, `y`, or `z` — the displaced component. |
| `displacement` | **yes** | Expression for the initial profile `u₀`. |
| `direction` | **yes** | Propagation axis: `x`, `y`, or `z`. |
| `wave speed` | **yes** | Signed speed `c`. |

All five keys are required and validated; a missing one is a hard error naming
the key, and `direction` must be `x`, `y`, or `z`.

Given `u(x, t) = u₀(s − c t)` with `s` the coordinate along the propagation
axis, the consistent initial velocity is

```
v(x, 0) = −c · ∂u₀/∂s
```

Carina forms `∂u₀/∂s` by **symbolic differentiation** of your `displacement`
expression with respect to the `direction` axis, then evaluates both fields at
`t₀`. There is no finite differencing and no automatic differentiation at
initialization time.

`wave speed` is **signed**: the sign selects the direction of travel along the
axis. Note that `direction` names the propagation axis while `component` names
the displaced component — for a shear wave travelling along `x` and displacing
`z`, use `direction: x` with `component: z`.

## Application order

Conditions are applied in a fixed order:

1. `displacement` entries
2. `velocity` entries
3. `traveling wave` entries
4. initial acceleration solve
5. Dirichlet values propagated into the state

Traveling-wave conditions come **after** the plain lists deliberately, so that
the derived velocity field overrides any zero-velocity default the earlier
lists left in place. If a node appears in both a `velocity` entry and a
`traveling wave` entry, the traveling wave wins.

At step 4 Carina solves for the initial acceleration. When the residual is
already negligible it reports:

```
[SETUP]   Initial Acceleration = 0 (trivial RHS, ...)
```

which is informational, not an error.
