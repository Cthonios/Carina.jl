# Input file reference

Carina is driven by a single YAML input file, passed on the command line:

```bash
bin/carina input.yaml
```

This reference documents every section and key of that file.

## Top-level sections

| Key | Required | Purpose |
|---|---|---|
| `type` | no | Simulation type. Only `single` is implemented. |
| `device` | no | `cpu`, `cuda`, `rocm`, or `auto`. See [Running Carina](../running.md). |
| `input mesh file` | **yes** | [Mesh and output files](mesh-and-io.md) |
| `output mesh file` | **yes** | [Mesh and output files](mesh-and-io.md) |
| `output interval` | no | Output cadence — [Mesh and output files](mesh-and-io.md) |
| `model` | **yes** | Physics and material — [Model](model.md), [Materials](materials.md) |
| `time integrator` | **yes** | [Time integrators](time-integrators.md) |
| `solver` | implicit only | [Solvers](solvers.md). Required for `quasi static` and `newmark`; ignored by `central difference`. |
| `boundary conditions` | no | [Boundary conditions](boundary-conditions.md) |
| `body forces` | no | [Boundary conditions](boundary-conditions.md#body-forces) |
| `initial conditions` | no | [Initial conditions](initial-conditions.md) |
| `output` | no | Field selection — [Output fields](output.md) |
| `quadrature` | no | [Quadrature](quadrature.md) |

`type` defaults to `single`, the only supported value. Anything else is a hard
error:

```
Simulation type "multi" not yet supported. Only "single" is implemented.
```

Multidomain and Schwarz coupling — a central capability of
[Norma.jl](https://github.com/sandialabs/Norma.jl) — are design goals for
Carina but are not implemented.

## A complete minimal example

An explicit dynamic simulation of a cube with one fixed face:

```yaml
type: single

input mesh file:  cube.g
output mesh file: cube.e

model:
  type: solid mechanics
  material:
    blocks:
      block_1: neohookean
    neohookean:
      elastic modulus: 1.0e9
      Poisson's ratio: 0.25
      density: 1000.0

time integrator:
  type: central difference
  initial time: 0.0
  final time:   1.0e-3
  time step:    1.0e-7
  CFL: 0.9

output interval: 1.0e-4

boundary conditions:
  dirichlet:
    - side set: ssz-
      component: z
      function: "0.0"
```

An implicit quasi-static run adds a `solver` section:

```yaml
time integrator:
  type: quasi static
  initial time: 0.0
  final time:   1.0
  time step:    0.1

solver:
  type: newton
  linear solver:
    type: direct
  termination:
    converge when any:
      - absolute residual: 1.0e-08
      - relative residual: 1.0e-12
    fail when any:
      - maximum iterations: 16
```

## How input errors surface

Carina validates in two different ways, and the difference matters when
debugging an input file.

**Unknown keys warn.** Most sections are checked against a set of known keys,
and an unrecognised key produces a warning with a Levenshtein-based suggestion:

```
[WARNING] Unknown key "time_step" in time integrator. Did you mean "time step"?
```

The run continues, using the default for whatever you meant to set.

**Missing required keys error.** Absent mesh files, model sections, or
`final time` / `time step` abort at startup.

**Some sections are not checked at all.** Key validation is applied to the
top level, `time integrator`, `solver`, `linear solver`, `output`, and
individual Dirichlet, Neumann, body-force, and traveling-wave entries. It is
**not** applied to:

| Section | Consequence of a typo |
|---|---|
| `model` | Silently ignored |
| `quadrature` | Silently ignored |
| `boundary conditions` sub-keys | `Dirichlet:` instead of `dirichlet:` silently applies **no** boundary conditions |
| `initial conditions` and its sub-keys | Silently applies no initial condition |
| `displacement` / `velocity` IC entries | Silently ignored |

The boundary-condition case is the most damaging, because a model with no
constraints still runs — it just drifts or produces a singular system. When a
simulation behaves as though a section were absent, check its spelling and
capitalisation before suspecting the physics.

## Conventions

- **Keys use spaces, not underscores.** `time step`, not `time_step`; `elastic
  modulus`, not `elastic_modulus`. Error messages sometimes render key names
  with underscores; the accepted spelling is the spaced one.
- **Values are matched case-insensitively**, after trimming. `Newton`,
  `newton`, and `NEWTON` are the same. Section and key *names* are not — those
  are exact.
- **Quote function expressions** so YAML does not reinterpret them.
- **Components are `x`, `y`, `z`** everywhere they appear.
