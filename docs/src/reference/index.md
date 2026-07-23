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
| `body forces` | no | [Boundary conditions](boundary-conditions.md#Body-forces) |
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

Carina reports input mistakes four different ways, and knowing which to expect
saves time when debugging a file.

**Unknown keys warn.** Every section is checked against a set of known keys —
the top level, `model`, `model.material`, `time integrator`, `solver`,
`linear solver`, `quadrature`, `output`, `boundary conditions`,
`initial conditions`, and the individual Dirichlet, Neumann, body-force,
initial-condition, and traveling-wave entries. An unrecognised key produces a
warning with a Levenshtein-based suggestion:

```
[WARNING] Unknown key "time_step" in time integrator. Did you mean "time step"?
```

The run continues, using the default for whatever you meant to set.

**Missing required keys error.** Absent mesh files, model sections, or
`final time` / `time step` abort at startup. So does an entry that omits a
field it needs — an initial condition without a `node set`, for instance,
naming the section and the entry index.

**Unknown *values* error.** Where a key selects behaviour from a fixed set —
`model.type`, `time integrator.type`, `solver.type`, `linear solver.type`,
`preconditioner.type`, `quadrature.type`, `output.recovery`, `combo` — an
unrecognised value aborts with the list of supported spellings. None of these
fall back to a default. The distinction from the warn-only case is deliberate:
a key you meant to set but misspelled leaves the default in place, which is
usually harmless and always visible in the log, whereas a *value* you misspelled
means you asked for something Carina does not have.

**Mesh names are checked against the mesh.** Every element block, node set, and
side set named in the input is verified as soon as the mesh is read, before any
of them is used:

```
Dirichlet BC entry 4 refers to side set "ssz_", which is not in the mesh.
Did you mean "ssz-"? Available: ssx+, ssx-, ssy+, ssy-, ssz+, ssz-.
```

**Values are matched case-insensitively**, and so are the sub-keys of
`boundary conditions` (`dirichlet` / `neumann`) and the material property dicts
under `model.material`. Everything else is matched exactly — but validated, so a
casing mistake warns rather than passing unnoticed.

## Conventions

- **Keys use spaces, not underscores.** `time step`, not `time_step`; `elastic
  modulus`, not `elastic_modulus`. Error messages sometimes render key names
  with underscores; the accepted spelling is the spaced one.
- **Values are matched case-insensitively**, after trimming. `Newton`,
  `newton`, and `NEWTON` are the same. Section and key *names* are matched
  exactly, apart from the two exceptions noted above, but every section is
  validated — so a casing mistake warns.
- **Quote function expressions** so YAML does not reinterpret them.
- **Components are `x`, `y`, `z`** everywhere they appear.
