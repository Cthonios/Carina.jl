# Mesh and output files

```yaml
input mesh file:  cube.g
output mesh file: cube.e
output interval:  1.0e-4
```

| Key | Required | Default | Description |
|---|---|---|---|
| `input mesh file` | **yes** | — | Exodus mesh to read. |
| `output mesh file` | **yes** | — | Exodus file to write results to. |
| `output interval` | no | `time step` | Simulation time between output frames. |

## Path resolution

Relative paths are resolved against **the directory containing the YAML input
file**, not the current working directory. Absolute paths are used as-is.

This means an input file referring to `cube.g` finds the mesh next to itself,
so a simulation directory can be run from anywhere:

```bash
bin/carina examples/mechanics/quasistatic/cube/cube.yaml   # works from repo root
```

Both keys are required. Omitting either is a hard error:

```
Missing required input key: "input mesh file"
```

Mesh entities referenced elsewhere in the input — side set and node set names
in [boundary conditions](boundary-conditions.md) and [initial
conditions](initial-conditions.md), block names in `body forces` — must match
names defined in the Exodus mesh.

## Output interval and subcycling

`output interval` sets how often results are written, independently of the
integrator's time step. When omitted it equals `time step`, giving one frame
per step.

The number of output frames is

```
num_stops = round((final time − initial time) / output interval) + 1
```

counting the initial state, which is always written.

When `output interval` is larger than `time step`, the integrator **subcycles**:
it takes as many internal steps as needed to reach the next output time, and
only then writes. This is the normal configuration for explicit dynamics, where
a stable step may be nanoseconds while you want output every few microseconds.

```yaml
time integrator:
  type: central difference
  time step: 1.0e-7
  final time: 1.0e-3
output interval: 1.0e-4        # 10 frames, ~1000 steps each
```

Carina reports which regime it is in at startup — either

```
[SETUP]   Time: [0, 1e-3], Δt = 1e-7, 10000 steps
```

when the interval equals the step, or

```
[SETUP]   Time: [0, 1e-3], Δt = 1e-7, output every 1.00e-04 (11 stops)
```

when it does not.

### Landing exactly on output times

Output times are exact. Rather than take a sliver step to land on a stop,
Carina stretches the last step of each interval: if the next full step would
finish within half a step of the output time, the whole remaining gap is taken
as one step instead. The actual step can therefore reach 1.5× the nominal Δt at
the end of an interval.

For explicit runs this matters — a stretched step is still bounded by the
stability limit only if your nominal Δt has headroom. Using the `CFL` key
rather than a hand-tuned `time step` leaves that headroom automatically; see
[Time integrators](time-integrators.md).

With adaptive stepping enabled the integrator may additionally shrink or grow
its step within an interval in response to solver convergence, independently of
this snapping behaviour.

## Output content

Which fields are written — and how element quantities are projected to nodes —
is controlled by the separate `output` section, documented in [Output
fields](output.md).
