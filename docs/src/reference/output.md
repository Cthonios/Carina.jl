# Output fields

The optional top-level `output` section selects which fields are written to the
Exodus output file. Output *cadence* is controlled separately by `output
interval` — see [Mesh and output files](mesh-and-io.md).

```yaml
output:
  velocity: true
  acceleration: true
  stress: true
  deformation gradient: false
  internal variables: true
  recovery: lumped
```

| Key | Type | Default | Description |
|---|---|---|---|
| `velocity` | bool | `false` | Nodal velocity. Dynamic integrators only. |
| `acceleration` | bool | `false` | Nodal acceleration. Dynamic integrators only. |
| `stress` | bool | **`true`** | Per-quadrature-point Cauchy stress. |
| `deformation gradient` | bool | **`true`** | Per-quadrature-point deformation gradient. |
| `internal variables` | bool | `false` | Per-quadrature-point constitutive state. |
| `recovery` | string | **`lumped`** | Nodal projection method for element fields. |

!!! note "Stress and deformation gradient are on by default"
    Omitting the `output` section does **not** give you a displacement-only
    file. `stress`, `deformation gradient`, and `lumped` recovery are enabled
    by default, which on a large mesh is a substantial amount of data. Set them
    to `false` explicitly for lean output.

Keys in this section are validated, so a misspelling produces an unknown-key
warning with a suggestion.

## Always written

Displacement and the time value are written unconditionally, regardless of the
`output` section:

| Variable | Meaning |
|---|---|
| `displ_x`, `displ_y`, `displ_z` | Nodal displacement components |

## Gated nodal fields

| Variable | Written when |
|---|---|
| `velo_x`, `velo_y`, `velo_z` | `velocity: true` **and** the integrator is dynamic |
| `acce_x`, `acce_y`, `acce_z` | `acceleration: true` **and** the integrator is dynamic |

Under `quasi static` these flags have no effect — there is no velocity or
acceleration state to write.

## Element fields

Element variables carry a trailing quadrature-point index `_1`, `_2`, … The
index suffix is deliberate: it stops ParaView from auto-grouping the components
into a tensor it would then mislabel.

| Field | Names | Component order |
|---|---|---|
| Stress | `sigma_xx_q` … `sigma_zz_q` | `xx, xy, xz, yy, yz, zz` (symmetric) |
| Deformation gradient | `F_xx_q` … `F_zz_q` | `xx, yx, zx, xy, yy, zy, xz, yz, zz` |
| Internal variables | `<name>_q` | From the constitutive model, e.g. `eqps_1` |

!!! warning "`F` component order is column-major"
    The deformation-gradient components are written in column-major order, so
    the **first** index is the row: `F_yx` is component (2,1), not (1,2). Stress
    is symmetric so its ordering is unambiguous.

The number of quadrature-point suffixes follows the [quadrature](quadrature.md)
rule. Names are registered for the block with the most quadrature points, so on
a mixed mesh some names may exist without values in blocks that have fewer.

Internal variables are emitted only when the constitutive model actually has
state — an elastic model produces none regardless of the flag.

The entire element-field block, including the GPU-to-host transfer of the state
array, is skipped when `stress`, `deformation gradient`, and `internal
variables` are all `false`. On GPU runs that transfer is the dominant output
cost, so turning these off is worth doing when you do not need them.

## Recovered nodal fields

Element quantities live at quadrature points; `recovery` controls how they are
projected to nodes for visualization. Recovered variables take a trailing `_n`.

| `recovery` | Aliases | Method |
|---|---|---|
| `lumped` | — | Projection with the lumped (row-sum) mass of the vertex space. Fast, slightly smoothing. **Default.** |
| `consistent` | `l2` | Full consistent-mass L² projection onto the vertex space; requires a Cholesky factorization at setup. |
| `none` | — | No projection; skip recovered output. |

The projection space is the continuous piecewise-linear space on the vertices
of the elements. For elements whose nodes are all vertices (`TETRA4`, `HEX8`)
that is the element's own basis. For elements with edge, face or interior
nodes (`TETRA10`, `TETRA15`) the projection is onto the four vertices of each
tetrahedron and the other nodes are interpolated from the vertices afterwards
(edge nodes the mean of their two vertices, face nodes of their three, the
interior node of the four). The full basis of those elements is not usable
for this purpose: its row-sum lumped mass is zero or negative at vertices,
and its consistent mass integrated with the element's quadrature rule (degree
5 for the fifteen-node tetrahedron, whose basis products reach degree 8) is
singular on the interior bubble; before 2026-10-15 both were attempted, and
on `TETRA15` the lumped output held stresses of 1e12 Pa against element
values of 1e7 Pa and the consistent one failed to factorize. The vertex mass
is integrated exactly by every rule of degree two or more and its row sums
are positive.

| Variable | Written when |
|---|---|
| `sigma_xx_n` … `sigma_zz_n` | `recovery` ≠ `none` and `stress: true` |
| `<name>_n` | `recovery` ≠ `none` and `internal variables: true` |

The value is lowercased and trimmed before matching, so `Lumped` and `L2` are
fine. An unrecognized value is a hard error naming the supported set:

```
Unknown output.recovery = "lmped". Supported: "lumped", "consistent" (alias "l2"), "none".
```

!!! warning "Recovered deformation-gradient names are declared but never written"
    With `deformation gradient: true` and recovery enabled, `F_*_n` variables
    appear in the Exodus file's variable list but are never populated. Use the
    per-quadrature-point `F_*_q` values instead.

## Naming summary

| Suffix | Meaning |
|---|---|
| *(none)* | Nodal primary field — `displ_x`, `velo_y`, `acce_z` |
| `_1`, `_2`, … | Element value at that quadrature point |
| `_n` | Nodal value recovered by L2 projection |
