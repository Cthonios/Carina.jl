# Model

The required top-level `model` section declares the physics and its material.

```yaml
model:
  type: solid mechanics
  material:
    blocks:
      my_block: neohookean
    neohookean:
      elastic modulus: 10.0e9
      Poisson's ratio: 0.25
      density: 1000.0
```

| Key | Required | Description |
|---|---|---|
| `type` | no | Physics type. `solid mechanics` (aliases `solidmechanics`, `mechanics`). |
| `material` | **yes** | Material assignment and properties — see [Materials](materials.md). |
| `volumetric projection` | no | `constant` or `linear`: the mean-dilatation formulation, see below. Absent: the pointwise formulation. |
| `volumetric strain` | no | `log J` or `J - 1`: the volumetric strain θ(J) that is projected, see below. Requires `volumetric projection`. |
| `volumetric form` | no | `split` or `general`: the form of the projected element, see below. Absent: `split` for a material with the volumetric-isochoric split, `general` otherwise. Requires `volumetric projection`. |

## `type` is checked but not yet dispatched on

Every example writes `type: solid mechanics`, and you should keep writing it.
Carina builds a solid-mechanics physics object unconditionally — there is no
branch on this key anywhere in the source — so omitting `type` produces the same
run as writing it.

What the key does do is *constrain* you to a physics that exists. Any other
value aborts:

```
Unknown model.type = "thermal". Supported: "solid mechanics".
Thermal and coupled physics are not implemented.
```

That matters more than it looks. Heat conduction and true multiphysics are
design goals, not present capabilities; without this check, an input file
written in anticipation of them would run as solid mechanics and report nothing.

Keys under `model` are validated, so `materials:` for `material:` warns before
it reaches the missing-section error below.

## `material`

The `material` sub-section is required and must contain a `blocks` mapping. See
[Materials](materials.md) for the full list of models, property keys, and
aliases — including the important limitation that Carina currently applies a
**single** material to the whole mesh.

## `volumetric projection`

With this key the volumetric strain θ(J) is replaced, element by element, by
its L² projection onto polynomials of degree 0 (`constant`) or 1 (`linear`)
in the reference coordinates.  This is the mean-dilatation formulation of
`research/tet15-p1/note.tex`, the reduction of the three-field functional in
the motion, the volumetric strain and the pressure with the two auxiliary
fields in one discontinuous space.  The formulation is intended for the
TETRA15 element with `linear`; it runs on any element.  The projection
couples the quadrature points of an element, so the kernels are assembled by
element; the assembled matrix keeps the sparsity of the displacement mesh.

The element has two forms, selected by `volumetric form`:

- `split`: for a material with an exact volumetric-isochoric split and a
  quadratic volumetric energy κ/2 θ² (currently `j2 plasticity`, with
  θ = J − 1).  The volumetric energy is evaluated at the projected strain,
  and the isochoric response and the internal variables at the quadrature
  points without projection.
- `general`: for any material.  With J̃ = θ⁻¹(P_h θ(J)), the inverse of θ
  applied to the projection P_h θ(J) of the pointwise θ(J), the material is
  evaluated at the deformation gradient F̃ = (J̃/J)^{1/3} F, whose isochoric
  part is that of F and whose volume ratio is J̃; its internal variables are
  updated there.  The element is the stationarity of the integral of the
  stored energy W(F̃).  The pressure that enters the residual is the
  projection of the material's mean stress at F̃ (note, section "General
  materials").  The element matrix and the diagonal kernels of the
  preconditioners are assembled in closed form, with the material's own
  tangent at F̃ evaluated once per quadrature point; the matrix-free action
  is the forward-mode derivative of the element residual along the vector.

When `volumetric form` is absent, the split form is used for a material
with the split and the general form otherwise.  `general` may be given for a
material with the split: for `j2 plasticity` with `volumetric strain: J - 1`
the two forms give the same energy, residual, tangent and internal variables
to rounding.

`volumetric strain` selects θ(J): `log J` or `J - 1` (case and white space
are ignored).  In the general form the default is `log J`.  The two choices
give different elements, which converge to the same solution under
refinement; the energies of one element differ by 5.6% at 20% strain (note,
Remark "The volumetric variable is a modeling choice").  In the split form θ
is the material's own volumetric strain, and the key, if present, must name
it: the projected pressure κ θ̄ of the split form is the stationarity
condition of κ/2 θ² written in the material's θ.

```yaml
model:
  type: solid mechanics
  volumetric projection: linear
  volumetric form: general
  volumetric strain: log J
  material:
    blocks:
      cube: neohookean
    neohookean:
      elastic modulus: 1.0e9
      Poisson's ratio: 0.45
      density: 1000.0
```

## Errors from this section

All of these abort the run at startup:

| Message | Cause |
|---|---|
| `Missing [model] section in input.` | No `model` key. |
| `Unknown model.type = "X".` | `type` names a physics that does not exist. |
| `Unknown model.volumetric projection = "X".` | Not `constant` or `linear`. |
| `Unknown model.volumetric strain = "X".` | Not `log J` or `J - 1`. |
| `Unknown model.volumetric form = "X".` | Not `split` or `general`. |
| `model.volumetric strain requires model.volumetric projection; ...` | `volumetric strain` or `volumetric form` without `volumetric projection`. |
| `model.volumetric projection with the material of block "B": ...` | `volumetric form: split` with a material that has no volumetric-isochoric split, or a `volumetric strain` that differs from the material's in the split form. |
| `Missing [model.material] section in input.` | No `material` under `model`. |
| `Missing [model.material.blocks] mapping.` | No `blocks` under `material`. |
| `[model.material.blocks] is empty; ...` | `blocks` present but with no entries. |
| `[model.material.blocks] lists N blocks, but Carina supports a single material per simulation.` | More than one block assigned — see [Materials](materials.md). |
| `Material model "X" is assigned to block "Y" ... but [model.material] has no "X" property dict.` | `blocks` names a material with no matching property dictionary. |
| `[model.material.blocks] refers to element block "X", which is not in the mesh.` | Block name does not match the mesh. |
| `Unknown material model "X". Supported: ...` | Material name not recognized. |

The property-dict one is the common mistake. A `blocks` entry such as
`my_block: neohookean` requires a sibling key `neohookean:` holding the
properties — the name in `blocks` is a *reference*, not a definition. The error
lists the property dicts you did write, which usually makes the mismatch obvious.

The block *name* on the left of that entry (`my_block`) is checked against the
element blocks in the mesh file. It is only used for the startup log line — the
material is applied to the whole mesh either way — so a mistyped block name used
to produce a correct-looking run with a wrong label.
