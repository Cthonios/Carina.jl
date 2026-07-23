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

## Errors from this section

All of these abort the run at startup:

| Message | Cause |
|---|---|
| `Missing [model] section in input.` | No `model` key. |
| `Unknown model.type = "X".` | `type` names a physics that does not exist. |
| `Missing [model.material] section in input.` | No `material` under `model`. |
| `Missing [model.material.blocks] mapping.` | No `blocks` under `material`. |
| `[model.material.blocks] is empty; ...` | `blocks` present but with no entries. |
| `[model.material.blocks] lists N blocks, but Carina supports a single material per simulation.` | More than one block assigned — see [Materials](materials.md). |
| `Material model "X" is assigned to block "Y" ... but [model.material] has no "X" property dict.` | `blocks` names a material with no matching property dictionary. |
| `[model.material.blocks] refers to element block "X", which is not in the mesh.` | Block name does not match the mesh. |
| `Unknown material model "X". Supported: ...` | Material name not recognised. |

The property-dict one is the common mistake. A `blocks` entry such as
`my_block: neohookean` requires a sibling key `neohookean:` holding the
properties — the name in `blocks` is a *reference*, not a definition. The error
lists the property dicts you did write, which usually makes the mismatch obvious.

The block *name* on the left of that entry (`my_block`) is checked against the
element blocks in the mesh file. It is only used for the startup log line — the
material is applied to the whole mesh either way — so a mistyped block name used
to produce a correct-looking run with a wrong label.
