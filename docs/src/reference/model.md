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
| `type` | no | Physics type. Conventionally `solid mechanics`. |
| `material` | **yes** | Material assignment and properties — see [Materials](materials.md). |

## `type` is not yet dispatched on

Every example writes `type: solid mechanics`, and you should keep writing it —
but be aware that the value is currently **parsed and ignored**. Carina builds
a solid-mechanics physics object unconditionally; there is no branch on this
key anywhere in the source, and no validation of it either. Any string, or no
`type` key at all, produces the same solid-mechanics run.

Heat conduction and true multiphysics are design goals, not present
capabilities. When they arrive, `type` is where they will be selected, which is
why the key is worth writing today.

Because the `model` section is not key-validated, misspelled keys under it are
silently ignored rather than warned about — including `materials:` instead of
`material:`, which will fail with the missing-section error below.

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
| `Missing [model.material] section in input.` | No `material` under `model`. |
| `Missing [model.material.blocks] mapping.` | No `blocks` under `material`. |
| `Material block "X" listed in blocks but no property dict found.` | `blocks` names a material with no matching property dictionary. |
| `Unknown material model "X". Supported: ...` | Material name not recognised. |

The third and fourth are the common ones. A `blocks` entry such as
`my_block: neohookean` requires a sibling key `neohookean:` holding the
properties — the name in `blocks` is a *reference*, not a definition.
