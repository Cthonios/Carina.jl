# Quadrature

The optional top-level `quadrature` section overrides the element integration
rule. If omitted, Carina uses Gauss–Legendre with degree 2.

```yaml
quadrature:
  type: gauss legendre
  order: 2
```

| Key | Required | Default | Description |
|---|---|---|---|
| `type` | no | `gauss legendre` | Quadrature family. |
| `order` | no | `2` | Quadrature degree passed to the function space as `q_degree`. |

## Accepted `type` values

| Value | Aliases | Rule |
|---|---|---|
| `gauss legendre` | `gl` | Gauss–Legendre (interior points) |
| `gauss lobatto legendre` | `gll` | Gauss–Lobatto–Legendre (includes endpoints) |

The value is lowercased before matching, so `Gauss Legendre` and
`GAUSS LEGENDRE` both work. Any other value is a hard error:

```
Unknown quadrature.type = "simpson". Supported: "gauss legendre", "gauss lobatto legendre".
```

An unrecognised quadrature type stops the run rather than warning — the
integration rule is too consequential to guess at.

The section's *keys* are validated too, so a misspelled `order` warns and falls
back to the default rather than passing unnoticed:

```
[WARNING] Unknown key "orders" in quadrature. Did you mean "order"?
```

## Effect on output

The quadrature rule sets the number of integration points per element, which in
turn determines how many per-quadrature-point element variables are written.
Raising `order` increases the `_1`, `_2`, … suffix range on stress,
deformation-gradient, and internal-variable output. See
[Output fields](output.md).
