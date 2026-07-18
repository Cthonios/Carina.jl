# Materials

Materials are declared under `model.material`. The section has two parts: a
`blocks` mapping that assigns a material name to each element block, and one
sub-dictionary per material name holding its properties.

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

The key `neohookean` in `blocks` must match a sibling key holding the property
dictionary. The material name is matched after `lowercase(strip(...))`, so
capitalisation and surrounding whitespace do not matter.

!!! warning "One material per simulation"
    Carina currently applies a **single** constitutive model and density to the
    entire mesh. The parser reads only the first entry of `blocks`
    (`first(blocks_dict)` in `_parse_material_section`) and builds one
    `SolidMechanics(cm, density)` physics object for the whole domain.

    Listing two blocks with different materials does **not** give you two
    materials — and because `blocks` is an unordered `Dict`, *which* of them
    wins is hash order, not file order. Do not rely on it. If your mesh has
    multiple blocks, give them all the same material.

## Material models

| YAML name | Aliases | Required constants | Notes |
|---|---|---|---|
| `neohookean` | `neo-hookean`, `neo hookean` | E, ν | Finite deformation. Coincides with linear elastic at small strain. General-purpose default. |
| `linear elastic` | `linearelastic` | E, ν | Small-strain only; invalid at large deformation. Enables stiffness/factorization caching (see [Solvers](solvers.md)). |
| `hencky` | — | E, ν | Logarithmic (Hencky) strain measure. Good for moderate strains. |
| `saint venant kirchhoff` | `saintvenant-kirchhoff`, `saintvenantkirchhoff`, `svk` | E, ν | Green–Lagrange strain. Unstable in strong compression. |
| `seth-hill` | `seth hill`, `sethhill` | E, ν, `m`, `n` | Generalized strain family parameterized by exponents `m`, `n`. |
| `linear elasto plasticity` | — | E, ν, `yield stress`, `hardening modulus` | Small-strain J2 plasticity, von Mises yield surface with linear isotropic hardening. |
| `j2 plasticity` | `finitedefj2plasticity`, `finite def j2 plasticity` | E, ν, `yield stress`, `hardening modulus` | Finite-deformation J2 plasticity. Path-dependent — use small time steps and keep line search on. |

## Elastic constants

Any two independent constants define an isotropic elastic material. Each YAML
key below is an alias for the canonical name that
[ConstitutiveModels.jl](https://github.com/Cthonios/ConstitutiveModels.jl)
expects; keys are matched case-insensitively.

| YAML key | Aliases | Canonical name |
|---|---|---|
| `elastic modulus` | `Young's modulus`, `youngs modulus` | `Young's modulus` |
| `Poisson's ratio` | `poissons ratio` | `Poisson's ratio` |
| `bulk modulus` | — | `bulk modulus` |
| `shear modulus` | — | `shear modulus` |
| `Lame's first constant` | `lames first constant`, `Lamé's first constant` | `Lamé's first constant` |

## Plasticity constants

| YAML key | Applies to | Meaning |
|---|---|---|
| `yield stress` | `linear elasto plasticity`, `j2 plasticity` | Initial yield stress |
| `hardening modulus` | `linear elasto plasticity`, `j2 plasticity` | Linear isotropic hardening slope |

## Seth–Hill exponents

| YAML key | Applies to | Meaning |
|---|---|---|
| `m` | `seth-hill` | First strain-family exponent |
| `n` | `seth-hill` | Second strain-family exponent |

## Density

| Key | Required | Default | Notes |
|---|---|---|---|
| `density` | effectively yes | `0.0` | Mass density (kg/m³). |

`density` is read from the material dictionary and is **not** passed through
the elastic-constant alias table. If it is missing or zero, Carina emits a
warning and continues with `0.0`:

```
[WARNING] No density specified for material "neohookean"; using 0.0.
```

A zero density is only meaningful for quasi-static runs. Both dynamic
integrators need mass: it makes the lumped mass matrix singular and the stable
time-step estimate degenerate.

## Unknown property keys are ignored

Any key in the material dictionary that is neither `density` nor a recognised
alias is dropped with a warning rather than an error:

```
[WARNING] Unknown material property key "youngs_modulus"; ignoring.
```

This is easy to miss in a long log. If a material behaves as though a property
were never set, check for this warning first — note the example above uses
underscores, which are *not* an accepted alias (Carina's keys use spaces).

## Example

```yaml
model:
  type: solid mechanics
  material:
    blocks:
      specimen: j2 plasticity
    j2 plasticity:
      elastic modulus: 70.0e9
      Poisson's ratio: 0.36
      density: 2700.0
      yield stress: 250.0e6
      hardening modulus: 0.7e9
```
