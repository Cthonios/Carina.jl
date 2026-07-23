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
    entire mesh: one `SolidMechanics(cm, density)` physics object covers the
    whole domain. `blocks` is a mapping for forward compatibility, but it must
    hold exactly one entry. Listing more is an error:

    ```
    ERROR: [model.material.blocks] lists 2 blocks (matrix, inclusion), but Carina
           supports a single material per simulation. The material is applied to
           the whole mesh; per-block materials are not implemented.
    ```

    Carina releases before this behaviour took `first(blocks_dict)` and applied
    it everywhere. Since `blocks` is an unordered `Dict`, *which* material won
    was hash order rather than file order — so a two-block input ran with an
    arbitrary one of the two, silently, on both blocks.

    If your mesh has multiple element blocks, that is fine; give them all the
    same material by naming any one of them in `blocks`. The block you name is
    checked against the mesh, but the material applies to every block.

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
| `density` | dynamic runs only | `0.0` | Mass density (kg/m³). |

`density` is read from the material dictionary and is **not** passed through
the elastic-constant alias table. If it is missing or zero, Carina emits a
warning and continues with `0.0`:

```
[WARNING] No density specified for material "neohookean"; using 0.0.
```

That is legal for `quasi static`, which never forms a mass matrix. For
`newmark` and `central difference` it is a hard error, because a zero density
makes the lumped mass matrix singular — every acceleration would be `0/0`, and
the run would fill the output file with `NaN` rather than report the missing
property:

```
ERROR: The material assigned to block "cube" has density 0.0, but a dynamic
       time integrator requires a mass matrix. Set `density` in the
       [model.material] property dict.
```

## Unknown property keys are ignored

Any key in the material dictionary that is neither `density` nor a recognised
alias is dropped with a warning rather than an error:

```
[WARNING] Unknown material property key "youngs_modulus"; ignoring.
```

This is easy to miss in a long log. If a material behaves as though a property
were never set, check for this warning first — note the example above uses
underscores, which are *not* an accepted alias (Carina's keys use spaces).

The one case that cannot be missed is `density`, since misspelling it leaves the
density at `0.0` and a dynamic run then aborts outright.

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
