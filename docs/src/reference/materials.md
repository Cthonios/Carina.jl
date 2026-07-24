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

Meshes with more than one element block assign a material to each; see
[One material per element block](#One-material-per-element-block) below.

## One material per element block

Each element block gets its own constitutive model and density. Name every block
in `blocks`, and give each material a property dictionary:

```yaml
model:
  type: solid mechanics
  material:
    blocks:
      lower: stiff
      upper: soft
    stiff:
      model: neohookean
      elastic modulus: 1.0e11
      Poisson's ratio: 0.25
      density: 1000.0
    soft:
      model: neohookean
      elastic modulus: 1.0e9
      Poisson's ratio: 0.25
      density: 1000.0
```

The value in `blocks` is a **material label**, and the dictionary it names
carries a `model` key selecting the constitutive model. The label is arbitrary,
which is what lets two blocks share a model with different properties — as
`stiff` and `soft` do above.

If you omit `model`, the label itself is read as the model name. That is the
older spelling and still works:

```yaml
    blocks:
      my_block: neohookean
    neohookean:
      elastic modulus: 10.0e9
      Poisson's ratio: 0.25
      density: 1000.0
```

!!! warning "Every block must be assigned"
    There is no default material. A mesh block missing from `blocks` is an
    error, not an inherited material:

    ```
    ERROR: [model.material.blocks] does not assign a material to "upper". Every
           element block in the mesh needs a material; there is no default.
           Mesh blocks: lower, upper.
    ```

    Assigning a material to a block that is not in the mesh is likewise an
    error, with a spelling suggestion:

    ```
    ERROR: [model.material.blocks] assigns a material to "uppr" (did you mean
           "upper"?), which is not an element block in the mesh.
           Mesh blocks: lower, upper.
    ```

    Both directions are checked because materials are matched to blocks by
    **name**, and a block that silently inherited some other block's material
    would produce a converged, plausible, wrong answer. Carina releases before
    per-block support took `first(blocks_dict)` and applied it everywhere —
    since `blocks` is an unordered `Dict`, *which* material won was hash order
    rather than file order.

    A property dictionary that no block references is reported as an unknown
    key in `model.material`, which usually means a block was renamed and its
    old material left behind.

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
the elastic-constant alias table; it is forwarded to ConstitutiveModels under
its own name, which requires every model to carry density as the **first**
entry of its property vector. That single copy is what the mass matrix and the
explicit stable-time-step estimate both read — the physics object does not
store a second one. If `density` is missing or zero, Carina emits a warning and
continues with `0.0`:

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

## Mixed materials and internal-variable output

Blocks may have different numbers of internal state variables — an elastic block
next to a J2 plasticity block is fine. Per-block **element** output of internal
variables works in that case, because Exodus element variables are written per
block; the declared set of names is the union across blocks.

**Nodal recovery** of internal variables is different. It averages a
quadrature-point quantity onto nodes, and a node on a block interface belongs to
both blocks. If the blocks disagree about what a given state variable *is* —
`eqps` in the plastic block, nonexistent in the elastic one — the projection has
no meaning, so Carina refuses rather than averaging over one side:

```
ERROR: Nodal recovery of internal variables requires every element block to have
       the same state variables, but they differ: "lower" => []; "upper" =>
       [Fp_xx, ..., eqps]. Nodes on a block interface belong to both blocks, so
       there is no meaningful value to project there. Set `output.recovery:
       none`, or turn off `output.internal variables`; per-block element output
       of internal variables is unaffected.
```

Stress and deformation-gradient recovery are unaffected — those exist for every
material.

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
