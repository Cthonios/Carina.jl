# HW3L

**A Three-Field Hu–Washizu Tetrahedron.** Research
note (`note.tex`, `note.pdf`).

## What the note contains

The note defines and documents a tetrahedral finite element for the
large-deformation plasticity of nearly incompressible materials, in the form
an implementation in another code needs:

- the continuum formulation: a three-field Hu–Washizu functional in the
  motion, a scalar volumetric strain θ(J) and a pressure, and its exact
  reduction to a displacement functional in which θ is replaced element by
  element by its L² projection (mean dilatation); the requirements on the
  material (an exact volumetric–isochoric split with a volumetric energy
  quadratic in θ); the two instances θ = log J (Hencky) and θ = J − 1
  (Simo–Hughes J2, the instance implemented);
- a self-contained account of volumetric locking, the mixed problem, the
  inf-sup condition, Brezzi's theorem and the construction by which bubble
  functions satisfy the inf-sup condition;
- the element: the three-dimensional Crouzeix–Raviart pair, quadratic
  displacement with four cubic face bubbles and one quartic interior bubble
  over a linear discontinuous pressure, carried by a nodal basis on fifteen
  nodes (TETRA15, Exodus node order), with the nodal transform, the
  fourteen-point quadrature rule of Keast with its points and weights, and
  the row-sum lumped mass;
- the element algorithm: the two passes over the quadrature points, the
  residual, the tangent with its coupling term, what the material must
  supply, and the consistency checks an implementation must pass;
- the measurements on the linearized operator: inf-sup constants over a
  mesh sequence for seven pairs against the Taylor–Hood control, locking
  counts, stability on deformed configurations, the dilatation of the soft
  modes under a plastic tangent (the composite tetrahedron of Albany-LCM
  included), quadrature and basis;
- the nonlinear results on Cook's membrane in three dimensions, elastic at
  ν = 0.4999 and elastoplastic, on three meshes, against the composite
  tetrahedron of Albany-LCM: tip displacements and pressure fields;
- the open measurements and the limitations.

## Implementation

The formulation is implemented in Carina (`src/projected_physics.jl`, input
key `model.volumetric projection: linear | constant`), on the TETRA15
element of ReferenceFiniteElements (`Tet{EnrichedLagrange, 2}`;
`bin/tetra15` converts TETRA4 and TETRA10 meshes), with the Simo–Hughes J2
model of ConstitutiveModels through its volumetric–isochoric split
interface, and with the element-level assembly of FiniteElementContainers
(`assembly_granularity`). `test/projected-element.jl` holds the consistency
checks. The Cook benchmark is `benchmark/hw3l/cook/`.

## Prototype scripts

`prototype/` holds the Julia scripts that produce the measurements on the
linearized operator; `prototype/README.md` documents each script and its
output.

## Building the note

```
make          # -> note.pdf
make watch    # rebuild continuously on save
make clean    # remove auxiliary files, keep the PDF
make purge    # remove auxiliary files and the PDF
```

Requires `pdflatex`, `bibtex`, `latexmk`, and the LaTeX packages
`boldtensors`, `booktabs`, `graphicx`, `microtype`, `natbib`. On Fedora:

```
sudo dnf install texlive-boldtensors texlive-booktabs texlive-microtype \
                 texlive-natbib latexmk
```

The PDF is tracked so that the document can be read without a TeX
installation. It is a build product of the source beside it: rebuild it with
`make` and commit it together with any change to the source. The figures
of the Cook benchmark are in `figures/`.

## Relationship to the rest of the repository

`../sec5l/` is an independent formulation of the same problem; the two share
no code.
