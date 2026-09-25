# HW3L

**A Three-Field Hu–Washizu Tetrahedron in Logarithmic Strain.** Research note.

Status: **proposal; the claim is stated as a prediction about the linearized
plastic operator and is supported by `prototype/plastic.jl`.** Nothing is
implemented in Carina. Norma carries an element-level prototype
(`src/three_field.jl`) of the two unenriched pairs this note rules out.

## Building

```
make          # -> note.pdf   (24 pages)
make watch    # rebuild continuously on save
make clean    # remove auxiliary files, keep the PDF
make purge    # remove auxiliary files and the PDF
```

Requires `pdflatex`, `bibtex`, `latexmk`, and the LaTeX packages `boldtensors`,
`booktabs`, `microtype`, `natbib`. On Fedora:

```
sudo dnf install texlive-boldtensors texlive-booktabs texlive-microtype \
                 texlive-natbib latexmk
```

The PDF is checked in so the document can be read without a TeX
installation. It is a build product of the source beside it: rebuild it with
`make` and commit it together with any change to the source.

## The claim in one paragraph

Volumetric locking and spurious soft modes are failures of the two hypotheses
of Brezzi's theorem. The inf-sup condition is a property of the pair and is
settled: the three-dimensional Crouzeix–Raviart pair, $P_2$ with face and
interior bubbles over a linear discontinuous pressure, is stable at
$\beta_h = 0.2968$, and every other element-local pair decays. Coercivity of
the deviatoric form on the discretely isochoric subspace holds in elasticity
for every pair, and an identity bounds its constant below by $(1-\beta)/2$
under a plastic tangent for every pair as well, so that constant cannot
distinguish pairs. What does is whether the soft modes are isochoric. Under a
plastic tangent with a uniaxial flow direction, the softest modes of a
constant-pressure element carry a dilatation of 0.20–0.27 that does not decay
under refinement and that the continuum would resist with $\kappa$; those of
the stable pair carry one that decays as $h^2$; those of Taylor–Hood are
dilatation almost entirely. No stabilization is needed where the spurious
modes are absent. The composite tetrahedron relieves locking with a constant
pressure and pays for the soft modes with a penalty; variational-multiscale
methods add a time-step-dependent subgrid term that reaches the plastic return
map. Neither is needed with a pair inside the admissible window.

## The formulation

Three fields — motion, volumetric log-strain $\bar\theta$, pressure $\bar p$ —
in a Hu–Washizu functional over Hencky strain. Because $\theta = \log J$
exactly, the field that must be treated mixedly is a *scalar* in closed form,
orthogonal to the deviatoric measure that carries the plasticity.

Properties that motivated the choices:

- **No stabilization exists to tune.** Nothing of subgrid or penalty type
  appears anywhere, so nothing can contaminate the deviator.
- **The constitutive law is untouched.** All finite-deformation content lives in
  two material-*independent* geometric transforms; between them sits an
  unmodified small-strain algorithm. Any small-strain model ports without
  reformulation.
- **Strain and stress never cross discretizations.** Both are evaluated at the
  same quadrature point of the same element.
- **Elimination is element-local.** Both auxiliary fields are discontinuous, so
  the tangent keeps displacement-mesh sparsity — the property that makes the
  composite tetrahedron affordable.

## What would kill it

The note ends with five falsifiable claims and the benchmark set. The ones most
likely to fail:

- The recommended pair needs **bubble enrichment** — plain $P_2/P_1^{\rm disc}$
  is *not* inf-sup stable on tetrahedra, only in 2D. Taylor–Hood
  $P_2/P_1$-continuous is a fallback for elasticity only: under a plastic
  tangent its softest modes are almost pure dilatation, and local
  elimination is lost with it.
- **Explicit dynamics is the weakest part.** HRZ lumping gives strictly
  positive masses for $P_2$ (row-sum does not, and cannot be used), but the
  critical time step is smaller than for linear elements. Whether fewer, larger
  elements repay that is unmeasured.
- Nothing here is novel in its parts. The contribution, if any, is that
  combining them correctly makes the stabilization unnecessary — worth exactly
  what the numerical evidence turns out to be worth.

## Relationship to the rest of the repository

Independent of `../sec5l/`. The two are separate attacks on the same problem and
share no machinery.
