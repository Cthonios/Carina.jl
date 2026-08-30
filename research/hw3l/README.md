# HW3L

**A Three-Field Hu–Washizu Tetrahedron in Logarithmic Strain.** Research note.

Status: **proposal, nothing implemented.** The note argues a position and lists
the experiments that would confirm or kill it.

## Building

```
make          # -> note.pdf   (8 pages)
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

The PDF is a build product and is not tracked.

## The claim in one paragraph

Volumetric locking and spurious soft modes are two failure directions of a
single inf-sup condition, not two problems needing two remedies. The composite
tetrahedron adopts an element-constant pressure to relieve locking and pays for
it with soft modes that need a penalty; variational-multiscale methods add a
subgrid term whose coefficient is tuned and whose magnitude scales with the
time step, and which reaches the deviator and therefore the plastic return map.
Choosing a displacement–pressure pair that is inf-sup stable in three
dimensions removes both failures with no stabilization of any kind.

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
  is *not* inf-sup stable on tetrahedra, only in 2D. If the bubbles prove
  awkward, Taylor–Hood $P_2/P_1$-continuous is the fallback, and local
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
