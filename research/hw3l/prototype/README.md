# HW3L prototype

Numerical experiments for the formulation in `../note.tex`. Each script is
standalone and prints a verdict.

Run from the repository root, which has the dependencies:

```
julia --project=. research/hw3l/prototype/softmode.jl
```

## `softmode.jl` — spurious-mode census

Tests the central claim about soft modes on a single element, where it reduces
to a rank count and needs no mesh, no material nonlinearity and no solver.

For linear response the three fields condense exactly to

```
K_e = K_dev + kappa * G' * inv(M) * G
```

`K_dev` annihilates every mode with zero deviatoric strain at quadrature — the
six rigid-body modes, plus the volumetric directions the displacement space can
produce. `K_vol` has rank equal to the volumetric directions the pressure space
can see. Whatever is left over has no deviatoric energy and no volumetric
energy, and is exactly zero-energy:

```
spurious = dim null(K_dev) - 6 - rank(K_vol)
```

**Result.** For the P2 tetrahedron, `null(K_dev) = 10` = 6 rigid + 4 volumetric.
A constant pressure constrains one of the four and leaves **three spurious
zero-energy modes**; a P1-discontinuous pressure constrains all four and leaves
none. The prediction matches the measured spectrum in every case tested, at
three levels of element distortion. Enriching further (P2-discontinuous
pressure) changes nothing, because the rank saturates at four.

This settles the mechanism. It does **not** settle inf-sup stability, which is a
global property of a mesh sequence and cannot be seen on one element.
