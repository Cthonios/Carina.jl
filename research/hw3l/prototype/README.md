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

## `infsup.jl` — the pair, settled by counting

Whether the pressure pair is inf-sup stable is a property of a mesh *sequence*
and cannot be seen on one element. This runs the Chapelle–Bathe numerical
inf-sup test over a refinement sequence — but the decisive part turned out to
need no eigenvalue at all.

The Schur complement `G K⁻¹ G'` is `n_p × n_p` with rank at most `n_u`. If
`n_p > n_u` it is singular by counting, and no inf-sup constant exists.

**Result**, on a Freudenthal-subdivided unit cube at `N = 12`:

| pair | `n_u` | `n_p` | `n_p/n_u` |
|---|---|---|---|
| P2 / P0 | 36,501 | 10,368 | 0.284 |
| **P2 / P1disc** | 36,501 | 41,472 | **1.136** |
| P2 ⊕ interior bubble / P1disc | 67,605 | 41,472 | 0.613 |
| P2 ⊕ interior ⊕ face bubbles / P1disc | 127,221 | 41,472 | 0.326 |

`P2/P1disc` exceeds one at *every* refinement, and the asymptotics are exact:
the quadratic nodes of this mesh family fill the `(2N+1)³` grid, so
`n_u → 24N³` while `n_p = 4·6N³ = 24N³`. The ratio approaches one from above and
never crosses. **The unenriched pair is not inf-sup stable, and refinement does
not rescue it** — so the bubble enrichment is mandatory, not advisory.

The count is *necessary and not sufficient*, and `beta.jl` shows two pairs that
clear it and fail anyway. Read this script as a cheap refutation tool, never as
a certificate.

## `beta.jl` — the discrete inf-sup constant over a mesh sequence

`beta_h` is computed as the smallest nonzero **singular value** of
`W = L⁻¹ G' M^(-1/2)`, never forming `G K⁻¹ G'`, so the condition number of the
small quantity being measured is not squared. The null dimension is fixed
independently as `n_p - rank(G)`, so the cut is taken by index rather than by
guessing a threshold. Taylor–Hood is carried as a **positive control**: a test
that reports every pair unstable cannot be distinguished from a broken test by
a negative control alone.

**Result**, all-Dirichlet, fitted rate over the last four points:

| pair | β at `h`=1/2 | 1/4 | 1/6 | rate | verdict |
|---|---|---|---|---|---|
| P2 / P1 continuous (control) | 0.1734 | 0.2186 | 0.2214 | −0.009 | stable |
| P2 / P0 | 0.1001 | 0.0755 | 0.0544 | 0.88 | decays |
| P2 / P1disc | 0.1132 | 0.0424 | 0.0277 | 1.05 | decays |
| P2 ⊕ interior / P1disc | 0.0782 | 0.0564 | 0.0406 | 0.77 | decays |
| **P2 ⊕ interior ⊕ face / P1disc** | 0.2875 | 0.2962 | **0.2968** | **−0.011** | **stable** |

**The three-dimensional Crouzeix–Raviart pair is stable**, at a constant above
the control's, with a one-dimensional pressure null space (the hydrostatic mode
and nothing else) at every mesh. Everything else decays.

The interior bubble alone is **not** enough. It vanishes on the whole element
boundary, so `∫ div b_int = 0` and its columns of `G` against a constant
pressure are identically zero — it reaches only the *linear* pressure modes.
The face bubbles are what supply a displacement response to the pressure jump
across a face. The sweep carries `P2 ⊕ interior / P0` as a check on that
reasoning: it reproduces plain `P2/P0` to four digits, as it must if the
enrichment is wired into `G` and not merely into `K`.

## `locking.jl` — assembled soft modes and locking

Both failure modes are rank statements about the same `G`: too small a rank
leaves volumetric directions unconstrained (soft modes), too large a rank
destroys the isochoric subspace (locking). The isochoric fraction
`(n_u - rank G)/n_u` is read against the bound `1 - n_p/n_u`, which it cannot
fall below; a pair that *attains* the bound spends every pressure unknown on a
distinct displacement direction.

**Results.** The three spurious modes per element found by `softmode.jl` do
**not** survive assembly — zero, under both boundary conditions, at every mesh.
`P2/P1disc` retains 3–7% of its deformation modes with the boundary fixed
against 58–63% for `P0`: it is the unenriched pair that locks, and that, rather
than the soft-mode argument, is the case for enrichment. The full enrichment
reaches 61–63% and attains the bound.

An earlier version of this script *extrapolated* the enriched pair by adding
`3·nelem` to `n_u` and carrying `rank(G)` over unchanged. That is right for
`P0` and wrong for `P1disc`, where the bubble raises `rank(G)` from 957 to 1532
at `N = 4`; the extrapolation overstated the isochoric fraction by a factor of
two. Both enriched pairs are now assembled.

## Quadrature

`ReferenceFiniteElements` supplies tetrahedron rules only to degree 3. The
quartic interior bubble has a cubic gradient, so its block of `Kdev` and `Kh1`
is degree 6, and under-integrating it would soften exactly the modes these
scripts measure — silently, and in a way the Taylor–Hood control cannot catch,
having no bubble to under-integrate. `common.jl` therefore builds a
conical-product (Duffy) rule of arbitrary degree from Gauss–Jacobi factors, and
`assemble_all` refuses an insufficient `q_degree` rather than accepting it.

Shape functions still come from `ReferenceFiniteElements`, evaluated at
arbitrary points rather than at its own quadrature points, so there is no
second implementation to disagree with the first. Swapping the quadrature
reproduced every previously published number in this directory exactly.
