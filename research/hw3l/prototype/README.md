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
| P2 ⊕ bubble / P1disc | 67,605 | 41,472 | 0.613 |

`P2/P1disc` exceeds one at *every* refinement, and the asymptotics are exact:
the quadratic nodes of this mesh family fill the `(2N+1)³` grid, so
`n_u → 24N³` while `n_p = 4·6N³ = 24N³`. The ratio approaches one from above and
never crosses. **The unenriched pair is not inf-sup stable, and refinement does
not rescue it** — so the bubble enrichment is mandatory, not advisory.

Taken with `softmode.jl`, the two experiments bracket the design from opposite
sides: P0 is safe on the pressure side and leaves three spurious zero-energy
displacement modes per element; P1disc fixes the displacement side exactly and
fails the pressure count. Only the enriched pair satisfies both.

The script also reports the eigenvalue sweep, with `P1/P0` as a control that the
test detects a pair known to be unstable. Those numbers carry less weight: the
meshes a dense generalized eigensolver can reach are coarse, and for
`P2/P1disc` the spectrum is dominated by the rank deficiency the count already
explains.
