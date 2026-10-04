# TET15-P1 prototype

Numerical experiments for the formulation in `../note.tex`. Each script is
standalone and prints a verdict.

Run from the repository root, which has the dependencies:

```
julia --project=. research/tet15-p1/prototype/softmode.jl
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

The census now integrates with the 27-point conical rule: the
quadratic-discontinuous pressure row has a 10 × 10 mass matrix that the
original 4-point rule left singular (the count happened to come out right);
the distortions are 0, 0.03 and 0.06, since 0.12 folds the element.

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
clear it and fail anyway (P2/P0 at 0.284, P2 ⊕ interior/P1disc at 0.613). Read this script as a refutation test that needs no factorization, never as
a certificate.

## `beta.jl` — the discrete inf-sup constant over a mesh sequence

`beta_h` is computed as the smallest nonzero **singular value** of
`W = L⁻¹ G' M^(-1/2)`, never forming `G K⁻¹ G'`, so the condition number of the
small quantity being measured is not squared. The null dimension is fixed
independently as `n_p - rank(G)`, so the cut is taken by index rather than by
guessing a threshold. Taylor–Hood is carried as a **positive control**: a test
that reports every pair unstable cannot be distinguished from a broken test by
a negative control alone.

**Result**, all-Dirichlet, in the order the Boffi–Brezzi–Fortin construction
runs (`null` = `n_p - rank G`, which must be 1; `local rate` at the last step):

| pair | β at `h`=1/2 | 1/4 | 1/6 | 1/8 | null | local rate | verdict |
|---|---|---|---|---|---|---|---|
| P2 / P1 continuous (control) | 0.1734 | 0.2186 | 0.2214 | 0.2216 | 1 | −0.003 | stable |
| P2 / P0 | 0.1001 | 0.0755 | 0.0544 | 0.0418 | **4** | 0.97 | decays |
| P2 ⊕ interior / P0 | 0.1015 | 0.0756 | 0.0544 | — | 4 | 0.84 | decays |
| P2 ⊕ face / P0 | 0.4671 | 0.4228 | 0.4060 | 0.3967 | 1 | 0.076 | consistent with stable |
| P2 / P1disc | 0.1132 | 0.0424 | 0.0277 | — | n_p−n_u | 1.04 | decays |
| P2 ⊕ interior / P1disc | 0.0782 | 0.0564 | 0.0406 | — | 4 | 0.83 | decays |
| P2 ⊕ face / P1disc | 0.0938 | 0.0701 | 0.0497 | — | 2 | 0.88 | decays |
| **P2 ⊕ interior ⊕ face / P1disc** | 0.2875 | 0.2962 | **0.2968** | — | 1 | **−0.003** | **stable** |

Every row is what Boffi–Brezzi–Fortin (2013, §8.7 and Example 8.7.2) predict;
the sweep validates the bench and supplies the constants. P2 does not control
a constant pressure in 3D — it has no face degrees of freedom — and carries
three exact spurious pressure modes at every refinement. The interior bubble
cannot help: it vanishes on the element boundary, so `∫ div b_int = 0` and its
columns of `G` against a constant pressure are identically zero (checked in
`checks.jl`). Face bubbles drop the null dimension to one and give a sequence
approaching a limit near 0.35 from above — five to seven points cannot separate
that from a slow decay, so the row is reported as consistent with stable and
its stability rests on the theory. Over P1disc the face bubbles alone leave one
spurious linear mode; the interior bubble removes it. Both together are the 3D
Crouzeix–Raviart pair, flat at 0.2968 with a one-dimensional null space.

## `locking.jl` — assembled soft modes and locking

Both failure modes are rank statements about the same `G`: too small a rank
leaves volumetric directions unconstrained (soft modes), too large a rank
destroys the isochoric subspace (locking). The isochoric fraction
`(n_u - rank G)/n_u` is read against the bound `1 - n_p/n_u`, which it cannot
fall below; a pair that *attains* the bound spends every pressure unknown on a
distinct displacement direction.

**Results.** The three exact zero-energy modes per element found by
`softmode.jl` do **not** survive assembly — zero, under both boundary
conditions, at every mesh. That is a statement about *exact* zeros only; the
energetically soft modes a constant pressure leaves are a coercivity question
that no linear operator here decides. `P2/P1disc` retains 3–7% of its
deformation modes with the boundary fixed against 58–63% for `P0`: it is the
unenriched pair that locks, and that is the case for enrichment. The full
enrichment reaches 61–63% and attains the bound.

An earlier version of this script *extrapolated* the enriched pair by adding
`3·nelem` to `n_u` and carrying `rank(G)` over unchanged. That is right for
`P0` and wrong for `P1disc`, where the bubble raises `rank(G)` from 957 to 1532
at `N = 4`; the extrapolation overstated the isochoric fraction by a factor of
two. All enriched pairs are now assembled.

## `deformed.jl` — the inf-sup constant on a deformed configuration

Every stability result above is at `F = I`. This assembles the same
operators on the isoparametric image of the P2 mesh under a prescribed map
(twist 90° and 180° about the vertical axis, a parabolic shear, a radial
inflation with `J` up to 2.3, and an affine compression to `J = 0.5` as the
control), with the reference cube's boundary conditions, and computes `β_h`
as `beta.jl` does. `julia deformed.jl extend` adds the CR pair at `N = 6` on
the three non-affine isochoric maps.

**Result.** `β_h` decreases with the local shear of the map for every pair,
most for the CR pair (0.297 → 0.204 at 90°, 0.116 at 180°, 0.183 under the
parabolic shear; Taylor–Hood 0.221 → 0.189, 0.137, 0.176), is unchanged by
the inflation (0.276) and flat under the affine compression (0.213: the
aspect-ratio effect alone). At a fixed deformation the local rate falls with
`N`, consistent with a positive limit. The three exact spurious pressure
modes of `P2/P0` exist only on the affine mesh: under every non-affine map
its null dimension drops to one and `β_h` to 0.004–0.05. Output kept in
`deformed_out.txt`.

## `plastic.jl` — soft modes under a plastic tangent

The consistent J2 tangent `C_ep = 2μ [I_dev − β n⊗n]` removes the shear
stiffness along a flow direction `n` (`β = 1` perfect plasticity). Brezzi's
second hypothesis, coercivity of the deviatoric form on `ker G`, is measured
as the smallest generalized eigenvalue of `Z'K_ep Z` against the `H1`
seminorm `Z'K_h1 Z`, with `Z` an exact basis of `ker G`.

**That constant cannot separate pairs.** For every `v ∈ H¹₀`,
`∫|dev ε|² = ½|∇v|² + ⅙∫(div v)²`, so the constant is at least `(1−β)/2` for
every pair and every kernel, and at `β = 1` the continuum operator admits
shear bands with zero energy, so it decays for every pair, Taylor–Hood included.
Both are what the sweep shows. The first version of this script expected the
constant to decide the question; it was wrong, and the identity says why.

**The dilatation of the soft modes does.** `r_v = ∫(div v)²/|∇v|²` of the
lowest plastic eigenmodes, all-Dirichlet cube, `N = 2..5`, uniaxial flow
direction: 0.20–0.27 without decay for `P2/P0` and `P2 ⊕ face/P0`; 0.040,
0.018, 0.0074, 0.0042 (`h²`) for the Crouzeix–Raviart pair; 0.86–0.96 for
Taylor–Hood, whose kernel is divergence-free only against continuous `P1`.
Under a shear direction every pair's soft modes are bands and decay, the CR
pair's fastest. The elastic lowest mode is `λ = 0.500`, `r_v = 0` on every row,
which the identity requires and which checks the bench. Output kept in
`plastic_out.txt`.

## `plastic_zone.jl` — the same, in a confined zone with a varying flow direction

`n = dev ε(u)/|dev ε(u)|` and `β = 1` where `|dev ε(u)|` exceeds 0.35 of its
peak, from a prescribed field `u`: an indentation of the top face (zone 51%
of the volume) and the 90° twist (97%), with the whole boundary fixed and
with the base alone fixed. Spurious modes are counted among the twenty
lowest (`r_v > 0.1`), because physical shear bands can lie below them and
hide them from the lowest three.

**Result**, counts at `N = 3, 4, 5` with the whole boundary fixed:
`P2/P0` 16, 15, 7 (indentation) and 14, 18, 15 (twist); `P2 ⊕ face/P0`
18, 17, 5 and 18, 13, 8; the CR pair 4, 0, 0 and 1, 4, 0; Taylor–Hood
20, 18, 17 and 20, 20, 20, its softest twist mode growing to `r_v = 0.74`.
With the base alone fixed the CR pair has none at any `N`. Output kept in
`plastic_zone_out.txt`.

## `composite_tet.jl` — the composite tetrahedron of Albany-LCM, same measurement

The element as Albany builds it with `Use Composite Tet 10`: ten nodes,
displacement piecewise linear on twelve subtetrahedra (Intrepid2
`Basis_HGRAD_TET_COMP12_FEM`, value at the centroid the mean of the six
midpoint values), and in the constitutive update the L2 projection of that
piecewise-constant gradient onto linear functions of the element, which is
what `OPERATOR_GRAD` of that basis returns (the script checks the identity to
8e-15). Albany evaluates the response at the five points of the Intrepid2
degree-3 rule, the rule of the Cook runs. Two volumetric variants: (a)
`Weighted Volume Average J` (Kinematics_Def.hpp), J replaced by its element
mean, whose constraint as κ → ∞ is a P0 pressure on the projected
dilatation; (b) J at each point, whose constraint tr ε̃ = 0 at five points is
tr ε̃ ≡ 0, a P1disc pressure. The deviatoric form uses the projected gradient;
the H1 seminorm and r_v use the gradient of the displacement field itself,
because r_v measures the volume change the continuum would charge. With the
quadratic element substituted the harness reproduces `assemble_all` to 2e-15
and the `P2/P0` entry of `plastic_out.txt` (0.208, 0.195, 0.218 at `N = 3`).

**Result**, all-Dirichlet cube, β = 1, axial flow direction:

- (a) volume-averaged J: r_v of the three lowest plastic modes 0.22–0.25 at
  `N = 3, 4, 5`, no decay, 20 of the 20 lowest modes spurious from `N = 4` on;
  in the confined zones at `N = 5`, 17 (indentation) and 19 (twist). The
  projected dilatation of these modes is 0.147: nonzero pointwise, zero in
  the element mean. The element's elastic constant on ker G_h is 0.038 at
  `N = 5` against 0.5 for the Lagrange pairs, because the projection removes
  part of the deviatoric energy of the field.
- (b) pointwise J: ker G_h has dimension 0, 0, 2, 16 at `N = 2..5` (P2/P1disc:
  12 at `N = 3`, 72 at `N = 4`): the element locks, and its few kernel modes
  (r_v 0.03–0.05, none above 0.1) are too few for a count to mean anything.

The volume average removes the locking of (b) and admits the spurious
dilatational modes of a constant pressure, with larger counts than `P2/P0`
(7 and 15 at `N = 5`). Output kept in `composite_tet_out.txt`.

## `quadrature.jl` — which rule the enriched element needs

On one element (reference, distorted with `det J` in [0.55, 1.19], and a
Kuhn tetrahedron), for each rule: the zero-energy modes of the condensed
stiffness (six is correct) and the extreme generalized eigenvalues against a
216-point rule on the complement of the rigid modes (1 for an exact rule).
Then the assembled `β_h` and the uniaxial spurious count with the reduced
rules, through `assemble_all(...; allow_reduced = true)`.

**Result.** RFE's 4- and 5-point rules leave 21 zero modes; the 8-point
conical rule is rank sufficient but softens to 0.03–0.10. Two degree-5 rules
are admissible: Keast's 14-point rule (verified here to integrate every
monomial through degree 5 to 4e-16) over-stiffens the interior bubble's block
by 35%; the 27-point conical rule under-stiffens it by 9–12%. Assembled, both
give `β_h` within 1% of the exact rule, one null mode, no spurious mode, and
the same lowest-mode dilatation. Recommendation: Keast 14. Output in
`quadrature_out.txt`. `element.jl` holds the shared element machinery.

## `basis.jl` — hierarchical bubbles or a nodal basis

Same space, two bases: the bench's hierarchical one and the nodal TETRA15
arrangement (`nodal_transform()` in `common.jl`). Measures condition numbers,
HRZ and row-sum lumping (positivity, momentum of a rigid translation), and
the explicit Courant number `c_p Δt/h` from the element bound, against
TETRA10 and TETRA4 on the same element.

**Result.** The nodal basis conditions the mass 8× better (94 vs 778). HRZ
on the hierarchical basis loses 51% of a translation's momentum unless
normalized over the Lagrange functions only; the nodal basis conserves it.
Courant numbers on the reference element: TETRA4 0.476, TETRA10 HRZ 0.197,
enriched nodal HRZ 0.145, enriched hierarchical Lagrange-HRZ 0.114 (same
ordering on the distorted and Kuhn elements). Recommendation: nodal. Output
in `basis_out.txt`.

## `materials.jl` — what the materials satisfy (needs Norma)

```
julia --project=/path/to/Norma.jl research/tet15-p1/prototype/materials.jl
```

Three measurements behind §2.5 of the note: the split test
(`W(F) − W(F̄)` at fixed `J` across random isochoric parts — machine zero iff
the split is exact), the quadratic fit of the extracted `W_vol(J)` to
`c(J−1)²` and `c(log J)²`, and the difference between projecting `log J` and
projecting `J − 1` on one element for the same Hencky material. Five of seven
materials split; Hencky is quadratic in `log J` and Simo–Hughes in `J − 1`,
each to machine precision and each 18% wrong in the other's variable. Output
kept in `materials_out.txt`.

## `checks.jl` — correctness checks with known targets

Quadrature exact through degree 7 and demonstrably *not* at degree 8; interior
bubble columns of `G` zero against a constant and nonzero against the linear
modes; `∫ div N_i = 0` for every free basis function in all four spaces (the
conformity test that catches a mis-shared or unconstrained face bubble); face
counts and multiplicities; and the refusals. Output kept in `checks_out.txt`.

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

## Voigt metric correction (2026-09-25)

`common.jl` and `softmode.jl` weighted the Voigt shear rows by 2 where the
engineering-shear energy metric is ½, overstating shear stiffness by four.
No kernel, rank, count or inf-sup constant depends on the metric; the
`lam_min/dev` column of `locking.jl` and the `1st nonzero` column of
`softmode.jl` do, and both outputs were regenerated. The identity
`λ_el = ½ + r_v/6` reproduced by `plastic.jl` is the check that the metric is
now right.

## Kept outputs

`beta_out.txt`, `locking_out.txt`, `materials_out.txt`, `checks_out.txt`,
`softmode_out.txt`, `plastic_out.txt`, `deformed_out.txt`, `plastic_zone_out.txt`,
`quadrature_out.txt`, `basis_out.txt`, `composite_tet_out.txt` are the runs the note's tables were
transcribed from.

## `general.jl` — the general form, one element

Checks the formulas of the general form of the element (note, sec:general,
case 2: the material is called at the modified deformation gradient
F~ = s F with the projected volume) before any kernel is written, on one
distorted TETRA15 with the 14-point rule, for theta = log J and theta = J - 1
and for the constant and the linear projection, with the materials of
ConstitutiveModels through the generic interface alone.

```
julia --project=. research/tet15-p1/prototype/general.jl
```

**Result** (`general_out.txt`). For the neo-Hookean and the elastic J2
material the hand-derived residual equals the automatic-differentiation
gradient of the energy to 4e-16, the closed-form tangent equals the Hessian
to 1.2e-14 and the central-difference Jacobian of the residual to 2e-9, and
the tangent is symmetric to rounding. The three reductions the note claims
hold to rounding: with J2 (split, quadratic volumetric energy, theta = J - 1)
the general residual, tangent and material state equal those of the
implemented split form in the elastic and in the plastic regime (eqps up to
0.52); with the neo-Hookean material (split, non-quadratic volumetric
energy) the general residual equals the case-1 residual with the second
projection; with the constant projection and theta = J - 1 the stress reduces
to P = s P~ + p~ (J - J~) F^-T at every point.

**Finding on the material, corrected in ConstitutiveModels (commit 6705a4e).** The
first run of this script found that in the plastic regime the tangent of
the J2 model (BOX 9.2 of Simo and Hughes) was not the symmetric part of the
derivative of its stress: 2e-5 at eqps 0.14 and 5e-2 at eqps 0.52 at the
fourteen states of the element, and 0.13 for the pointwise element against
the central-difference Jacobian of its residual. The cause was the
coefficient beta_2 of BOX 9.2, which divided by the shear modulus mu
instead of the effective modulus mu_bar = mu tr(b_bar_e_trial)/3; the two
coincide at small strain and the term vanishes with H = 0, which is why the
model's own checks had not seen it. With the correction the plastic rows
of the central-difference column are 7e-7 to 9e-6, the antisymmetric part
that BOX 9.2 drops by construction.
