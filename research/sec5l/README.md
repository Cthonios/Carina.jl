# SEC5L

**A Five-Field Mixed Tetrahedral Finite Element on the Symmetric Elasticity
Complex with Logarithmic Strain.** Alejandro Mota, James W. Foulk III.

Status: **theory in progress, nothing implemented.** Carina has no SEC5L
element, and none of the infrastructure it needs (below). This directory holds
the argument, not an implementation.

```
make        # -> main.pdf   (20 pages)
```

## The problem it addresses

One tetrahedral formulation that is robust across four regimes *at once*:

1. near-incompressible materials, κ/μ up to ~10³–10⁴;
2. isochoric plasticity (volume-preserving plastic flow);
3. weak shocks where κ exceeds μ by orders of magnitude;
4. large rotations.

Tetrahedra are non-negotiable — complex geometry has to be meshable. True
κ → ∞ incompressibility was dropped as a target on 2026-04-19; the practical
bound is finite.

## The formulation

Five fields — displacement **u**, deviatoric strain **E**_dev, volumetric
strain θ = log J, deviatoric stress **T**_dev, and pressure p — on the
symmetric elasticity complex, with an orthogonal volumetric–deviatoric split
throughout. It is deliberately the symmetric-tensor analog of the five-field
composite-tet extension of Foulk et al. (2021).

The constitutive backbone is fixed: **Hencky logarithmic strain with Kirchhoff
stress.** That pairing is chosen because isotropic elasticity becomes
algebraically *linear* in log-strain space, the J2 return map reduces exactly
to small-strain radial return, the volumetric–deviatoric split is exactly
orthogonal, and the response stays thermodynamically sensible under large
compression where neo-Hookean degenerates.

`log C` is evaluated by **inverse scaling-and-squaring with Padé
approximants**, not by spectral decomposition — the distinction matters and the
paper is written that way.

## Decisions already made

| Decision | Date | Substance |
|---|---|---|
| Phase 1 target | 2026-04-19 | SEC five-field with Hencky. Projection-and-eliminate to a displacement-only solve; no saddle point, since κ → ∞ was dropped. |
| Phase 2 target | 2026-04-19 | GEC three-field (u, K, P) for anisotropic materials, where Hencky–Kirchhoff conjugacy breaks down and unsymmetric K = ∇u is natural. Out of scope for Phase 1. |
| Constitutive backbone | — | Hencky + Kirchhoff, Simo (1992) multiplicative finite-strain J2. |

## Ruled out, and why

Recorded here so the design space is not re-litigated:

- **VMS/SUPG stabilized elements (Scovazzi-style).** Produced unacceptable
  equivalent-plastic-strain oscillations that could not be mitigated. The
  failure was in the *plasticity* regime, not the shock regime: subgrid
  stabilization couples back into the deviator and corrupts eqps.
- **F-bar and statically condensed composite-tet.** This lineage is considered
  exhausted by the prior work it descends from (CompTet 2016, Foulk 2021,
  Mota et al. 2013). The goal is a genuinely different architecture, not
  another patch in the same family.

## Hard constraints on any candidate

1. **Strain is an independent field**, not reconstructed from ∇u. Two-field
   Hellinger–Reissner is out for this reason.
2. **Constitutive evaluation must not cross discretization spaces.** Feeding a
   return map strain from one space and projecting stress back into another is
   the diagnosed cause of the VMS failure above. Strain and stress should be
   independent fields in conforming spaces, evaluated pointwise at shared
   quadrature points.
3. **One formulation for statics and dynamics.** Only the time integrator
   changes between regimes — no statics/dynamics hybrid.

## What implementation would require

The largest item is not in Carina at all: **symmetric-tensor H(div; 𝕊) and
H(curl; 𝕊) tetrahedral elements** in ReferenceFiniteElements.jl and
FiniteElementContainers.jl — weakly symmetric Arnold–Falk–Winther with a skew
multiplier, TDNNS, or Hu–Zhang. Until those exist there is nothing to
discretize onto.

## Validation target

The benchmark set is chosen from where the composite tet is known to fail:
punch indentation, nearly-incompressible Cook's membrane in 3D, large-rotation
twist of a bar, and a weak shock in a high-κ/μ material. The comparison of
record is CompTet versus SEC5L on those four.

## Provenance

Developed on Sandia Overleaf and moved here on 2026-08-30, with the six
commits of its edit history preserved through `git subtree`. Overleaf is
retired as of that move; this repository is now the single source of truth.
Co-authoring therefore happens through git rather than through the Overleaf
editor.

The 21 cited papers are **not** in this repository — they are copyrighted and
would add ~47 MB to history permanently. `references.bib` carries the full
citation record, which is what makes the bibliography reproducible. The PDFs
live outside the repo alongside the original session bundle.
