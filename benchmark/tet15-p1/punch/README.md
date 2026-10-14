# Punch indentation

Frictionless indentation of a cylinder by a rigid flat circular punch, in
quasi-static finite-deformation J2 plasticity, nearly incompressible: the
benchmark on which the composite tetrahedron's soft mode is expected, under
a plastic zone confined by elastic material.  It has no published
three-dimensional tetrahedral instance; the definition below is the
reference.

## Problem

Units m, kg, s (Pa).  Punch radius a = 1 mm.

- **Geometry:** a cylinder of radius R = 4a and height H = 4a, axis along z,
  base at z = 0, top face at z = H.  The punch footprint is the disk r <= a
  on the top face.
- **Boundary conditions:**
  - Base: u_z = 0 (a frictionless rigid plane).
  - Lateral surface and the top face outside the footprint: free.
  - Footprint: u_z = −d(t) on every node of the faces inside the footprint,
    the nodes on the footprint circle included; u_x and u_y free.  This is
    frictionless contact with a flat rigid punch whose contact set is the
    footprint; no contact algorithm is used.
  - The rigid-body modes the base plane leaves (translations in x and y,
    rotation about z) are removed by three point constraints:
    u_x = u_y = 0 at the base node on the axis (node set `axis-base`) and
    u_y = 0 at the base node (R, 0, 0) (node set `pin-base`).  Their
    reactions are reported as a check.
- **Loading:** d(t) = 0.2a t for t in [0, 1]; output at the 20 times
  t = 0.05k (depth 0.01a k).
- **Material:** J2 plasticity with linear isotropic hardening, finite
  deformation.  μ = 1.0e9 Pa, κ = 1.0e13 Pa (κ/μ = 1e4; E = 2.99990e9 Pa,
  ν = 0.49995, computed from κ and μ to double precision);
  σ_y = 1.0e6 Pa; hardening modulus H = 1.0e7 Pa (case A) or 0 (case B).
  Density 8930 kg/m³ (unused).
- **Rigid-plastic reference:** the mean pressure under a smooth circular
  punch on a rigid-perfectly-plastic half-space is about 5.7 k, k = σ_y/√3,
  so the punch force approaches about 5.7 σ_y/√3 · π a² = 10.3 N for case B,
  modified by the finite body and the pile-up.  A check of order.

## Meshes

`punch.jou` (Cubit, mm, four-node tetrahedra):
- The cylinder is cut by the planes x = 0 and y = 0, so that the axis and
  the base half-line y = 0, x > 0 are mesh curves (node sets `axis`, `pin`,
  and the one-node sets `axis-base`, `pin-base`).
- The region r <= 2a, z >= H − 2a is meshed at size h, the rest at 4h.
- The footprint circle is imprinted on the top face only.
- Smoothing: Cubit's condition-number smoother only.

`convert.jl` writes TETRA10 in m, with the lateral midpoints on r = R and the
midpoints of the footprint-circle edges on r = a; `Carina.tetra15_mesh` then
writes TETRA15, and `check_tetra15.jl` checks that the footprint nodes lie
on z = H and the circle nodes on r = a.  Counts in `mesh-sizes.tsv`:

| level | h (mm) | tetrahedra | TETRA10 nodes | TETRA15 nodes | shape q min |
|---|---|---|---|---|---|
| a/4 | 0.25 | 24 635 | 34 194 | 108 729 | 0.599 |
| a/8 | 0.125 | 156 450 | 213 014 | 684 439 | 0.585 |
| a/16 | 0.0625 | 1 110 516 | 1 498 075 | 4 837 409 | 0.570 |

## Starting each step

The first iterate of a quasi-static step is the previous solution with the
new u_z on the footprint nodes only.  The whole increment Δd is then taken
by the first element layer, a volumetric strain of order Δd/h and a pressure
of order κΔd/h: 400 μ at a/4 with the nominal increment 0.01a.  Newton does
not recover from that state, in Carina or in Sierra/SM, for any element and
even for an elastic material at κ/μ = 1e4 (it does at κ/μ = 1).  A step must
therefore start from a predictor that spreads the boundary increment through
the body, and the first step, which has no previous increment, must be small
enough that κΔd/h stays near μ.  The driver starts at Δt = 0.05/4096 and lets
the step double after each converged step up to 0.05.

## Driver

`run.jl`: stages `mesh`, `convert`, `run`; `--h` (divisors of a), `--case`
(A, B), `--elements` (tet15-p1, tetra15), `--device` (cpu, cuda).  The punch
force is the sum over node set `punch` of the z component of the assembled
internal force.  `extract_fields.jl` writes the reported files of one run
(force, element fields at steps 10 and 20, the jumps between neighboring
elements, the axisymmetry of the top face, the profiles).
