# Punch indentation (quarter model)

Frictionless indentation of a cylinder by a rigid flat circular punch, in
quasi-static finite-deformation J2 plasticity, nearly incompressible: the
benchmark on which the composite tetrahedron's soft mode is expected, under
a plastic zone confined by elastic material.  It has no published
three-dimensional tetrahedral instance; the definition below is the
reference.

## Problem

Units m, kg, s (Pa).  Punch radius a = 1 mm.

- **Geometry:** the quarter x >= 0, y >= 0 of the cylinder of radius R = 4a
  and height H = 4a, axis along z, base at z = 0, top face at z = H.  The
  punch footprint is the quarter disk r <= a on the top face.
- **Boundary conditions:**
  - Symmetry planes: u_x = 0 on x = 0 (node set `symm-x`), u_y = 0 on y = 0
    (`symm-y`).  They remove the translations in x and y and the rotation
    about z; no other constraint.
  - Base: u_z = 0 (a frictionless rigid plane).
  - Lateral surface and the top face outside the footprint: free.
  - Footprint: u_z = −d(t) on every node of the faces inside the footprint,
    the nodes on the footprint circle and on the symmetry planes included;
    u_x and u_y free.  This is frictionless contact with a flat rigid punch
    whose contact set is the footprint; no contact algorithm is used.
- **Loading:** d(t) = 0.2a t for t in [0, 1]; output at the 20 times
  t = 0.05k (depth 0.01a k).
- **Material:** J2 plasticity with linear isotropic hardening, finite
  deformation.  μ = 1.0e9 Pa, κ = 1.0e13 Pa (κ/μ = 1e4; E = 2.99990e9 Pa,
  ν = 0.49995, computed from κ and μ to double precision);
  σ_y = 1.0e6 Pa; hardening modulus H = 1.0e7 Pa (case A) or 0 (case B).
  Density 8930 kg/m³ (unused).
- **Rigid-plastic reference:** the mean pressure under a smooth circular
  punch on a rigid-perfectly-plastic half-space is about 5.7 k, k = σ_y/√3,
  so the full punch force approaches about 5.7 σ_y/√3 · π a² = 10.3 N for
  case B, modified by the finite body and the pile-up.  A check of order.
- **Punch force:** the full force is four times the sum of the z reactions
  on the footprint nodes of the quarter.

## Meshes

`punch.jou` (Cubit, mm, four-node tetrahedra):
- The cylinder is cut by the planes x = 0 and y = 0 and the three other
  quarters are deleted before meshing, so that the symmetry planes, the
  base, the lateral surface, the footprint and the rest of the top face are
  separate surfaces; the axis x = y = 0 is a curve (node set `axis`).
- The region r <= 2a, z >= H − 2a is meshed at size h, the rest at 4h.
- The footprint quarter circle is imprinted on the top face only.
- Smoothing: Cubit's condition-number smoother only.

`convert.jl` writes TETRA10 in m, with the lateral midpoints on r = R and the
midpoints of the footprint-circle edges on r = a; `Carina.tetra15_mesh` then
writes TETRA15, and `check_tetra15.jl` checks that the footprint nodes lie
on z = H, the circle nodes on r = a, node set `punch` in r <= a and the
symmetry sets on their planes.  Counts in `mesh-sizes.tsv`:

| level | h (mm) | tetrahedra | TETRA10 nodes | TETRA15 nodes | shape q min |
|---|---|---|---|---|---|
| a/4 | 0.25 | 6 187 | 9 146 | 28 129 | 0.618 |
| a/8 | 0.125 | 38 818 | 54 582 | 172 391 | 0.602 |
| a/16 | 0.0625 | 277 530 | 380 564 | 1 218 149 | 0.583 |
| a/32 | 0.03125 | 2 113 349 | 2 860 382 | 9 219 491 | 0.583 |

a/32 is counted, not run (above about 1.5 million TETRA15 elements).

## Time stepping

The first iterate of a quasi-static step from the previous solution with the
new u_z on the footprint nodes only puts the whole increment Δd in the first
element layer, a volumetric strain of order Δd/h and a pressure of order
κΔd/h; Newton converges from it only while (Δd/h)(κ/μ) stays below about 1
(Δt below about 1.2e-4 at a/4).  Carina's tangent predictor (`predictor:
tangent`) starts each step from the response of the tangent at the converged
state to the boundary increment, so that the error of the first iterate is
of second order and the admissible step is set by (Δd/h)² κ/μ below about 1.
Both codes therefore start at Δt = 0.05/4096 and double the step after each
converged step up to the maximum step of the level, 0.00625 at a/4, halved per
refinement level; a failed step is halved (at most ten times), and after a
cut the step regrows by 1.5.  The 20 output times are stops that every step
lands on.

## Driver

`run.jl`: stages `mesh`, `convert`, `run`; `--h` (divisors of a), `--case`
(A, B), `--elements` (tet15-p1, tetra15), `--device` (cpu, cuda).  The deck
has the tangent predictor, inexact Newton with Eisenstat–Walker forcing (floor
1e-8) and CG with AMG (at most 1000 iterations); the driver steps the
controller itself, to apply the growth of the step described above (Carina's
controller has one increase factor).  The full punch force is four times the
sum over node set `punch` of the z component of the assembled internal force
of the quarter.  `extract_fields.jl` writes the reported files of one run
(force, element fields at steps 10 and 20, the jumps between neighboring
elements, the axisymmetry of the top face, the profiles); `render.py`
(pvpython) renders p and the plastic strain on the symmetry face y = 0 and on
the deformed top face at step 20.
