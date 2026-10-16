# Cook's membrane: TET15-P1 against CT10

This benchmark measures volumetric locking and its remedies on Cook's
membrane in three dimensions, with the same TETRA10 meshes in two codes:
Carina, for TET10 and TET15 with the volumetric response at each quadrature
point (pointwise) and for TET15-P1 and TET15-P0, the fifteen-node element
with the linear and the constant projection of the volumetric strain
(`model.volumetric projection`, `research/tet15-p1/note.tex`), and Albany-LCM, for the composite
tetrahedron CT10 with volume-averaged J and pressure (`Use Composite Tet 10`,
`Weighted Volume Average J`, `Volume Average Pressure`) and for the pointwise
TET10 as the cross-code control.

## Problem

The trapezoid with corners (0, 0), (48, 44), (48, 60), (0, 44) in the x-y
plane, extruded by 10 in z (`cook.jou`, Cubit).  The face x = 0 is clamped in
all three components; the face x = 48 carries a uniform shear traction q in
y, applied over ten equal load steps.  The material is the Simo-Hughes J2
model in both codes.  Two cases:

| case | E | ν | σ_y | K | q |
|---|---|---|---|---|---|
| elastic | 240.565 | 0.4999 | 1e10 (no yield) | 0 | 6.25 |
| plastic | 206.9 | 0.29 | 0.45 | 0.12924 | 0.14 |

The elastic case is the near-incompressible membrane of the plane-strain
benchmark, with q = F/16 for the force F = 100 per unit thickness.  The
plastic case has the material of the elastoplastic membrane of Simo and
Armero (1992); their load F = 1.8, q = 0.1125, leaves the three-dimensional
membrane with free faces almost elastic (tip displacement 0.21 with no
difference between elements), the pointwise TET10 collapses between
q = 0.16 and q = 0.20 (the hardening modulus is 0.06% of E), and Albany's
Newton on CT10 stops converging at q = 0.158 on the
h = 4 mesh (the residual stagnates at 2e-4 with the line search collapsing,
which indicates a tangent inconsistent with the residual in the plastic
regime), so the traction is 0.14.  The
two codes differ in the volumetric law of the J2 model: κ(J − 1) in Carina,
κ(J − 1/J)/2 in Albany; at ν = 0.4999 the response is set by the deviatoric
part and the two pointwise TET10 runs agree to 0.2% in the tip
displacement on the coarsest mesh.

The measured quantity is the displacement u_y over the nodes of the loaded
face at full load: its mean and its maximum.  Albany renumbers the nodes of
its output file, so the loaded nodes are identified by node id through the
node maps of both files.

## Results

`RESULTS.md` holds the tables (mean u_y of the loaded face at full load;
`results.tsv` has every record with the maximum, the unknown count and the
wall time).  Findings, 2026-09-27, Carina 68c135e with the local branches
named in `run.jl`; the TET15-P1 and TET15-P0 values are those of the rerun
of 2026-10-16 (Carina 29c9606, ConstitutiveModels a0550bd), in which the
projected strain is log J, and change the earlier ones by at most 0.1%
(elastic) and 1e-4 relative (plastic):

- The pointwise TET10 agrees between the two codes to six digits on every
  mesh and in both cases (elastic: 8.19605 at h = 8, 8.32411 at h = 4;
  plastic: 0.26893, 0.27101, 0.27204).  The comparison of the other elements
  rests on this check.
- Elastic, ν = 0.4999: the pointwise TET10 and TET15 converge from below
  (TET10 8.196, 8.324, 8.420 and TET15 8.224, 8.349, 8.430 at h = 8, 4,
  2) and TET15-P0 from above (8.564, 8.512, 8.495).  TET15-P1 (8.410,
  8.454, 8.471) and CT10
  (8.460, 8.464, 8.477) are within 0.6% of each other on every mesh and
  change least with refinement; the limit lies between 8.477 and 8.500.
  The h = 2 runs (2026-09-28) used Carina b1d9ca4 with `--no-line-search`
  on Rigel (1099 s for TET10, 3214 to 3639 s for TET15, TET15-P1 and TET15-P0,
  16 threads) and Albany with serial KLU2 on Sirius (26 496 s for TET10,
  28 867 s for CT10).  The two TET10 results differ
  by 3e-7.
- Pressure: `extract_pressure.jl` fits a polynomial to the quadrature-point
  values of p = −tr σ / 3 in each element (quadratic for the 14-point rule of
  TET15, linear for the 4-point rule of TET10, weighted by the rule) and
  writes it at the ten nodes of a quadratic tetrahedron; `render_pressure.py`
  draws it on the front face of the deformed membrane.  The fit reproduces
  TET15-P1 and TET15-P0 to rounding error and the pointwise
  TET15 pressure to 0.11 of max |p| (h = 2, plastic).  Albany writes one
  pressure per CT10 element (equal at its five points).  In the
  elastic case at h = 2 the pointwise TET10 and TET15 change sign inside
  elements over the whole membrane (front-face range −226 to 211 and −512 to
  1010); TET15-P1 is smooth except in the elements along the
  clamped edge (−43 to 61), and TET15-P0 and CT10
  give −7.5 to 22 and −9.4 to 35.
- Plastic, traction 0.14: all elements converge to about 0.273; the spread
  shrinks from 2.3% (h = 8) to 0.5% (h = 2), with the projected and composite
  elements closest to the limit on every mesh.  At this load the plastic
  zone does not lock the quadratic elements.  Albany's Newton on the
  CT10 stops converging near the collapse load (0.158 at
  h = 4, 0.137 at h = 2), which the other elements pass.

## Meshes and the comparison in another code

The meshes are not in git (`meshes/` is ignored); `cook.jou` regenerates them
exactly, since the Cubit size is h itself and there is no smoothing.  To build
the TETRA10 meshes of CT10 (`meshes/cook-h<h>.g`, Cubit)
and the TETRA15 meshes of TET15-P1 (`meshes/cook-h<h>-tet15.g`,
`Carina.tetra15_mesh`, Exodus TETRA15 node order) and stop:

```
julia --project=. benchmark/tet15-p1/cook/run.jl --h 8,4,2 --meshes-only
```

| h | elements | TETRA10 nodes | TETRA15 nodes |
|---|---|---|---|
| 8 | 327 | 647 | 1 721 |
| 4 | 1 860 | 3 208 | 9 127 |
| 2 | 14 475 | 22 059 | 66 813 |

Sets, in both meshes: node set `clamp` (the face x = 0), side set `load` and
node set `load_nodes` (the face x = 48); block `membrane`.  The names do not
collide, so Sierra/SM reads the meshes as they are (the Taylor meshes need
renamed side sets, `../taylor/sierra/rename-sidesets.jl`).  Units are
consistent and unnamed (length as in `cook.jou`, stress as E).

A run in another code, for CT10 on the TETRA10 mesh and
for TET15-P1 on the TETRA15 mesh, reproduces this one with:

- the three displacement components fixed on node set `clamp`;
- a traction q in +y on side set `load`, a dead load per unit reference
  area, increased linearly over ten equal quasi-static load steps (more
  steps where Newton fails; the result is the state at full load);
- the Simo-Hughes J2 model with linear isotropic hardening and the constants
  of the table under "Problem" (elastic case: σ_y = 1e10, so that it does
  not yield);
- the measured quantity: the mean of u_y over the nodes of node set
  `load_nodes` at full load (`results.tsv` also has the maximum).

The volumetric part of the J2 energy differs between the codes used here:
κ log J / J for the pressure in Carina (energy κ/2 (log J)², since
2026-10-16; before, κ(J − 1)), κ(J − 1/J)/2 in Albany.  At
ν = 0.4999 the two pointwise TET10 runs agree to six digits, and at ν = 0.29
the difference is below the reported digits; another code should state its
own.  Reference values (mean u_y of the loaded face at full load):

| h | LCM CT10 | Carina TET15-P1 | TET10 pointwise (both) |
|---|---|---|---|
| elastic, 8 | 8.4596 | 8.4099 | 8.1960 |
| elastic, 4 | 8.4643 | 8.4536 | 8.3241 |
| elastic, 2 | 8.4772 | 8.4709 | 8.4203 |
| plastic, 8 | 0.2716 | 0.2706 | 0.2689 |
| plastic, 4 | 0.2725 | 0.2721 | 0.2710 |
| plastic, 2 | 0.2728 | 0.2724 | 0.2720 |

The pointwise TET10 run is the check that the other code reads the same
problem: it should give the same values to the digits shown.

### Sierra/SM TET15-P1

Sierra/SM TET15-P1 was run on the elastic
case (ν = 0.4999) on these meshes (Rigel session, 2026-10-08; record
`sierra-cook-results.tsv`), with three differences from the runs above:

- the material is a neo-Hookean model with Carina's κ = 4.01e5 and
  μ = 80.2, whose deviatoric part is Carina's and whose volumetric part,
  p = κ(J − 1/J)/2, differs from Carina's at second order in J − 1, which is
  negligible at |J − 1| of order 1e-4;
- the traction is applied as consistent nodal forces, integrated with a
  degree-5 rule on the faces (Sierra/SM TET15-P1 does not yet take a traction on
  the 7-node faces of TETRA15), the same treatment as for the
  6-node faces of CT10;
- Newton with a preconditioned conjugate-gradient linear solver, relative
  residual 1e-8, ten load steps.

Mean u_y of the loaded face at full load (maximum in parentheses):

| h (elements) | Sierra/SM TET15-P1 | Carina TET15-P1 | Sierra/SM TETRA15 | Carina TETRA15 | Sierra/SM CT10 | LCM CT10 |
|---|---|---|---|---|---|---|
| 8 (327) | 8.4139 (8.4663) | 8.4099 (8.4663) | 8.2278 | 8.2235 | 8.4289 (8.5700) | 8.4596 |
| 4 (1 860) | 8.4555 (8.4976) | 8.4536 (8.4976) | 8.3511 | 8.3490 | 8.4530 (8.5410) | 8.4643 |
| 2 (14 475) | 8.4722 (8.5081) | 8.4709 (8.5081) | 8.4315 | 8.4300 | did not converge | 8.4772 |

TET15-P1 agrees between the two codes to 0.05% (h = 8), 0.02% (h = 4) and
0.015% (h = 2) in the mean and to the five digits shown in the maximum, and
TETRA15 without the projection to 0.05%.  The Carina TET15-P1 values are
those of 2026-10-16 (29c9606), with the projected strain log J, the measure
Sierra/SM TET15-P1 projects; with J − 1 (Carina 68c135e) they were 8.4122,
8.4551 and 8.4716 in the mean and 8.4685, 8.4990 and 8.5088 in the
maximum, so the common measure brings the maxima to agreement and moves the
means by 0.01 to 0.03%, within the difference of the two volumetric laws.
Sierra/SM CT10 is within 0.4% of LCM CT10 at h = 8
and 4; at h = 2 its Newton iteration with this solver stalled at 0.8 of the
load with ten steps and at 0.5 with twenty.  Wall time on two AMD EPYC 9634
sockets (total Newton iterations over the ten steps): TET15-P1 46 s (113),
45 s (120) and 153 s (511) at h = 8, 4 and 2 on 1, 8 and 32 MPI ranks;
TETRA15 38 s (105), 65 s (112) and 294 s (119); CT10
18 s (110) and 18 s (116) at h = 8 and 4.  At h = 2 one load step of
TET15-P1 was accepted at a relative residual of 2.2e-6, the residual floor
near 1e-6 that Carina also reaches on this problem.

Sierra/SM TET15-P1 writes one stress per element; `extract_pressure.jl` reads it
(element variables `stress_xx`, ..., nodal `displacement_x`, ...) as one
value per element.  At h = 2 its element pressures correlate with the
element means of Carina's TET15-P1 to 0.999999 (RMS difference 0.0054
against max |p| = 30.3).  Without the projection the two fields are not
comparable element by element: Carina's pressure oscillates in sign inside
each element, and the single value of Sierra/SM TET15-P1 is its average.

## Running

From the Carina root, with Cubit at `/usr/local/cubit/cubit` and Albany at
`~/LCM/lcm-build-serial-gcc-release/src/Albany`:

```
julia -t 12 --project=. benchmark/tet15-p1/cook/run.jl --h 8,4 --cases elastic,plastic
julia -t 12 --project=. benchmark/tet15-p1/cook/run.jl --report
```

The threads serve Carina's element loops.  Albany runs on twelve MPI ranks in
the plastic case (OpenMPI of the system, meshes decomposed with the SEACAS
`decomp` built from `~/Repos/seacas` and installed under `~/LCM/seacas-tools`)
and on one rank in the elastic case, for the reason given below.

`run.jl` builds the meshes it lacks (`meshes/cook-h<h>.g`, and the TETRA15
conversion `meshes/cook-h<h>-tet15.g` through `Carina.tetra15_mesh`), writes
one deck per run under `runs/`, runs Carina in the same process and Albany as
a subprocess, appends one line per run to `results.tsv`, and writes
`RESULTS.md` from the newest record of each cell.

## Solver settings

Near incompressibility makes the residual grow with the square of the
volumetric strain of a Newton step, scaled by κ, so Carina's load steps are
halved on a failed solve (to 1/1000 of the nominal step at least) and grown
back by 1.5 after a successful one; the pointwise TET10 at ν = 0.4999
needs about one hundred steps on the coarsest mesh.  With κ of order 1e5 the
assembled residual has a rounding floor near 1e-8, so both codes converge on
a residual norm of 1e-6 absolute or 1e-8 relative.  Carina uses the SparseArrays direct factorization.  Albany
uses the Amesos2 KLU2 direct solver on one rank in the elastic case: KLU2
fails on a distributed matrix in this build, and GMRES with a
smoothed-aggregation multigrid preconditioner degrades Newton at ν = 0.4999
(207 iterations against 76 for the coarse mesh at a GMRES tolerance of 1e-10;
at 1e-8 it took 25 to 30 iterations per step and cut the load step), so that
eight ranks were slower than the serial direct solver (66 s against 5 s).  In
the plastic case (ν = 0.29) GMRES with the multigrid preconditioner at 1e-10
leaves Newton unchanged and eight ranks were six times faster than the serial
direct solver (8 s against 50 s on the h = 4 mesh), so Albany runs there on
twelve ranks.  Albany uses LOCA natural continuation on the
traction component, with
the step halved on a failed solve (`Failed Step Reduction Factor`) and grown
back after successful ones (`Aggressiveness`, as in the LCM ACE tests); LOCA's
default, arc-length continuation,
stops at the maximum number of steps short of the target traction when the
structure softens, and the driver checks the final parameter value in the
log.
