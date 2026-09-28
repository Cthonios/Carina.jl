# Cook's membrane: projected TETRA15 against the composite tetrahedron

This benchmark measures volumetric locking and its remedies on Cook's
membrane in three dimensions, with the same TETRA10 meshes in two codes:
Carina, for the TETRA10 and TETRA15 elements with and without the projected
volumetric strain (`model.volumetric projection`, the mean-dilatation
reduction of `research/hw3l/note.tex`), and Albany-LCM, for the composite
tetrahedron with volume-averaged J and pressure (`Use Composite Tet 10`,
`Weighted Volume Average J`, `Volume Average Pressure`) and for the pointwise
TETRA10 as the cross-code control.

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
difference between elements), the pointwise TETRA10 collapses between
q = 0.16 and q = 0.20 (the hardening modulus is 0.06% of E), and Albany's
Newton on the composite tetrahedron stops converging at q = 0.158 on the
h = 4 mesh (the residual stagnates at 2e-4 with the line search collapsing,
which indicates a tangent inconsistent with the residual in the plastic
regime), so the traction is 0.14.  The
two codes differ in the volumetric law of the J2 model: κ(J − 1) in Carina,
κ(J − 1/J)/2 in Albany; at ν = 0.4999 the response is set by the deviatoric
part and the two pointwise TETRA10 runs agree to 0.2% in the tip
displacement on the coarsest mesh.

The measured quantity is the displacement u_y over the nodes of the loaded
face at full load: its mean and its maximum.  Albany renumbers the nodes of
its output file, so the loaded nodes are identified by node id through the
node maps of both files.

## Results

`RESULTS.md` holds the tables (mean u_y of the loaded face at full load;
`results.tsv` has every record with the maximum, the unknown count and the
wall time).  Findings, 2026-09-27, Carina 68c135e with the local branches
named in `run.jl`:

- The pointwise TETRA10 agrees between the two codes to six digits on every
  mesh and in both cases (elastic: 8.19605 at h = 8, 8.32411 at h = 4;
  plastic: 0.26893, 0.27101, 0.27204).  The comparison of the other elements
  rests on this check.
- Elastic, ν = 0.4999: the pointwise TETRA10 and TETRA15 converge from below
  (TETRA10 8.196, 8.324, 8.420 and TETRA15 8.224, 8.349, 8.430 at h = 8, 4,
  2) and the constant projection from above (8.572, 8.518, 8.500).  The
  linear projection (8.412, 8.455, 8.472) and the composite tetrahedron
  (8.460, 8.464, 8.477) are within 0.6% of each other on every mesh and
  change least with refinement; the limit lies between 8.477 and 8.500.
  The h = 2 runs (2026-09-28) used Carina b1d9ca4 with `--no-line-search`
  on Rigel (1099 s for TETRA10, 3214 to 3639 s for the TETRA15 variants,
  16 threads) and Albany with serial KLU2 on Sirius (26 496 s for TETRA10,
  28 867 s for the composite tetrahedron).  The two TETRA10 results differ
  by 3e-7.
- Pressure at h = 2: `extract_pressure.jl` writes the element-mean pressure
  p = −tr σ / 3 at full load, and `render_pressure.py` draws it on the
  front face of the deformed membrane.  In the elastic case the pointwise
  TETRA10 and TETRA15 show element-to-element oscillation of p along the
  clamped edge (front-face range −12.2 to 37.2 and −12.9 to 55.4); the
  linear and constant projections and the composite tetrahedron do not
  (−7.43 to 26.8, −7.46 to 21.7, −6.79 to 15.2).
- Plastic, traction 0.14: all elements converge to about 0.273; the spread
  shrinks from 2.3% (h = 8) to 0.5% (h = 2), with the projected and composite
  elements closest to the limit on every mesh.  At this load the plastic
  zone does not lock the quadratic elements.  Albany's Newton on the
  composite tetrahedron stops converging near the collapse load (0.158 at
  h = 4, 0.137 at h = 2), which the other elements pass.

## Running

From the Carina root, with Cubit at `/usr/local/cubit/cubit` and Albany at
`~/LCM/lcm-build-serial-gcc-release/src/Albany`:

```
julia -t 12 --project=. benchmark/hw3l/cook/run.jl --h 8,4 --cases elastic,plastic
julia -t 12 --project=. benchmark/hw3l/cook/run.jl --report
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
back by 1.5 after a successful one; the pointwise TETRA10 at ν = 0.4999
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
