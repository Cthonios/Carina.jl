# Taylor bar impact: TET15-P1 against the composite tetrahedron

This benchmark measures TET15-P1 under localized plastic flow at plastic
strains above 3 in explicit dynamics, the regime in which the composite
tetrahedron shows a soft mode without its stabilization (Foulk et al. 2021,
Sec. 4.4).  TET15-P1 runs in Carina; the composite tetrahedron runs in
Sierra/SM, the code of the paper, on the same TETRA10 meshes.

## Problem

A copper bar of length 32.4 mm and radius 3.2 mm, axis along z, strikes a
rigid frictionless wall at z = 0 with velocity 227 m/s (Foulk et al. 2021;
Simo 1992, Sec. 7.2; Wilkins and Guinan 1973).  Density 8930 kg/m³,
E = 117 GPa, ν = 0.35, J2 plasticity with linear isotropic hardening, yield
stress 400 MPa, hardening modulus 100 MPa.  Explicit central differences to
80 μs, no bulk viscosity.  The wall is u_z = 0 on the impact face, which is
in contact at t = 0 and stays in contact while the bar moves toward the wall
(Simo 1992 constrains nodes that reach the wall to remain on it).  Units in
the Carina and Sierra runs: m, kg, s.

Measured: the radius of the impact face (maximum over its nodes of the
distance from the axis) and the length of the bar (maximum z of the free
face) against time, every 1 μs.  Reference values of the final radius:
7.22 mm for the converged Q1/P0 hexahedron of Foulk et al. (h = 0.047 mm),
7.12 mm (Simo 1992), 7.11 mm (Zienkiewicz et al. 1998); the composite
tetrahedron at h = 0.75 mm is within 0.11% of its own converged value.

## Meshes

`taylor.jou` (Cubit, four-node tetrahedra, mm): the full bar for
h = 1.5, 0.75, 0.38 and 0.19 mm and the quarter bar (x ≥ 0, y ≥ 0, symmetry
planes x = 0 and y = 0) for h = 0.094 and 0.047 mm, as in the paper
(Figure 14).  The Cubit size is adjusted until the element count is within 3%
of the paper's count (3495, 24 739, 187 819, 1 533 115, 2 712 078 and
20 521 040), because the paper's h is not Cubit's size parameter (a Cubit
size of 1.5 gives 2536 elements; 1.39 gives 3522).  Sets: node sets `all`,
`impact`, and on the quarter bar `symx`, `symy`; side sets `impact`, `free`,
`lateral`, and on the quarter bar `symx`, `symy`.

Each mesh is then smoothed by Norma's energetic mesh smoothing (four-node
tetrahedra only; `model.type: mesh smoothing`, Seth–Hill energy, L-BFGS,
ten steps of 64 iterations), with the nodes of each surface held on it by
level-set constraints: the cylinder x² + y² = R², the planes z = 0 and
z = 32.4, and the symmetry planes.  `convert.jl` adds the edge midpoints,
moves those on the lateral surface onto the cylinder, scales to meters and
writes TETRA10 (the mesh of the composite tetrahedron);
`Carina.tetra15_mesh` adds the face and interior nodes (the mesh of
TET15-P1).  At h = 1.5 mm smoothing raises the smallest shape quality
q = 12 (3V)^(2/3) / Σ l² from 0.648 to 0.708 and the mean from 0.879 to
0.895.

The meshes are not in git (`meshes/` is ignored; the h = 0.19 mm TETRA15
mesh has 6.7 million nodes); they are regenerated from the files here.  The Cubit size
that gave each level its element count is recorded in `mesh-sizes.tsv`,
from which `run.jl` starts, so that Cubit reproduces the four-node mesh:
at h = 1.5 mm the regenerated mesh has the coordinates and connectivity of
the one used, to the last bit.  The smoothing is not reproduced bit for
bit: two smoothing runs of the same h = 0.75 mm Cubit mesh, on two
machines, gave meshes whose radius histories differ by 2.8e-7 relative.  A
regenerated mesh therefore reproduces the results below to that order, and
a comparison of two elements is exact as long as both run on the same
files.  The h = 0.19 mm level was meshed on Rigel, and regenerated there on
2026-10-08 from the recorded size with the same element count, node counts
and shape quality (smallest 0.691, mean 0.912 after smoothing).

| h (mm) | elements | paper | TETRA10 nodes | TETRA15 nodes | Cubit size |
|---|---|---|---|---|---|
| 1.5 | 3 522 | 3 495 | 5 601 | 16 587 | 1.38968 |
| 0.75 | 25 029 | 24 739 | 36 666 | 113 331 | 0.72460 |
| 0.38 | 190 727 | 187 819 | 267 150 | 845 573 | 0.36591 |
| 0.19 | 1 521 212 | 1 533 115 | 2 083 320 | 6 672 877 | 0.17902 |

To regenerate the meshes of all four levels (Cubit and Norma installed;
`TAYLOR_CUBIT`, `TAYLOR_NORMA` and `TAYLOR_NORMA_THREADS` locate them):

```
julia -t 16 --project=. benchmark/tet15-p1/taylor/run.jl --h 1.5,0.75,0.38,0.19 \
    --stages mesh,smooth,convert
```

which writes `meshes/taylor-h<h>-tet4.g` (Cubit), `smooth/taylor-h<h>-smooth.e`
(Norma), `meshes/taylor-h<h>-tet10.g` (the composite tetrahedron) and
`meshes/taylor-h<h>-tet15.g` (TET15-P1), in meters.

## Running

From the Carina root, with Cubit and Norma installed:

```
julia -t 16 --project=. benchmark/tet15-p1/taylor/run.jl --h 1.5,0.75 \
    --stages mesh,smooth,convert,run --elements tet15-p1,tet15-p0
```

`TAYLOR_CUBIT`, `TAYLOR_NORMA` and `TAYLOR_NORMA_THREADS` locate the tools.
`--device rocm` or `--device cuda` runs on a GPU; the vendor package must be
loaded in the calling session, which the library does not depend on:

```
julia --project=bin -e 'using AMDGPU; append!(ARGS, ["--h", "0.75", "--stages", "run",
    "--elements", "tet15-p1", "--device", "rocm"]); include("benchmark/tet15-p1/taylor/run.jl")'
```

with `JULIA_LOAD_PATH="@:$PWD:@stdlib"` so that the `bin` environment finds
the test dependencies of the root one.  A device run writes to
`runs/<element>-h<h>-<device>` and records the device in `results.tsv`.
Each run writes `runs/<element>-h<h>/taylor.e`, `taylor.yaml` and
`history.tsv` (time, radius, length in m), and appends one record to
`results.tsv` (final radius and length in mm, wall time).

The time step is 0.8 times the critical step of central differences,
2/√λ_max with λ_max the largest eigenvalue of M_L⁻¹K (lumped mass M_L,
tangent stiffness K), capped at 1e-8 s (`stable time step method: global`).
λ_max is computed by power iteration every 200 steps; between these, the
step follows the element-length estimate, recomputed every 10 steps on the
current configuration, scaled by the ratio of the two at the last
computation of λ_max.  The element-length estimate alone, the smallest
distance between two nodes of an element over the dilatational wave speed,
is not a bound on the critical step in this problem.  For TET15-P1 at
h = 0.75 mm the ratio of the critical step to that estimate was 1.14 at
1 μs, 1.02 at 10 μs, 0.69 at 20 μs, 0.42 at 28 μs, 0.31 at 40 μs and 0.29
at 60 and 80 μs: the run with Courant number 0.5 on the element estimate
stopped with a non-finite residual at 28 μs, and the run with 0.25 reached
80 μs with a step 0.82 times the critical one at 40 μs.

| Stable step at h = 0.75 mm, TET15-P1, 12 threads (AMD 9900X) | Steps | Wall time | Final radius |
|---|---|---|---|
| element estimate, Courant number 0.25 | about 54 000 | 847 s | 7.2225 mm |
| global, 0.8 | 43 863 | 803 s | 7.2232 mm |

The computation of λ_max adds 17% to the time per step.

On a GPU the run gives the same history.  At h = 0.75 mm with the global
stable step, the radius and length of every output frame of the run on an
AMD RX 7600 equal those of the run on the CPU to every written digit, and
the 221 stable-step estimates are the same numbers; the wall time is 513 s on
the RX 7600 against 839 s on 12 threads of an AMD Ryzen 9 9900X.  At
h = 1.5 mm the two also agree to every digit (80 s against 67 s: the GPU
is slower than the CPU on 3522 elements).  The device run holds about 590 MB
of VRAM throughout: the collector is run every 100 steps on a device
(`src/simulation.jl`), without which the temporaries of the explicit step
exhausted the 8 GB of the card after about 5 000 steps.

Two defects of Carina were found and corrected in setting up this problem:
`tetra15_mesh` left the element centroids out of node sets that cover the
whole volume, so the initial velocity on node set `all` left 0.305 of the
mass at rest (the final radius was 5.45 mm on every mesh); and the stable
time step recomputed during a run used the reference mesh, not the
current one.

## Composite tetrahedron and TET15-P1 in Sierra/SM

On the same `meshes/taylor-h<h>-tet10.g`: the explicit dynamics of Sierra/SM
with the composite tetrahedron of Foulk et al. (2021) at its default
stabilization α = 0.1, the J2 model with linear hardening of the same
constants, the initial velocity on node set `all`, u_z = 0 on node set
`impact`, the symmetry conditions on the quarter bar, output every 1 μs to
80 μs.  The same `history.tsv` is extracted from its output.

The files of those runs are in `sierra/`:

- `ct-template.i`, the input deck: the copper of the problem with the
  `fefp` J2 model of LAME (linear hardening), a total Lagrange section with
  `formulation = composite_tet` and `vem exponent = 0.0` (as in the deck of
  the paper), no bulk viscosity, the wall and the initial velocity, output
  and a heartbeat every 1 μs.  `{MESH}`, `{OUT}`, `{HB}` and a `{VEM}` line
  are substituted by the run script.
- `rename-sidesets.jl`: IOSS rejects a node set and a side set of the same
  name (`impact`), so Sierra reads a copy of the mesh with `_face` appended
  to the side-set names; coordinates, connectivity and node sets are
  unchanged.
- `run-ct.sh <h> <ranks> [alpha0|off]`, from this directory: renames the
  side sets, writes the deck, runs `adagio` on `<ranks>` MPI ranks, joins
  the output with `epu`, and writes `sierra/ct-h<h>[-<variant>]/` with the
  deck, the log and `sierra-ct-h<h>-history.tsv` (`history.jl`).  It
  expects Sierra at `~/sierra/code/bin` and OpenMPI at `/usr/lib64/openmpi`;
  the variants set the VEM stabilization parameter (`alpha0`) or both VEM
  parameters (`off`) to zero.

TET15-P1 in Sierra/SM runs on `meshes/taylor-h<h>-tet15.g`, the same
elements with the face and interior nodes that `Carina.tetra15_mesh` adds
(Exodus TETRA15 node order), renamed by `rename-sidesets.jl` in the same
way, with the section of the TET15-P1 implementation in place of
`composite_tet`.  The node sets, side sets, material constants, loading and
output are those of `ct-template.i`.

Reference final values at 80 μs, from `data/` (radius of the impact face and
length of the bar, mm):

| h (mm) | composite tetrahedron, Sierra/SM | TET15-P1, Carina | TET15-P1, Sierra/SM prototype | TET15-P0, Carina |
|---|---|---|---|---|
| 1.5 | 7.2904, 24.439 at 41.6 μs (the run stopped) | 7.2399, 21.425 | 7.2344, 21.429 | 7.4415, 21.551 |
| 0.75 | 7.2820, 21.476 | 7.2225, 21.421 | 7.2186, 21.423 | 7.4747, 21.533 |
| 0.38 | 7.2453, 21.439 | 7.2255, 21.420 | 7.2181, 21.422 | — |
| 0.19 | 7.2326, 21.430 | — | — | — |

The composite tetrahedron converges from above, with an extrapolated limit
of 7.2259 mm (order 1.53 in h); TET15-P1 in Carina is within 0.05% of that
limit at h = 0.75 and 0.38 mm.

## TET15-P1 in a prototype implementation in Sierra/SM

A prototype implementation of TET15-P1 in Sierra/SM was run on the same
TETRA15 meshes (side sets renamed as above), with the finite-deformation J2
model of the Sierra material library with linear hardening, as in the
composite-tetrahedron runs, and no bulk viscosity; its time step was an
element estimate, 0.5 times the smallest node distance over the
dilatational wave speed.  Results (Rigel session, 2026-10-08; records
`data/sierra-taylor-results.tsv` and
`data/sierra-taylor-{tet15-p1,tetra15}-h<h>-history.tsv`; figure
`plot.py`, `taylor-history-sierra.png`):

- Final radius 7.2344, 7.2186 and 7.2181 mm at h = 1.5, 0.75 and 0.38 mm,
  0.08%, 0.05% and 0.10% below Carina's TET15-P1 and within 0.12% of the
  composite tetrahedron's extrapolated limit at every level; the length
  agrees with Carina's to 0.02%.
- Over 0 to 80 μs the radius differs from Carina's by at most 0.14%, 0.16%
  and 0.13% (h = 1.5, 0.75, 0.38 mm), the largest differences in the first
  10 μs; after 40 μs the difference is constant, −0.08%, −0.05% and −0.10%.
  It does not decrease with h, and is consistent with the different J2
  implementations and time steps of the two codes.
- Without the projection (TETRA15, the volumetric response at each
  quadrature point) the final radius at h = 1.5 mm is 7.1268 mm, 1.5% below
  TET15-P1 in the same code: the volumetric locking the projection removes.
- The composite-tetrahedron runs repeated alongside reproduce the recorded
  histories of the study to 1.8e-7 relative (h = 0.75 mm) and exactly
  (h = 0.38 mm).
- Pressure at 12 μs (runs to 12 μs with the element stress written, and
  at h = 0.19 mm the 12 μs output of the run to 80 μs; one value per
  element): largest value on the impact end 1.56, 1.62, 1.63 and 1.63 GPa
  at h = 1.5, 0.75, 0.38 and 0.19 mm (Carina 1.67, 1.58, 1.56, 1.54),
  radius of the impact face 5.447, 5.444, 5.446 and 5.452 mm (Carina 5.452,
  5.451, 5.453, 5.457).  The renders of `render_pressure.py` at `range=-3e8:1.6e9` were
  made where the output is; `compose_pressure.py` takes the rows as its last
  argument, for the grid of the composite tetrahedron, Carina's TET15-P1 and
  the prototype.

Cost on two AMD EPYC 9634 sockets, with MPI ranks:

| run | ranks | steps | wall (s) | Δt at 1 μs | Δt at 80 μs |
|---|---|---|---|---|---|
| TET15-P1, h = 0.75 mm | 96 | 50 832 | 271 | 3.86e-9 | 1.34e-9 |
| composite tetrahedron, h = 0.75 mm | 96 | 44 260 | 90 | 2.76e-8 | 1.13e-9 |
| TET15-P1, h = 0.38 mm | 160 | 187 209 | 6 473 | 1.85e-9 | 3.38e-10 |
| composite tetrahedron, h = 0.38 mm | 160 | 62 821 | 626 | 1.27e-8 | 8.54e-10 |

Status of the prototype, as found in these runs:

- The global (eigenvalue) estimate of the stable step diverged on the
  TETRA15 meshes of the Taylor bar, with and without the projection, and
  those runs stopped.  With the element estimate every run reached 80 μs.
  The cause (Rigel, 2026-10-09): the estimate is a power iteration for the
  largest eigenvalue of M⁻¹K, with K·x formed from the internal force under
  a trial displacement of fixed size, 1e-5 in model length units (10 μm on
  these meshes).  Once the iterate concentrates on a few nodes, that
  displacement inverts quadrature points of the 15-node elements, whose
  nodes are 0.1 to 0.4 mm apart; the estimate rises from 2.66e15 to
  6.6e17 s⁻², and with the projection the run stops on a negative volume.
  The same size makes the composite tetrahedron's first estimate 22% low at
  h = 0.75 mm, a step 13% larger than intended.
- A correction, not used in the runs above: the trial displacement is
  1e-5 of the local element length, and the iteration stops only when the
  remaining change extrapolated from successive changes is also below
  5e-4.  With the global estimate at factor 0.8, against the element
  estimate:

  | case | final radius (mm), global / element | steps |
  |---|---|---|
  | TET15-P1, h = 1.5 mm, 80 μs | 7.2349 / 7.2344 | 19 077 / 24 480 (−22%) |
  | TET15-P1, h = 0.75 mm, 80 μs | 7.2189 / 7.2186 | 43 389 / 50 832 (−15%) |
  | TETRA15, h = 1.5 mm, 80 μs | 7.1274 / 7.1268 | 20 232 / 22 358 (−10%) |
  | composite tetrahedron, h = 0.75 mm, 20 μs | 6.2511 / 6.2516 | 1 115 / 1 240 (−10%) |

  The power iterations add 3 to 5% to the number of steps.  The critical
  step 2/√λ, with λ from the converged global estimate, is 4.45, 6.24,
  3.10, 1.24 and 1.21 times the element estimate at 0, 5, 20, 40 and 80 μs
  at h = 1.5 mm, and 4.58, 2.10, 2.48, 1.25 and 1.24 times at h = 0.75 mm.
- The element estimate is safe but conservative: the critical step of
  lumped-mass central differences is 1.00 times the smallest node distance
  over the wave speed on the reference element, 0.85 on a regular
  tetrahedron and 0.50 to 0.96 on distorted ones, hence the factor 0.5.  At
  1 μs the step is 7 times smaller than the composite tetrahedron's, and at
  h = 0.38 mm TET15-P1 takes 3 times as many steps; a sharper estimate is
  the main means of reducing the explicit cost.
- The shape functions equal those of Carina's `Tet{EnrichedLagrange,2}` to
  1.4e-15 after the node relabelling, in the Exodus TETRA15 node order.

## Results

`data/` holds the histories (time, radius, length in m) of every run of the
study and of the cost comparison, and `taylor-results.tsv`, the record of the
Carina runs; the Carina runs at h = 0.38 and the Sierra/SM runs were made on
a two-socket AMD EPYC 9634.  `plot.py` draws the final radius against the
number of elements and the radius against time from them.  The results and
the cost comparison are in Sec. "Explicit dynamics: the Taylor bar impact" of
`research/tet15-p1/note.pdf`.

## Performance across machines

`kernel_timing.jl` times, on the mesh of a level in a deformed state with
plastic flow (400 explicit steps from the impact, equivalent plastic strain up
to 1.0), the residual assembly, one explicit step and the element estimate of
the stable step, each the median of 20 repetitions with the device
synchronized, and the eigenvalue estimate of the stable step, the median of
5, each preceded by a garbage collection on a device; it estimates the time
per step of a run from them (step + element estimate / 10 + eigenvalue
estimate / 200).  One JSON record per run goes to `kernel-timing.jsonl` (this
machine) or `data/kernel-timing-<host>[-median5].jsonl`; `timing_table.py`
prints the table below from them, the newest record per host, device,
threads, h and form (`--all` prints every record).

Records made before Carina 4300630 timed the eigenvalue estimate once, right
after the warm-up, and overstated it: every estimate runs about 20 power
iterations, but the single timed estimate took 1.5 to 4.4 times the median
of five on the GPUs, and with the median the time per run step fell by 7 to
36%.  On a CPU the two agree: 459.6 ms single against 457.1 ms median on the
Sirius CPU.  The GPU rows below and the Sirius CPU rows are median-of-five
records of 2026-10-06 (Carina 36b4a9f with the corrected script, and
178e811 on the idle Sirius CPU); the Rigel CPU rows are the records of
2026-10-04 (Carina 9e4bfaf) with the single estimate.  Between days the time
of the explicit step on the Sirius CPU varies by up to 8%: alternating Carina
9e4bfaf and 178e811 with the same dependencies gave residuals of 9.47 to
9.93 ms and steps of 14.0 to 15.3 ms for both commits.

Conditions: ConstitutiveModels 273af9e, `OPENBLAS_NUM_THREADS=1`, each run
alone on its GPU, nothing pinned; the same h = 0.75 and 0.38 mm meshes on the
Rigel side (Rigel CPU, L4, V100, A100, H100) and on the Sirius side (CPU,
RX 7600); the two sides' h = 0.75 meshes are different smoothing runs of the
same Cubit mesh, which changes the radius history by 2.8e-7 relative, while
within a side every device gives the same history to every written digit.
Machines: Sirius, AMD Ryzen 9 9900X (12 cores) with an AMD RX 7600 (ROCm);
Rigel, 2 x AMD EPYC 9634 (168 cores) with an NVIDIA L4 (driver 615.71.09,
CUDA 13.4); ascicgpu22, Tesla V100-PCIE-32GB; ascicgpu073, A100-PCIE-40GB;
ascicgpu080, H100 80GB HBM3 (SXM) (driver 580.126.20, CUDA 13.0 on the three
ascicgpu hosts); Julia 1.13.0 everywhere.

| host | device | threads | h (mm) | form | elements | residual (ms) | step (ms) | element dt (ms) | global dt (ms) | per run step (ms) | us per element-step |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rigel | 2 x AMD EPYC 9634 | 16 | 0.38 | split | 190727 | 90.87 | 150.48 | 28.96 | 8720.49 | 196.98 | 1.033 |
| rigel | 2 x AMD EPYC 9634 | 32 | 0.38 | split | 190727 | 57.99 | 115.24 | 23.45 | 5943.29 | 147.30 | 0.772 |
| rigel | 2 x AMD EPYC 9634 | 64 | 0.38 | split | 190727 | 41.87 | 98.88 | 19.90 | 4682.35 | 124.28 | 0.652 |
| ascicgpu073 | NVIDIA A100-PCIE-40GB | 16 | 0.38 | split | 190727 | 13.99 | 15.16 | 0.93 | 652.00 | 18.51 | 0.097 |
| ascicgpu080 | NVIDIA H100 80GB HBM3 | 16 | 0.38 | split | 190727 | 7.42 | 8.04 | 0.46 | 375.87 | 9.96 | 0.052 |
| ascicgpu22 | NVIDIA V100-PCIE-32GB | 16 | 0.38 | split | 190727 | 31.20 | 33.02 | 1.59 | 1352.26 | 39.94 | 0.209 |
| rigel | NVIDIA L4 | 16 | 0.38 | split | 190727 | 40.30 | 45.10 | 2.42 | 1569.74 | 53.19 | 0.279 |
| rigel | 2 x AMD EPYC 9634 | 16 | 0.75 | general | 25029 | 21.27 | 30.75 | 3.94 | 945.33 | 35.87 | 1.433 |
| sirius | AMD Ryzen 9 9900X | 12 | 0.75 | general | 25029 | 15.45 | 20.46 | 1.50 | 782.81 | 24.53 | 0.980 |
| ascicgpu073 | NVIDIA A100-PCIE-40GB | 16 | 0.75 | general | 25029 | 5.04 | 5.43 | 0.18 | 213.67 | 6.51 | 0.260 |
| ascicgpu080 | NVIDIA H100 80GB HBM3 | 16 | 0.75 | general | 25029 | 3.24 | 3.48 | 0.11 | 142.19 | 4.20 | 0.168 |
| ascicgpu22 | NVIDIA V100-PCIE-32GB | 16 | 0.75 | general | 25029 | 12.24 | 12.80 | 0.31 | 509.64 | 15.38 | 0.615 |
| rigel | NVIDIA L4 | 16 | 0.75 | general | 25029 | 10.89 | 11.47 | 0.37 | 438.67 | 13.70 | 0.548 |
| sirius | AMD RX 7600 | 12 | 0.75 | general | 25029 | 12.27 | 13.20 | 0.43 | 410.57 | 15.29 | 0.611 |
| rigel | 2 x AMD EPYC 9634 | 16 | 0.75 | split | 25029 | 13.92 | 22.65 | 4.07 | 671.96 | 26.42 | 1.056 |
| rigel | 2 x AMD EPYC 9634 | 32 | 0.75 | split | 25029 | 7.64 | 15.10 | 2.97 | 422.89 | 17.52 | 0.700 |
| rigel | 2 x AMD EPYC 9634 | 64 | 0.75 | split | 25029 | 6.49 | 18.58 | 3.31 | 400.06 | 20.92 | 0.836 |
| sirius | AMD Ryzen 9 9900X | 12 | 0.75 | split | 25029 | 9.33 | 15.18 | 1.47 | 457.12 | 17.61 | 0.704 |
| ascicgpu073 | NVIDIA A100-PCIE-40GB | 16 | 0.75 | split | 25029 | 2.12 | 2.49 | 0.17 | 96.65 | 2.99 | 0.119 |
| ascicgpu080 | NVIDIA H100 80GB HBM3 | 16 | 0.75 | split | 25029 | 1.17 | 1.42 | 0.12 | 58.48 | 1.73 | 0.069 |
| ascicgpu22 | NVIDIA V100-PCIE-32GB | 16 | 0.75 | split | 25029 | 5.11 | 5.62 | 0.30 | 224.83 | 6.78 | 0.271 |
| rigel | NVIDIA L4 | 16 | 0.75 | split | 25029 | 6.08 | 6.79 | 0.39 | 240.97 | 8.03 | 0.321 |
| sirius | AMD RX 7600 | 12 | 0.75 | split | 25029 | 8.60 | 9.36 | 0.42 | 272.87 | 10.76 | 0.430 |

Full Taylor runs to 80 μs at h = 0.75 mm with the global stable step, wall
time of the whole run (setup included):

| Device | Carina | Wall time (s) |
|---|---|---|
| NVIDIA H100 | 36b4a9f | 159.4 |
| NVIDIA A100 | 9e4bfaf | 227.3 |
| NVIDIA V100 | 9e4bfaf | 403.6 |
| NVIDIA L4 | 9e4bfaf | 405.9 |
| AMD RX 7600 | 9e4bfaf | 513.1 |
| Sirius CPU, 12 threads | 9e4bfaf | 838.9 |
| Rigel CPU, 16 threads | 9e4bfaf | 1282.7 |

Observations on the GPUs (per run step, corrected records).  The H100 is
1.7 to 1.9 times faster than the A100; the A100 is 2.2 to 2.3 times faster
than the V100 and 2.7 to 2.9 times faster than the L4; the V100 is 1.18
times faster than the L4 at h = 0.75 mm and 1.33 times at h = 0.38 mm; the
RX 7600 is 1.34 times slower than the L4.  The time per element-step falls
on every GPU from h = 0.75 to 0.38 mm (H100 0.069 to 0.052 μs, A100 0.119 to
0.097, V100 0.271 to 0.209, L4 0.321 to 0.279): 25 029 elements do not fill
the larger GPUs.  The general form costs 1.42 times the split form per run
step on the RX 7600, 1.71 on the L4, 2.18 on the A100, 2.27 on the V100 and
2.43 on the H100; on the CPUs 1.36 (Rigel, 16 threads) and 1.39 (Sirius,
12 threads).  On the Rigel CPU the best thread
count is 32 at h = 0.75 mm and 64 at h = 0.38 mm.

The eigenvalue estimate.  In the Taylor runs at h = 0.75 mm every estimate
ran 20 to 40 power iterations (4495 in the 220 estimates of the fixed
interval of 200 steps), each two residual evaluations.  A fixed longer
interval is not safe: the ratio of the eigenvalue step to the element
estimate falls by up to 11% between two estimates 200 steps apart (8 to
30 μs) and by up to 23% within 1000 steps, more than the 20% margin of
CFL 0.8.  The interval set from the observed change (`--eigenvalue-change
0.02`, the key `stable time step eigenvalue change`) takes 52 estimates
(1104 iterations) instead of 220, with radius and length histories equal to
the fixed interval's to within 7.5e-7 relative on every device and the same,
frame for frame, on the L4, V100, A100 and H100:

| Device | Fixed interval (s) | Adaptive, c = 0.02 (s) | Reduction | Estimates' share of the fixed run |
|---|---|---|---|---|
| AMD RX 7600 | 499.7 | 443.4 | 11.3% | 14.8% |
| NVIDIA L4 | 449.0 | 399.5 | 11.0% | 14.4% |
| NVIDIA V100 | 401.7 | 366.6 | 8.7% | 11.4% |
| NVIDIA H100 | 159.4 | 155.3 | 2.6% | 3.4% |

Carina 44de54e (RX 7600) and 36b4a9f; the share is the reduction divided by
the fraction of estimates removed, 168/220.  The A100 runs of 2026-10-06 are
not used: another user's CPU jobs loaded the host (load average 24 to 33,
`data/taylor-results-nvidia-notes.txt`).  The H100 reduction of 4.1 s is
within the variation between runs, and its share is not a precise figure.
Records: `data/taylor-results-nvidia.tsv`, `data/taylor-h0.75-<device>-eig-
{fixed,change0.02}-history.tsv`.

## Implicit kernels across machines

`implicit_timing.jl` times, at the same deformed state (h = 0.75 mm, 400
explicit steps; internal variables of the step before as the old state), the
residual, the matrix-free action of the tangent, its diagonal (the Jacobi
preconditioner) and, on a CPU, the assembled tangent, for the split and the
general form (J2, θ = J − 1), and the general residual with the material
called again in the third pass instead of storing its fourteen stresses
(`recompute`).  Median of 20, ms, Carina 36b4a9f (Sirius 44de54e, the
same kernels; the Sirius CPU row at 178e811 on the idle machine); records in
`data/implicit-timing-<host>.jsonl`.

| Device | split residual | split action | split diagonal | general residual | general action | general diagonal | recompute |
|---|---|---|---|---|---|---|---|
| Sirius CPU, 12 threads | 11.71 | 65.84 | 69.44 | 19.52 | 76.20 | 86.71 | 25.31 |
| AMD RX 7600 | 8.95 | 20.81 | 21.60 | 12.59 | 30.35 | 53.29 | 16.99 |
| NVIDIA L4 | 6.15 | 18.67 | 22.95 | 10.89 | 27.31 | 54.46 | 16.10 |
| NVIDIA V100 | 5.32 | 15.38 | 20.11 | 12.30 | 23.70 | 101.40 | 10.71 |
| NVIDIA A100 | 3.01 | 9.41 | 13.56 | 7.73 | 16.09 | 43.23 | 5.82 |
| NVIDIA H100 | 1.17 | 3.71 | 4.37 | 3.26 | 6.33 | 25.06 | 2.42 |

The assembled tangent on the Sirius CPU: 138.0 ms split, 131.8 ms general.
Calling the material again in the third pass is slower than storing the
stresses on the CPU (1.30 times), the RX 7600 (1.35) and the L4 (1.48), and
faster on the V100 (0.87), the A100 (0.75) and the H100 (0.74); Carina
stores them.  The diagonal of the general form then took 2.0 (L4), 1.8
(RX 7600), 2.7 (A100), 4.0 (H100) and 4.3 (V100) times its action.

### The general-form diagonal restructured, and inlining

Carina 5f9b897 with ConstitutiveModels 252194c restructures the diagonal and
nodal-block kernels of the general form: the coupling matrix D is formed
from the geometry before the material loop, each quadrature point's
contribution is added to the diagonal at that point (no stored second
pass), and the material returns stress and tangent from one call.
`--always-inline` (Carina 096c0e2, CUDA only) compiles the kernels with
CUDA.jl's `always_inline`, which inlines every device-function call; it is
applied by redefining `KA.get_backend(::CuArray)` in the timing process,
because FiniteElementContainers launches its kernels with the backend of its
arrays.  General-form diagonal and action, ms (records in `data/`,
`implicit-timing-<host>-{diag-v2,diag-v2-ref,inline-e2c}.jsonl`):

| Device | diagonal: 36b4a9f | 5f9b897 | + always_inline | action: 36b4a9f | 5f9b897 | + always_inline |
|---|---|---|---|---|---|---|
| Ryzen 9 9900X, 12 threads | 82–88 | 81–90 | — | 71–76 | 60–69 | — |
| AMD RX 7600 | 53.1 | 38.9 | — | 30.3 | 26.8 | — |
| NVIDIA L4 | 54.46 | 58.15 | 41.53 | 27.31 | 25.71 | 21.25 |
| NVIDIA V100 (ascicgpu16) | 112.44 | 34.63 | 21.03 | 22.98 | 22.78 | 12.58 |
| NVIDIA A100 | 43.23 | 19.99 | 11.13 | 16.09 | 15.06 | 6.49 |
| NVIDIA H100 | 25.06 | 8.94 | 4.97 | 6.33 | 6.06 | 3.55 |

The `always_inline` runs used ConstitutiveModels 5fd1082 (J2 methods marked
inline, not kept); with `always_inline` every call is inlined, so that
marking does not change the kernels.  The V100 rows are from one GPU of
ascicgpu16 for all three columns (V100-PCIE-16GB, the GV100 of the reference
card).  Registers per thread (`CUDA.registers`): the former general
diagonal 255 (L4) and 32 (H100, local frame 50 KB per thread; not counted
on the V100 and the A100); the restructured one 128 on every card; with `always_inline` 128
and a 13.7 KB frame on every card.  `always_inline` also shortens the split
kernels: action 13.98 -> 4.59 (V100), 9.24 -> 2.71 (A100), 3.68 -> 1.17
(H100), 18.63 -> 10.93 ms (L4); diagonal 19.31 -> 11.68, 13.55 -> 7.06,
4.41 -> 3.17 and 23.12 -> 24.17 ms.  Marking only the J2 methods inline
(ConstitutiveModels 5fd1082) gave the general diagonal 32 registers on the
A100 (19.99 -> 28.69 ms) and is not kept.  The launcher (`bin/carina`) now
compiles the CUDA kernels with `always_inline` through
`bin/cuda_always_inline.jl`, and so do run.jl, kernel_timing.jl and
implicit_timing.jl (`--no-always-inline` restores CUDA.jl's default in
implicit_timing.jl).  AMDGPU.jl compiles with `always_inline = true` by
default, which is why marking the J2 methods inline changed nothing on the
RX 7600 (general diagonal 38.92 against 38.9 ms).

