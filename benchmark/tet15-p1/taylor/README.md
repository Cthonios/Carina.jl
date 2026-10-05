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

## Composite tetrahedron (Sierra/SM)

On the same `meshes/taylor-h<h>-tet10.g`: the explicit dynamics of Sierra/SM
with the composite tetrahedron of Foulk et al. (2021) at its default
stabilization α = 0.1, the J2 model with linear hardening of the same
constants, the initial velocity on node set `all`, u_z = 0 on node set
`impact`, the symmetry conditions on the quarter bar, output every 1 μs to
80 μs.  The same `history.tsv` is extracted from its output.

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
to 1.0), the residual assembly, one explicit step, the element estimate of the
stable step and the eigenvalue estimate, each the median of 20 repetitions
with the device synchronized, and estimates the time per step of a run from
them (step + element estimate / 10 + eigenvalue estimate / 200).  One JSON
record per run goes to `kernel-timing.jsonl` (this machine) or
`data/kernel-timing-<host>.jsonl`; `timing_table.py` prints the table below
from them.  All records: Carina 9e4bfaf, ConstitutiveModels 273af9e,
`OPENBLAS_NUM_THREADS=1`, each run alone on its machine, nothing pinned, the
same h = 0.75 and 0.38 mm meshes on the Rigel side (Rigel, L4, V100, A100) and
the same on the Sirius side (CPU, RX 7600); the two sides' h = 0.75 meshes are
different smoothing runs of the same Cubit mesh, which changes the radius
history by 2.8e-7 relative, while within a side every device gives the same
history to every written digit.  Machines: Sirius, AMD Ryzen 9 9900X (12
cores) with an AMD RX 7600 (ROCm); Rigel, 2 x AMD EPYC 9634 (168 cores) with
an NVIDIA L4 (driver 615.71.09, CUDA 13.4); ascicgpu22, Tesla V100-PCIE-32GB
and ascicgpu073, A100-PCIE-40GB (driver 580.126.20, CUDA 13.0); Julia 1.13.0
everywhere.

| host | device | threads | h (mm) | form | elements | residual (ms) | step (ms) | element dt (ms) | global dt (ms) | per run step (ms) | us per element-step |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rigel | 2 x AMD EPYC 9634 | 16 | 0.38 | split | 190727 | 90.87 | 150.48 | 28.96 | 8720.49 | 196.98 | 1.033 |
| rigel | 2 x AMD EPYC 9634 | 32 | 0.38 | split | 190727 | 57.99 | 115.24 | 23.45 | 5943.29 | 147.30 | 0.772 |
| rigel | 2 x AMD EPYC 9634 | 64 | 0.38 | split | 190727 | 41.87 | 98.88 | 19.90 | 4682.35 | 124.28 | 0.652 |
| ascicgpu073 | NVIDIA A100-PCIE-40GB | 16 | 0.38 | split | 190727 | 14.00 | 15.32 | 0.90 | 1465.92 | 22.74 | 0.119 |
| ascicgpu22 | NVIDIA V100-PCIE-32GB | 16 | 0.38 | split | 190727 | 31.18 | 33.04 | 1.60 | 2881.79 | 47.61 | 0.250 |
| rigel | NVIDIA L4 | 16 | 0.38 | split | 190727 | 40.17 | 44.90 | 2.41 | 3268.32 | 61.48 | 0.322 |
| rigel | 2 x AMD EPYC 9634 | 16 | 0.75 | general | 25029 | 21.27 | 30.75 | 3.94 | 945.33 | 35.87 | 1.433 |
| sirius | AMD Ryzen 9 9900X | 12 | 0.75 | general | 25029 | 15.51 | 22.58 | 1.56 | 799.73 | 26.73 | 1.068 |
| ascicgpu073 | NVIDIA A100-PCIE-40GB | 16 | 0.75 | general | 25029 | 5.04 | 5.43 | 0.18 | 463.29 | 7.77 | 0.310 |
| ascicgpu22 | NVIDIA V100-PCIE-32GB | 16 | 0.75 | general | 25029 | 12.27 | 12.76 | 0.30 | 776.30 | 16.67 | 0.666 |
| rigel | NVIDIA L4 | 16 | 0.75 | general | 25029 | 10.91 | 11.42 | 0.37 | 640.30 | 14.66 | 0.586 |
| sirius | AMD RX 7600 | 12 | 0.75 | general | 25029 | 13.51 | 14.37 | 0.42 | 679.14 | 17.81 | 0.712 |
| rigel | 2 x AMD EPYC 9634 | 16 | 0.75 | split | 25029 | 13.92 | 22.65 | 4.07 | 671.96 | 26.42 | 1.056 |
| rigel | 2 x AMD EPYC 9634 | 32 | 0.75 | split | 25029 | 7.64 | 15.10 | 2.97 | 422.89 | 17.52 | 0.700 |
| rigel | 2 x AMD EPYC 9634 | 64 | 0.75 | split | 25029 | 6.49 | 18.58 | 3.31 | 400.06 | 20.92 | 0.836 |
| sirius | AMD Ryzen 9 9900X | 12 | 0.75 | split | 25029 | 8.95 | 14.12 | 1.52 | 459.57 | 16.57 | 0.662 |
| ascicgpu073 | NVIDIA A100-PCIE-40GB | 16 | 0.75 | split | 25029 | 2.13 | 2.52 | 0.17 | 338.09 | 4.23 | 0.169 |
| ascicgpu22 | NVIDIA V100-PCIE-32GB | 16 | 0.75 | split | 25029 | 5.11 | 5.62 | 0.30 | 496.23 | 8.13 | 0.325 |
| rigel | NVIDIA L4 | 16 | 0.75 | split | 25029 | 6.06 | 6.82 | 0.38 | 457.04 | 9.15 | 0.365 |
| sirius | AMD RX 7600 | 12 | 0.75 | split | 25029 | 8.61 | 9.41 | 0.41 | 593.73 | 12.41 | 0.496 |

Full Taylor runs to 80 μs at h = 0.75 mm with the global stable step, wall
time of the whole run (setup included):

| Device | Wall time (s) |
|---|---|
| NVIDIA A100 | 227.3 |
| NVIDIA V100 | 403.6 |
| NVIDIA L4 | 405.9 |
| AMD RX 7600 | 513.1 |
| Sirius CPU, 12 threads | 838.9 |
| Rigel CPU, 16 threads | 1282.7 |

Observations.  Per run step the A100 is 1.9 to 2.1 times faster than the
V100 and 2.2 to 2.7 times faster than the L4; the V100 and the L4 are within
11% of each other at h = 0.75 mm, and the V100 is 1.3 times faster at
h = 0.38 mm; the RX 7600 is 1.4 times slower than the L4.  The time per
element-step falls on every GPU from h = 0.75 to 0.38 mm (A100 0.169 to
0.119 μs, V100 0.325 to 0.250, L4 0.365 to 0.322): 25 029 elements do not
fill the larger GPUs.  On the Rigel CPU the best thread count is 32 at
h = 0.75 mm and 64 at h = 0.38 mm; at 64 threads the explicit step at
h = 0.75 mm is slower than at 32 although the residual alone is faster.  The
Sirius CPU with 12 threads (0.66 μs per element-step) is faster than the
Rigel CPU with 32 (0.70).  The general form costs 1.36 times the split form
per run step on the CPUs, 1.44 on the RX 7600, 1.60 on the L4, 1.84 on the
A100 and 2.05 on the V100.  The eigenvalue estimate of the stable step costs
25 to 80 run steps and runs every 200, so it takes 13% of the stepping time
on the 16-thread CPU, 25% on the L4, 31% on the V100 and 40% on the A100: on
a GPU the interval `stable time step eigenvalue interval` should be raised,
which the ratio of the two estimates, varying by 1% per 200 steps late in
the run, permits.

