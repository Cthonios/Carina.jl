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
