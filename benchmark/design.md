# Design: native-Julia GPU preconditioning for Carina's implicit paths

**This is the design document written *before* the work, kept as the record of
what was considered and why.  For what was actually built and measured, see
`benchmark_report.md`; where the two disagree, the report wins.  An Outcome
section at the end notes where this document's expectations were wrong.**

Constraints: pure Julia, zero TPLs,
vendor-agnostic (KernelAbstractions), matrix-free or otherwise GPU-native.
Targets: quasistatic and Newmark implicit dynamics on p=1 unstructured
hex/tet meshes, 0.5M–8M DOF.

## Problem statement

The GPU implicit path today is matrix-free CG with Jacobi or Chebyshev-Jacobi
preconditioning.  Prior measurements (July 2026, torsion 530k DOF):

- Newmark dt=5e-5, CG+Jacobi: ~22 s/step — usable but linear iteration counts
  grow with dt (mass shift c_M = 1/(β·Δt²) weakens as dt grows).
- Quasistatic: Jacobi-preconditioned CG iteration counts run into the many
  hundreds; L-BFGS stalls near |r| ≈ 1.7e-3; NLCG converges but slowly.
  On CPU, AMG (smoothed aggregation + rigid-body near-nullspace from the
  current configuration + staleness-lagged rebuilds) flattens linear
  iterations to a few dozen.  AMG is CPU-only in Carina.

The missing piece is a GPU-resident preconditioner with AMG-like iteration
counts, without vendor sparse libraries.

## Candidates considered

**(a) Host-built SA-AMG hierarchy, device-resident V-cycle apply — CHOSEN.**
Reuse the entire existing CPU setup machinery verbatim (assembly on `asm_cpu`,
`AlgebraicMultigrid.smoothed_aggregation` with rigid-body modes from the
current configuration, the c_M/staleness lazy-rebuild logic).  Convert the
hierarchy's levels to device CSR once per (re)build and run the V-cycle
entirely on the GPU with KernelAbstractions kernels.  Setup stays on the
host where the sparse pattern already lives (`as_matrix_free` strips it from
the device copy precisely because the device never needed it — the
preconditioner hierarchy is a different, much smaller object).

**(b) Fully assembled AMG on device.**  Same as (a) but fine-level smoothing
through assembled fine A on the device.  Rejected as the default: fine CSR at
7.8M DOF is ~7 GB — most of the VRAM budget; and Carina already has a
zero-allocation matrix-free fine-level action.

**(c) Geometric agglomeration multigrid, fully matrix-free.**  No assembled
levels at all.  Rejected for this round: requires mesh-hierarchy machinery
Carina does not have, unknown robustness on unstructured tet meshes near
incompressibility, and it discards the proven SA + near-nullspace setup.
Revisit if (a)'s setup cost or hierarchy VRAM becomes the bottleneck.

**(d) Polynomial-only preconditioning (higher-degree Chebyshev, s-step).**
Already partially in place; measured insufficient on quasistatic (iteration
counts still grow with problem size — no coarse-grid correction).  Kept as
the smoother inside (a) instead.

**(e) Linear-solve-free nonlinear methods (L-BFGS, NLCG).**  Measured at the
time: L-BFGS ~13x slower than CG+Jacobi on Newmark, stalls on quasistatic;
NLCG needs hundreds of iterations.  Not competitive as primary solvers.
(The Newmark half of this was later overturned — see Outcome.)

## Chosen architecture

```
CG (Krylov.jl, device vectors)                        [exists]
 |- operator: matrix-free K_eff action (KA kernels)   [exists, zero-alloc]
 |- preconditioner: one V(2,2)-cycle per application  [NEW]
      level 1 (fine):    smoothing via MATRIX-FREE action + diag(K_eff)
                         (Chebyshev-Jacobi, degree 2-3)  [action+diag exist]
      levels 2..L:       assembled device CSR A_l, P_l, R_l
                         Chebyshev-Jacobi smoothing      [NEW: KA CSR SpMV]
      coarsest:          dense pinv(A_L) as device matvec [NEW: small GEMV]
```

- **Hybrid fine level.**  The V-cycle's fine-level smoothing and residual use
  the existing matrix-free action, so the fine CSR is never formed on the
  device.  Only levels ≥ 2 ship as CSR: SA operator complexity ≈ 1.3–1.6,
  i.e. the device hierarchy costs ~0.3–0.6× the fine-matrix memory it avoids,
  plus P/R (1–3 nnz per fine row).
- **Smoothers.**  Chebyshev-Jacobi everywhere (AMG.jl's default Gauss-Seidel
  is sequential and unusable on GPU).  Eigenvalue bounds per level via the
  existing power-iteration estimator on the symmetrically scaled operator.
- **Kernels to write (all bandwidth-bound, vendor-agnostic KA):** CSR SpMV,
  fused axpby/diag-scale, restriction/prolongation SpMV (CSR and transpose
  stored explicitly — no atomics), small dense GEMV for the coarse solve.
  Zero allocations in apply (preallocated per-level workspaces).
- **Setup path (host, unchanged logic):** assemble K_eff on `asm_cpu`,
  symmetrize, `_rigid_body_modes(x_cur)`, `AMG.smoothed_aggregation`, then
  NEW: hierarchy → device conversion (CSC→CSR transpose + upload, Int32
  indices).  Rebuilds keep the existing triggers (c_M drift, iteration-growth
  staleness detector).
- **Failure containment:** if the V-cycle apply produces a non-finite value
  the preconditioner poisons the CG residual, which the existing isfinite
  machinery turns into a step failure — same contract as the eversion guard.

## Cost model / win condition

Per CG iteration, V(2,2) with matrix-free fine smoothing costs ≈ 5 fine
operator actions (2+2 smoothing + 1 residual) + coarse-level work (~0.3–0.6
fine-action-equivalents) ≈ 6× the per-iteration cost of CG+Jacobi.  Win
condition: ≥ 6× iteration reduction.

Quasistatic torsion measured Jacobi-CG iteration counts in the several
hundreds per Newton step; CPU AMG brings equivalent systems to a few dozen —
an expected 10–30× reduction ⇒ projected 2–5× wall-clock gain, growing with
problem size (Jacobi degrades with N, AMG does not).  Newmark at small dt is
mass-dominated and Jacobi-CG already converges in tens of iterations; AMG is
expected roughly cost-neutral there — Jacobi remains the recommended Newmark
default and the benchmark must show this honestly.

Bandwidth accounting for the report: count bytes moved per fine action
(element coordinates, connectivity, field, properties) and per CSR SpMV
(values + colind + x/y), divide by measured times, compare against device
peak.  (This expectation did not survive contact with the hardware — the
development GPU is a consumer RX 7600, not an MI-series part, and the action
turned out to be compute-bound rather than bandwidth-bound.  See Outcome.)

## Deliverables

1. `src/` implementation: device hierarchy type + KA kernels + V-cycle apply,
   wired as `preconditioner: type: amg` on the GPU path (parse-time guard
   removed), with loud failure on unsupported combinations.
2. Tests: V-cycle apply vs CPU AMG apply agreement on a small mesh; CG+AMG
   GPU vs CPU solution agreement; staleness/rebuild behavior on device.
3. `benchmark_report.md` per the benchmark gate.

## Outcome

Built and measured; `benchmark_report.md` is the record.  Three places where
this document guessed wrong, kept here because the reasoning is instructive:

- **The roofline was the wrong one.**  This document assumed an MI-series HBM
  part and a bandwidth-bound fine action, and planned the report's hardware
  section around byte accounting.  The development machine is a consumer
  Radeon RX 7600 (288 GB/s, FP64 at 1/32 rate), and direct ablation later
  showed the matrix-free action is **FP64-compute-bound**, not
  bandwidth-bound: ~73% of it was forming the fourth-order material tangent,
  and the FP64 atomic scatter — long suspected as the culprit — costs nothing
  measurable.  Replacing the tangent with a directional derivative gave 3.2x
  on the kernel.
- **The win condition was met, but for a different reason than projected.**
  The cost model predicted V(2,2) at ~6x a Jacobi iteration needing a >=6x
  iteration reduction; both held (measured 6.67x per application, ~16x fewer
  iterations).  The projected "2-5x wall-clock gain" was initially not
  realised — GPU AMG merely tied CPU AMG — because the fine action was
  leaving 3x on the table.  With that fixed the projection holds.
- **L-BFGS on Newmark reversed.**  Candidate (e) recorded L-BFGS as ~13x
  slower than CG+Jacobi on Newmark.  After the zero-allocation fixes it became
  competitive there.  It still stalls on quasistatic, so the conclusion that
  it cannot be the general answer stands.

Unchanged and vindicated: the hybrid fine level (never forming the fine matrix
on device), host-side setup, rigid-body modes from the current configuration,
lagged rebuilds, and Chebyshev-Jacobi smoothing in place of Gauss-Seidel.
