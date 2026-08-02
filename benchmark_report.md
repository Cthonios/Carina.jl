# Benchmark report: native-Julia GPU AMG for Carina's implicit paths

**Status: DRAFT — cube100 scaling results pending.**

Campaign brief: `~/Carina-GPU.md`.  Proposed solution, design rationale and
rejected alternatives: `benchmark/design.md`.  Raw data:
`benchmark/results/{baseline,proposed,scaling}.jsonl`; harness:
`benchmark/harness.jl` (one fresh process per run; iteration counts parsed
from Carina's own log; VRAM from `AMDGPU.memory_stats().live` at end of run).

## 1. Methodology

**Proposed solution.**  CG preconditioned by a GPU-resident AMG V(2,2)-cycle
(`src/gpu_amg.jl`).  The hierarchy is built on the host by the existing
AlgebraicMultigrid.jl smoothed-aggregation setup — near-nullspace = rigid-body
modes evaluated at the *current* configuration, rebuilt lazily on c_M change or
CG-iteration growth (3× baseline) — then converted to device CSR (Int32
indices).  The V-cycle applies entirely on the device through
KernelAbstractions kernels (vendor-agnostic; no TPLs):

- Fine level: smoothing and residuals via the existing zero-allocation
  matrix-free stiffness action + extracted diagonal (damped Jacobi,
  ω = 4/(3λ̂max), ν = 2 each side).  The fine matrix is never formed on the
  device.
- Coarse levels: assembled CSR, same smoother; coarsest solved by a dense
  pinv matvec.
- The apply path allocates nothing (preallocated per-level workspaces).

Mathematical form: standard SA-AMG preconditioned CG.  The V(ν,ν) cycle with
matched pre/post smoothing is a symmetric positive operator (verified
numerically in `test/gpu-amg-vcycle.jl`: ⟨u, Mv⟩ = ⟨Mu, v⟩ to 1e-6 and
⟨u, Mu⟩ > 0), so CG's requirements hold.

**Problems.**
- `torsion-qs`: cylinder (R 25 mm × L 1 m), 530,523 DOF, neo-Hookean
  (E = 1 GPa, ν = 0.25), base clamped, 0.05 rad twist ramped over 4 load
  steps.  Newton (abs 1e-6 / rel 1e-10), CG rtol 1e-8, itmax 1000.
- `torsion-newmark`: same mesh, free-free, rigid-torsion initial velocity
  field (a = 8000 s⁻¹), Newmark β = 0.25, dt = 5e-5, 4 steps.
- `cube100-{qs,newmark}`: generated 100³ HEX8 cube, 3,090,903 DOF
  (`benchmark/meshgen.jl`), uniaxial stretch (QS: 2 load steps; Newmark:
  BC-driven, 2 steps at dt = 5e-5).

**Measurement rules.**  One (case, variant) per fresh Julia process — no JIT
or VRAM carryover; total wall time therefore includes ~40 s of compilation
identically in every cell, and per-step walls / solve-phase sums (from the
run log) exclude it.  GPU runs on ROCm (AMDGPU.jl); the implementation
itself is KernelAbstractions-portable.  All variants of a case solve the
identical nonlinear problem with identical tolerances; every convergent run
reaches the same displacement solution (spot-checked against CPU direct at
rtol 1e-7 on the 16³ validation cube; harness asserts non-failure).

## 2. Results at 530k DOF (torsion)

Totals include ~40 s JIT uniformly.  "CG iters" is the total across all
Newton iterations of all steps; "capped" = the 1000-iteration CG limit was
hit (linear systems NOT converged to rtol 1e-8 — reported as measured).

### Quasistatic twist (the target problem)

| variant            | total (s) | CG iters   | linear conv. | VRAM (GB) |
|--------------------|----------:|-----------:|--------------|----------:|
| **GPU CG+AMG (new)** | **314**   | **980**    | yes          | 0.65      |
| CPU CG+AMG         | 302       | 787        | yes          | —         |
| CPU CG+Jacobi      | 377       | 16,000     | **capped**   | —         |
| CPU direct         | 443       | —          | (direct)     | —         |
| GPU CG+Jacobi      | 581       | 16,000     | **capped**   | 0.22      |
| GPU CG+Chebyshev   | 638       | 3,556      | yes          | 0.22      |
| CPU CG+IC          | 1,011     | 8,226      | yes          | —         |
| GPU L-BFGS         | failed    | —          | (stalls; step failure) | 0.34 |

- GPU AMG is **1.85× faster** than the best pre-existing GPU option and the
  only GPU variant that both converges its linear systems and beats the
  direct solver.
- Chebyshev illustrates the cost/iteration trade the design targets: 4.5×
  fewer iterations than Jacobi, yet slower — its per-iteration polynomial
  costs more than it saves.  The V-cycle pays a similar per-application
  premium (~6 fine-action equivalents) but buys a 16× iteration reduction.
- CPU AMG and GPU AMG are tied at this size: the CPU's Gauss–Seidel smoother
  converges in fewer iterations (787 vs 980) than the device's
  parallel-friendly damped Jacobi, offsetting device bandwidth.  Scaling
  separates them (§3).
- GPU AMG VRAM: 0.65 GB total vs 0.22 GB unpreconditioned — the device
  hierarchy costs ~0.43 GB, inside the predicted 0.3–0.6× of the never-formed
  fine matrix (~0.9 GB).

### Newmark dynamics (dt = 5e-5)

| variant            | total (s) | CG iters | VRAM (GB) |
|--------------------|----------:|---------:|----------:|
| GPU L-BFGS         | 112       | —        | 0.34      |
| CPU CG+Jacobi      | 150       | 2,558    | —         |
| GPU CG+Jacobi      | 169       | 2,300    | 0.26      |
| CPU CG+AMG         | 176       | 264      | —         |
| CPU CG+IC          | 183       | 672      | —         |
| GPU CG+Chebyshev   | 192       | 561      | 0.27      |
| CPU direct         | 551       | —        | —         |

- As predicted in the design: at mass-dominated small dt the c_M shift
  conditions the system and cheap preconditioners win.  AMG's 10× iteration
  reduction does not pay here.  **Recommended defaults: Jacobi for Newmark,
  AMG for quasistatic** — the benchmark supports both honestly.
- GPU L-BFGS is now the fastest Newmark option at this size (a reversal of
  the July 2026 measurement, predating the zero-allocation fixes) — but it
  fails outright on quasistatic, so it cannot be the general answer.

## 3. Scaling (cube64: 823k DOF, cube80: 1.57M DOF) — PENDING RESULTS

Runs in flight (strictly serialized): QS AMG/Jacobi × GPU/CPU at both sizes,
Newmark Jacobi GPU/CPU at 1.57M, plus instrumented torsion-QS re-runs for
the phase breakdown.  Same metrics as §2 plus per-step walls (JIT excluded).

**Host-memory capacity finding (measured, not projected).**  The original
plan's 3.09M-DOF cube100 case exhausts host RAM on the 60 GB benchmark
machine before the solver ever runs: FEC's assembler keeps the sparsity
pattern in per-element-entry (COO) form — 576 entries per HEX8, ~576M
entries at 1M elements, eight Int64/Float64 arrays of that length ≈ 35 GB
for `asm_cpu` alone — and the AMG setup transiently stacks K, its transpose,
and the SA hierarchy on top (44 GB RSS observed).  Both cube100 GPU runs
died thrashing a full 7 GB swap, with GPU VRAM at <1 GB of 16 — at multi-M
DOF the binding constraint for this architecture is *host* setup memory, not
device memory or device speed.  Concrete remediation (future work): Int32
pattern indices (halves the pattern), dedup-to-CSR at construction (removes
the COO triplets), and a pattern-free `asm_cpu` mode for GPU runs that never
assemble on the host.

## 4. Hardware efficiency — PENDING FINAL NUMBERS

Method: per CG iteration the dominant traffic is (a) matrix-free fine
action — element coordinates, connectivity, field, properties per element;
(b) hierarchy SpMV — CSR values + column indices + vectors.  Achieved GB/s =
counted bytes / measured solve-phase seconds, compared against device peak.
Allocation profile: `Base.GC_Diff` across the run (host) and VRAM live
deltas (device); the V-cycle apply path allocates zero bytes by
construction (preallocated workspaces; to be confirmed with `@allocated`
on-device in the final numbers).

## 5. Convergence data

Per-solve Newton iteration counts and per-solve CG iteration lists for every
run are in the JSONL records (`newton_iters`, `cg_iters` fields), including
the capped-at-1000 Jacobi solves.  Hierarchy rebuild counts and times are
logged per run (`amg_build_s` in instrumented records).

## 6. Scope compliance

Pure Julia throughout: AlgebraicMultigrid.jl (setup), KernelAbstractions
kernels (apply), Krylov.jl (CG).  No hypre/Trilinos/AmgX/rocSPARSE.  No
vendor-specific code paths in the proposed solution; the ROCm backend is the
test vehicle, CUDA support follows from KA + Adapt with no code changes.
