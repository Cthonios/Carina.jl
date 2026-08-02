# Benchmark report: native-Julia GPU AMG for Carina's implicit paths

**Status: PASSED Critic review (3 rounds: REVISE, REVISE, PASS; verdicts and
resolutions tracked in the gpu-amg branch history, commits d209ecd/094ca3b/b9f82e4).**

Campaign brief: `~/Carina-GPU.md`.  Proposed solution, design rationale and
rejected alternatives: `benchmark/design.md`.  Raw data:
`benchmark/results/{baseline,proposed,scaling2,variance,detail,bisect,nbuilds-check}.jsonl`
plus the evidence appendix `benchmark/evidence/`; harness:
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
- `cube64-qs` / `cube80-qs`: generated 64³ / 80³ HEX8 cubes
  (`benchmark/meshgen.jl`), uniaxial stretch over 2 load steps; cube80
  Newmark BC-driven, 2 steps at dt = 5e-5.  (The planned 3.09M-DOF cube100
  could not run on this host — §3.)

DOF convention: table labels are mesh totals (torsion 530,523; cube64
823,875; cube80 1,594,323); the JSONL `n_dofs` field records free DOFs
after BC elimination (torsion-qs 527,877; cube64 806,975; cube80
1,568,079).

**Measurement rules.**  One (case, variant) per fresh Julia process — no JIT
or VRAM carryover; total wall time therefore includes ~40 s of compilation
identically in every cell, and per-step walls / solve-phase sums (from the
run log) exclude it.  The three headline torsion-QS cells were measured
3× or more (2× for CPU Jacobi); tables report the mean, with observed
run-to-run spread (max−min over mean) of 0.8% (GPU Jacobi: 581/576/577),
8.0% (GPU AMG: 314/299/291/290, n=4 — the
larger spread tracks a small CG-iteration difference, 980 vs 951,
consistent with atomics nondeterminism in the device diagonal
extraction), and 2.3% (CPU Jacobi: 377/368).  Single-shot cells carry
that uncertainty.  GPU runs on ROCm (AMDGPU.jl); the implementation
itself is KernelAbstractions-portable.  All variants of a case solve the
identical nonlinear problem with identical tolerances.  Solution agreement
is a checked-in artifact: the "GPU AMG preconditioner" testset in
`test/mechanics-gpu-device.jl` solves a 16³ generated cube (14,739 DOF,
multilevel hierarchy asserted) with device CG+AMG and requires agreement
with CPU direct at rtol 1e-7; its ROCm output is recorded at
`benchmark/evidence/gpu_amg_test_rocm.txt`.  The harness asserts
non-failure on every run.

## 2. Results at 530k DOF (torsion)

Totals include ~40 s JIT uniformly.  "CG iters" is the total across all
Newton iterations of all steps; "capped" = the 1000-iteration CG limit was
hit (linear systems NOT converged to rtol 1e-8 — reported as measured).

### Quasistatic twist (the target problem)

| variant            | total (s) | CG iters   | linear conv. | VRAM (GB) |
|--------------------|----------:|-----------:|--------------|----------:|
| **GPU CG+AMG (new)** | **299** (n=4) | **951–980** | yes          | 0.65      |
| CPU CG+AMG         | 302       | 787        | yes          | —         |
| CPU CG+Jacobi      | 372 (n=2) | 16,000     | **capped**   | —         |
| CPU direct         | 443       | —          | (direct)     | —         |
| GPU CG+Jacobi      | 578 (n=3) | 16,000     | **capped**   | 0.22      |
| GPU CG+Chebyshev   | 638       | 3,556      | yes          | 0.22      |
| CPU CG+IC          | 1,011     | 8,226      | yes          | —         |
| GPU L-BFGS         | failed    | —          | (stalls; step failure) | n/a\* |

\* The failed L-BFGS run writes no JSONL record; its failure log is
`benchmark/evidence/torsion_qs_lbfgs_failure.txt`.

- GPU AMG is **1.9× faster** (578/299, means over repeats) than the best pre-existing GPU option and the
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

## 3. Scaling (823k and 1.57M DOF)

Quasistatic uniaxial stretch, totals include ~40 s JIT uniformly.
(The stretch is better conditioned than the torsion twist, so iteration
counts are lower across the board; the *relative* trends are the point.)

| size            | GPU AMG (new)   | GPU Jacobi      | CPU AMG        | CPU Jacobi      |
|-----------------|-----------------|-----------------|----------------|-----------------|
| 530k (torsion)  | 299 s / 966 (n=4) | 578 s / 16,000c (n=3) | 302 s / 787 | 372 s / 16,000c (n=2) |
| 823k (cube64)   | 167 s / 118     | 290 s / 2,670   | 164 s / 100    | 169 s / 2,670   |
| 1.57M (cube80)  | **249 s / 136** | 439 s / 3,320   | **OOM**        | 312 s / 3,320   |

("c" = CG iteration cap hit; linear systems not converged to tolerance.)

- **At 1.57M DOF the proposed solver is the fastest quasistatic option on
  the machine and the only AMG that runs at all.**  Stock
  AlgebraicMultigrid setup is OOM-killed there (see below); the proposed
  build survives because its fine-level Galerkin product is evaluated in
  column slabs (`_slab_galerkin`, peak transient = one slab).
- Jacobi iteration counts grow with size (2,670 → 3,320) while AMG's stay
  flat (118 → 136 — h-independence as designed).
- Newmark at 1.57M (dt = 5e-5): GPU Jacobi 166 s vs CPU Jacobi 171 s — the
  first size where the GPU edges ahead on dynamics; AMG remains
  unnecessary there.

**Host-memory findings (measured, not projected).**  Two distinct walls:

1. *Assembler pattern:* FEC keeps the sparsity pattern in per-element-entry
   (COO) form — 576 entries per HEX8 — so `asm_cpu` alone is 25.3 GB at
   1.57M DOF and ~35 GB at 3.09M; the 3.09M cube100 case exhausts the
   60 GB host before the solver runs (44 GB RSS observed mid-setup, swap
   thrash, OOM) with GPU VRAM below 1 GB of 16.  The binding constraint at
   multi-M DOF is host setup memory, not the device.
2. *Stock Galerkin product:* AlgebraicMultigrid's fine-level `A*P` uses
   stdlib `spmatmul`, whose up-front preallocation demanded > 17.5 GB at
   1.57M DOF (true product ~4 GB), OOM-killing both CPU-AMG and (before
   the fix) GPU-AMG builds.  Fixed for the proposed solver by the slab
   Galerkin evaluation; the stock CPU AMG path retains the limit.

Remediations identified for (1): Int32 pattern indices (halves it),
dedup-to-CSR at construction, pattern-free host assembler for runs that
never assemble on host.

## 4. Hardware efficiency

From the instrumented torsion-QS detail runs (530k DOF, solve-phase times
from the run log, JIT excluded):

| quantity                 | GPU CG+Jacobi | GPU CG+AMG | CPU CG+Jacobi |
|--------------------------|--------------:|-----------:|--------------:|
| solve-phase time         | 491.5 s       | 200.6 s    | 268.0 s       |
| CG iterations            | 16,000        | 980        | 16,000        |
| time per CG iteration    | 30.7 ms       | 204.7 ms   | 16.8 ms       |
| AMG hierarchy builds     | —             | 15.6 s     | —             |

Byte accounting per iteration (basis of the achieved-bandwidth figures):
- GPU matrix-free action, per element (160,181 hexes): connectivity
  8×8 B + coordinate gather 24×8 B + field gather 24×8 B + properties
  ~24 B + scattered accumulation ~2×24×8 B ≈ 856 B ⇒ ~137 MB/action,
  plus ~25 MB CG vector work ⇒ ~160 MB / 30.7 ms ≈ **5.2 GB/s**.
- CPU assembled SpMV: K nnz ≈ 42M ⇒ values+colind 16 B/nnz ≈ 0.67 GB +
  ~40 MB vectors ⇒ ~0.71 GB / 16.8 ms ≈ **42 GB/s** against ~50–90 GB/s
  DDR-class peak — the CPU path runs near its roofline; the GPU path
  runs at 0.5–0.7% of its.

- **The genuine model check is the per-application cost:** measured
  6.67× a Jacobi iteration vs the design's predicted ~6× (5 fine actions
  + coarse work).  (The 2.45× solve-phase speedup then follows from the
  iteration ratio by identity, so it is a consistency check, not an
  independent one.)  Per Newton solve, capped Jacobi spends 1,000
  iterations without converging while AMG converges in 74–85; the
  aggregate 16.3× total-iteration ratio also reflects the Jacobi run
  needing more Newton iterations (16 solves vs 12) on top of its capped,
  non-converged linear solves.
- **Bandwidth utilization is the honest weak spot.**  One matrix-free fine
  action moves ~160 MB (per the byte accounting above) in ~30 ms:
  ≈ 5.2 GB/s achieved against HBM-class peak of ~1 TB/s — ~0.5% of
  roofline.  This, not
  preconditioning, is why CPU assembled SpMV stays competitive with the
  GPU through 1.57M DOF (instrumented CPU figures in the table below).
  The gap between achieved and peak bandwidth is consistent with a
  launch-latency/occupancy limitation in the action kernel rather than
  bandwidth saturation, but that is a hypothesis pending kernel-level
  profiling — recorded as the top follow-up item either way, since
  optimizing the action multiplies every GPU variant, AMG included.
- **Allocation:** the V-cycle apply path preallocates all per-level
  workspaces at hierarchy conversion; the checked-in GPU test
  (`test/mechanics-gpu-device.jl`, "GPU AMG preconditioner") asserts that
  20 repeated V-cycle applications — driven through the real matrix-free
  fine action, the same operator production uses — leave live VRAM exactly
  unchanged.
  Host allocation per run is recorded in the JSONL (`cpu_alloc_bytes`);
  end-of-run VRAM: 0.65 GB at 530k, 2.0 GB at 1.57M — hierarchy plus the
  0.22–0.67 GB base matrix-free footprint.

## 5. Convergence data

Per-solve Newton iteration counts and per-solve CG iteration lists for every
run are in the JSONL records (`newton_iters`, `cg_iters` fields), including
the capped-at-1000 Jacobi solves.  Hierarchy rebuild seconds are in
instrumented records (`amg_build_s`); build *counts* are in records from
the current harness revision (`nbuilds` — e.g. nbuilds-check.jsonl:
1 build, 11.8 s, torsion-qs — the staleness detector requested no rebuild
in that rep; the earlier instrumented rep's 15.6 s `amg_build_s` predates
the counter).

## 6. Follow-up work, in priority order

1. **Matrix-free action bandwidth** (§4): the fine action achieves ~5.2 GB/s
   of ~1 TB/s-class peak (~0.5% of roofline) — the reason CPU assembled SpMV
   stays competitive through 1.57M DOF.  Kernel-level profiling to test the
   launch-latency/occupancy hypothesis, then restructuring (fused gather/
   scatter, per-element→per-DOF parallelism, launch batching).  Any gain here
   multiplies every GPU variant, AMG included.
2. **Host-memory remediations** (§3): Int32 assembler pattern indices
   (halves the 25–35 GB COO pattern), dedup-to-CSR at construction (removes
   the per-element-entry triplets), and a pattern-free `asm_cpu` mode for GPU
   runs that never assemble on the host.  These, not device limits, gate
   multi-M-DOF problems on 60 GB-class hosts.
3. **Smoother tuning**: CPU AMG's Gauss–Seidel converges in 787 iterations
   where the device's damped Jacobi needs 951–980 (§2) — ~20% headroom via
   Chebyshev-polynomial smoothing (machinery exists), ν/cycle-shape tuning,
   or l1-Jacobi.
4. **Newmark large-dt regime**: AMG should win Newmark once dt grows enough
   that c_M stops conditioning the system; the crossover dt was not mapped.
5. **CUDA validation**: the implementation is KernelAbstractions-portable and
   contains no ROCm-specific paths, but only ROCm was exercised; a CUDA run
   of the GPU test suite would close the portability claim.
6. **Upstream candidates**: the `_slab_galerkin` memory fix (AlgebraicMultigrid.jl
   would benefit directly) and the `KA.@index` qualified-macro CPU-backend
   miscompilation (KernelAbstractions.jl issue).

## 7. Scope compliance

Pure Julia throughout: AlgebraicMultigrid.jl (setup), KernelAbstractions
kernels (apply), Krylov.jl (CG).  No hypre/Trilinos/AmgX/rocSPARSE.  No
vendor-specific code paths in the proposed solution; the ROCm backend is the
test vehicle, CUDA support follows from KA + Adapt with no code changes.
