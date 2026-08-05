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

**Reproducing the study.**  Everything needed is in the repository:
standalone input decks for every (case, variant) pair are checked in under
`benchmark/inputs/` and run directly (e.g.
`bin/carina benchmark/inputs/torsion-qs-gpu-cg-amg.yaml`); the raw
JSON-lines records behind every number in this report are committed in
`benchmark/results/`; the torsion mesh (with its Cubit journal) is tracked
at `examples/meshes/torsion/`, and the large cube meshes are regenerated
deterministically by `benchmark/meshgen.jl`.  `benchmark/README.md` gives
the full procedure — mesh generation, single runs, and the sweep scripts.

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

**Machine.**  AMD Ryzen 9 9900X (12 cores / 24 threads, dual-channel DDR5,
~90 GB/s theoretical / 50–90 GB/s achievable) and an AMD Radeon RX 7600
(Navi 33, RDNA3, 32 CU, 8 GB GDDR6, **288 GB/s** peak, FP64 at 1/32 rate).
This is a consumer card, not a datacenter part: the peak-bandwidth ratio
between the two devices is only ~3–4×, which caps what *any*
bandwidth-bound solver can gain here.  Datacenter GPUs (MI250X/MI300,
A100/H100) sit at 1.6–5.3 TB/s with full-rate FP64 and would move that cap
by an order of magnitude — the algorithmic conclusions below transfer, the
absolute ratios do not.

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
  runs at ~1.8–2.4% of its.

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
  ≈ 5.2 GB/s achieved against the card's 288 GB/s peak — **~1.8% of
  roofline**, or ~191 ns per element.  This, not
  preconditioning, is why CPU assembled SpMV stays competitive with the
  GPU through 1.57M DOF (instrumented CPU figures in the table below):
  the CPU runs at ~100% of its machine and the GPU at under 2% of its,
  which more than cancels the ~3–4× peak-bandwidth advantage.
- **Prime suspect: the FP64 atomic scatter.**  Every element of the action
  ends in `_assemble_element!`
  (FiniteElementContainers `src/assemblers/Assemblers.jl:78-85`), a loop of
  24 `fec_atomic_add!` calls, and `fec_atomic_add!` is
  `Atomix.@atomic field.data[index] += val` (`src/Utils.jl:5`) — an FP64
  global atomic.  At 160k HEX8 that is 3.85M FP64 atomics per action with
  ~8 elements contending for each node, on an RDNA3 consumer part where
  FP64 runs at 1/32 rate and FP64 global atomic-add commonly lowers to a
  CAS retry loop.  This also explains the explicit/implicit asymmetry:
  explicit assembly pays the same scatter (`src/Formulations.jl:153`) but
  amortizes it over a full constitutive update per element per step,
  whereas the implicit action does a comparatively cheap `Bᵀ C B v` per
  quadrature point and then pays the identical scatter ~6× per V-cycle ×
  ~980 CG iterations.  Same cost, hidden in one case, dominant in the
  other.  Still a hypothesis pending rocprof confirmation, but it is the
  top follow-up item either way, since optimizing the action multiplies
  every GPU variant, AMG included.
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
   of the card's 288 GB/s peak (~1.8% of roofline) — the reason CPU assembled
   SpMV stays competitive through 1.57M DOF.  Profile with `rocprof` to
   confirm the FP64-atomic-scatter hypothesis (§4), then eliminate the
   atomics: element coloring, or better the libCEED/MFEM element-restriction
   pattern — accumulate into a conflict-free E-vector, then apply the
   transpose restriction through a precomputed node→element map.  Any gain
   here multiplies every GPU variant, AMG included.
   Prior art worth reading before implementing: Pazner, Kolev & Camier,
   "End-to-end GPU acceleration of low-order-refined preconditioning"
   (IJHPCA 2023); Brown *et al.*, "Performance portable solid mechanics via
   matrix-free p-multigrid" (Ratel), which reports only "up to 2× benefit
   for linear elements" and so bounds what to expect at HEX8; and hypre's
   GPU BoomerAMG guidance (PMIS coarsening, ℓ1-Jacobi/Chebyshev smoothers,
   never Gauss–Seidel) for item 3.
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

## 8. Explicit dynamics: what the GPU is actually worth here

The implicit result reads as a disappointment only against a belief that
explicit dynamics runs far faster on this GPU.  That belief was never
measured against the CPU's best showing, so this section does it.

**Method.**  Same torsion bar at seven refinements (39k to 7.81M DOF),
generated by `benchmark/torsiongen.jl` — `N=20` reproduces the checked-in
`torsion.g` exactly (160,000 elements, identical extents and coordinates).
Central difference, neo-Hookean, free-flying bar with the twist initial
velocity of `examples/mechanics/explicit-dynamic/torsion`.  The step scales
as 1/N so the CFL number is fixed and only cost-per-step varies with size.
Each run is two equal control intervals: the first absorbs warm-up and device
kernel compilation, the second is the measured one.  Output is stripped to
nodal displacement with no recovery.  The CPU baseline uses **24 threads**,
which is its best: at N=20, 24 threads beats 12 (23.6 vs 27.5 ms/step).
Driver `benchmark/explicit_sweep.jl`, sweep `benchmark/run_explicit_scaling.sh`,
raw records `benchmark/results/explicit-scaling.jsonl`.

| N  | elements  | DOF       | CPU ms/step | GPU ms/step | ratio | CPU ns/elem | GPU ns/elem |
|----|----------:|----------:|------------:|------------:|------:|------------:|------------:|
| 8  |    10,240 |    39,123 |        1.84 |        1.42 | 1.30× |       179.7 |       138.4 |
| 12 |    34,560 |   122,187 |        5.62 |        1.86 | 3.02× |       162.6 |        53.8 |
| 20 |   160,000 |   530,523 |       23.55 |        6.62 | 3.55× |       147.2 |        41.4 |
| 28 |   439,040 | 1,415,403 |       63.00 |       17.67 | 3.57× |       143.5 |        40.2 |
| 36 |   933,120 | 2,961,147 |      131.12 |       37.62 | 3.49× |       140.5 |        40.3 |
| 44 | 1,703,680 | 5,352,075 |      231.40 |       68.80 | 3.36× |       135.8 |        40.4 |
| 50 | 2,500,000 | 7,810,803 |      341.50 |      100.50 | 3.40× |       136.6 |        40.2 |

- **The explicit GPU advantage is ~3.4×, and it saturates.**  The ratio
  climbs from 1.30× at 39k DOF — too little work to fill 32 CUs — to 3.55× by
  530k, then stays flat within noise through 7.81M DOF.  There is no widening
  at scale to wait for; both devices are in their asymptotic regime, with
  per-element cost flat at ~40 ns (GPU) and ~136–147 ns (CPU).
- **Nothing here is 25×.**  That figure is reproducible only against a
  *single-threaded* CPU: at N=20, one thread gives 232.8 ms/step, so
  232.8/6.62 = **35.2× against the GPU** — while the same CPU with 24 threads
  gives 3.55×.  Comparing a GPU against a serial CPU mostly reports the
  thread count.  The full curve at N=20 (160k elements):

  | threads | ms/step | vs 1 thread | parallel efficiency | vs GPU |
  |--------:|--------:|------------:|--------------------:|-------:|
  |       1 |  232.80 |       1.00× |                100% |  35.2× |
  |       2 |  126.20 |       1.84× |                 92% |  19.1× |
  |       4 |   66.30 |       3.51× |                 88% |  10.0× |
  |       8 |   37.93 |       6.14× |                 77% |   5.7× |
  |      12 |   27.50 |       8.47× |                 71% |   4.2× |
  |      24 |   23.55 |       9.89× |          41% (SMT)  |   3.55× |

  **This is a trap for every CPU/GPU number in this report: Julia 1.12
  defaults to one thread**, so `bin/carina deck.yaml` with no `--threads` runs
  serial and flatters the GPU by ~10×.  Note also that the 1-thread row is not
  simply "the same code, serialized" — `fec_atomic_add!`
  (FEC `src/Utils.jl:11-21`) branches on `Threads.nthreads() > 1` and skips
  the atomic entirely on a single thread, so the serial path is *cheaper per
  element* and the 9.9× scaling is if anything conservative.
- **This reframes §2 and §4.**  Explicit's honest advantage on this hardware
  is ~3.4×; GPU AMG's 1.9× over the best pre-existing GPU option, tying CPU
  AMG, is within a factor of ~2 of it, not the order of magnitude a 25×
  baseline would imply.  The implicit path is behind, but not anomalously so.
- **Both paths are far below the hardware, and by the expected margin.**
  Using the same byte accounting as §4 (~856 B/element), explicit moves
  ~21 GB/s at 40.2 ns/element — ~7% of the card's 288 GB/s, against the
  implicit action's ~1.8%.  (A floor: explicit also touches velocity,
  acceleration and state, so true traffic is higher and the achieved fraction
  better.)  A roughly 4× better bandwidth fraction is exactly what the §4
  hypothesis predicts — the same 24-atomic scatter, amortized over a full
  constitutive update instead of a `Bᵀ C B v` product.  Note also that FP64
  runs at 1/32 rate on this part, so explicit may be partly compute-bound;
  separating the two needs `rocprof`, which is follow-up item 1 either way.
- **Headroom.**  Even explicit is leaving ~13× on the table relative to
  roofline.  The atomics fix is worth more than any solver change on either
  path, and on a datacenter GPU (full-rate FP64, 1.6–5.3 TB/s) the ceiling
  for both moves by an order of magnitude.
