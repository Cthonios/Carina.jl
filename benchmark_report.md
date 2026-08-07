# Benchmark report: native-Julia GPU AMG for Carina's implicit paths

**Status: PASSED Critic review (3 rounds: REVISE, REVISE, PASS; verdicts and
resolutions tracked in the gpu-amg branch history, commits d209ecd/094ca3b/b9f82e4).**

**Amended 2026-08-06 (§9).**  Follow-up item 1 was carried out and its stated
cause turned out to be wrong: the FP64 atomic scatter costs nothing, and the
matrix-free action's real expense was forming the material tangent.  Replacing
that with a directional derivative gave 3.17× on the kernel and ~2.1×
end-to-end, which **overturns this report's "CPU AMG and GPU AMG are tied"
conclusion** — GPU AMG is now roughly 2.0–2.2× faster than CPU AMG (the
"current" cells are single runs; see §2).  §2's tables carry
both the as-benchmarked and current numbers; §4, §6 and §8 are corrected in
place; §9 has the ablation.  The Critic reviewed the original campaign, not
this amendment.

Campaign brief: `~/Carina-GPU.md` — **not present on this host as of
2026-08-06**; it lived outside the repository and was not archived, so this
citation is currently dangling.  Proposed solution, design rationale and
rejected alternatives (in-repo, authoritative): `benchmark/design.md`.  Raw data:
`benchmark/results/{baseline,proposed,scaling2,variance,detail,bisect,nbuilds-check,jvp}.jsonl`
plus the evidence appendix `benchmark/evidence/`; harness:
`benchmark/harness.jl` (one fresh process per run; iteration counts parsed
from Carina's own log; VRAM from `AMDGPU.memory_stats().live` at end of run).
The action microbenchmark behind §9 is `benchmark/action_bench.jl`.

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

The **"as benchmarked"** column is the campaign measurement that went through
Critic review.  The **"current"** column is the same run after the §9 action
rewrite; only GPU variants that apply the matrix-free stiffness action are
affected, and CG/Newton counts are unchanged, so the two columns describe the
same solve at different cost.  CPU variants solve assembled, never call
`stiffness_action`, and are therefore identical in both.

| variant            | as benchmarked (s) | current (s) | CG iters   | linear conv. | VRAM (GB) |
|--------------------|-------------------:|------------:|-----------:|--------------|----------:|
| **GPU CG+AMG (new)** | 299 (n=4)        | **141** (n=1) | **951–980** | yes       | 0.65      |
| **GPU CG+Jacobi**  | 578 (n=3)          | **204** (n=1) | 16,000    | **capped**   | 0.22      |
| CPU CG+AMG         | 302                | 302         | 787        | yes          | —         |
| CPU CG+Jacobi      | 372 (n=2)          | 372         | 16,000     | **capped**   | —         |
| CPU direct         | 443                | 443         | —          | (direct)     | —         |
| GPU CG+Chebyshev   | 638                | not re-run  | 3,556      | yes          | 0.22      |
| CPU CG+IC          | 1,011              | 1,011       | 8,226      | yes          | —         |
| GPU L-BFGS         | failed             | failed      | —          | (stalls; step failure) | n/a\* |

\* The failed L-BFGS run writes no JSONL record; its failure log is
`benchmark/evidence/torsion_qs_lbfgs_failure.txt`.  GPU Chebyshev was not
re-measured after the rewrite; it applies the same action and would improve,
but it was never competitive and nothing rests on its number.

- GPU AMG is **1.9× faster** (578/299, means over repeats) than the best
  pre-existing GPU option as benchmarked, and the only GPU variant that both
  converges its linear systems and beats the direct solver.  After §9 the
  margin over GPU Jacobi narrows to 1.4× (204/141) — the action rewrite helps
  the unpreconditioned variant more, because a larger share of its time was
  the action.
- Chebyshev illustrates the cost/iteration trade the design targets: 4.5×
  fewer iterations than Jacobi, yet slower — its per-iteration polynomial
  costs more than it saves.  The V-cycle pays a similar per-application
  premium (~6 fine-action equivalents) but buys a 16× iteration reduction.
- ~~CPU AMG and GPU AMG are tied at this size~~ — **true as benchmarked
  (302 vs 299), no longer true.**  The CPU's Gauss–Seidel smoother still
  converges in fewer iterations (787 vs 951–980) than the device's
  parallel-friendly damped Jacobi, but after §9 GPU AMG is **roughly 2.0–2.2×
  faster than CPU AMG** (141 vs 302), and GPU CG+Jacobi alone (204) now also
  beats CPU AMG.  The range, not a point estimate: the "current" cells are
  single runs against a documented 8% before-spread on GPU AMG
  (290/291/299/314), so 2.15× is the ratio of one measurement to one
  measurement.  The direction is not in doubt — the smallest plausible
  numerator and largest plausible denominator still leave GPU AMG ahead by
  ~2× — but the third significant figure is not earned.  Scaling widens this
  further (§3).
- GPU AMG VRAM: 0.65 GB total vs 0.22 GB unpreconditioned — the device
  hierarchy costs ~0.43 GB, inside the predicted 0.3–0.6× of the never-formed
  fine matrix (~0.9 GB).

### Newmark dynamics (dt = 5e-5)

| variant            | as benchmarked (s) | current (s) | CG iters | VRAM (GB) |
|--------------------|-------------------:|------------:|---------:|----------:|
| **GPU CG+Jacobi**  | 169                | **121** (n=1) | 2,300  | 0.26      |
| GPU L-BFGS         | 112                | not re-run  | —        | 0.34      |
| CPU CG+Jacobi      | 150                | 150         | 2,558    | —         |
| CPU CG+AMG         | 176                | 176         | 264      | —         |
| CPU CG+IC          | 183                | 183         | 672      | —         |
| GPU CG+Chebyshev   | 192                | not re-run  | 561      | 0.27      |
| CPU direct         | 551                | 551         | —        | —         |

- As predicted in the design: at mass-dominated small dt the c_M shift
  conditions the system and cheap preconditioners win.  AMG's 10× iteration
  reduction does not pay here.  **Recommended defaults: Jacobi for Newmark,
  AMG for quasistatic** — the benchmark supports both honestly.
- GPU CG+Jacobi gains least from §9 (1.40×) of any variant measured, because
  its effective operator also applies `mass_action`, which was already at the
  floor the ablation identified.  It nonetheless moves from 3rd place to
  **fastest CG variant on either device**, ahead of CPU Jacobi (150).
- GPU L-BFGS was the fastest Newmark option as benchmarked (a reversal of the
  July 2026 measurement, predating the zero-allocation fixes) — but it fails
  outright on quasistatic, so it cannot be the general answer.  It uses no
  stiffness action and so gains nothing from §9; at 112 s it stays ahead of
  the re-measured CG+Jacobi's 121 s, but the gap is now 8% rather than 34%.

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

**These are as-benchmarked figures; the 823k and 1.57M rows were NOT re-run
after the §9 action rewrite.**  Only the 530k row has current numbers (GPU AMG
141 s, GPU Jacobi 204 s — §2).  Both GPU columns at the larger sizes would
improve by a similar factor and the CPU columns would not, so every
GPU-favourable conclusion below holds *a fortiori*; no conclusion here depends
on the GPU numbers being as large as printed.  Re-running the sweep is
tracked as §6 item 4.

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
- **Low bandwidth utilization is real but is a symptom, not the disease.**
  One matrix-free fine action moves ~160 MB (per the byte accounting above)
  in ~30 ms: ≈ 5.2 GB/s achieved against the card's 288 GB/s peak — **~1.8%
  of roofline**, or ~191 ns per element, against a CPU SpMV running at ~100%
  of its own machine.  **§9 measured what that 30 ms is actually spent on,
  and it is not memory traffic:** ~73% is forming the 4th-order material
  tangent, ~19% is a bundle of geometry, gradient interpolation and the
  contraction that consumes the tangent, ~8% is gather+scatter+launch.  The
  kernel is FP64-compute-bound, so a bandwidth
  roofline is the wrong yardstick for it — a matrix-free method is *supposed*
  to trade bytes for flops, and on a part running FP64 at 1/32 rate that
  shows up as a small bandwidth fraction.  Read §9 before citing the 1.8%
  figure as a defect.
- ~~**Prime suspect: the FP64 atomic scatter.**~~  **REFUTED — see §9.**  This
  report originally named the 24 `fec_atomic_add!` calls per element in
  `_assemble_element!` (FiniteElementContainers
  `src/assemblers/Assemblers.jl:83`, via `Atomix.@atomic` in `src/Utils.jl:5`)
  as the prime suspect, on the reasoning that 3.85M FP64 global atomics per
  action with ~8 elements contending per node would lower to CAS retry loops
  on RDNA3.  A direct ablation — replace the atomic with a plain non-atomic
  `+=`, accept wrong results, keep the timing — showed **no gain whatsoever**
  (30.05 ms vs 29.67 ms baseline, i.e. marginally slower and inside the noise
  band).  The hypothesis was wrong, and the coloring / E-vector-restriction
  work it implied would have bought nothing.  §9 has the arms and the
  replacement diagnosis.
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

1. ~~**Matrix-free action bandwidth**~~ — **DONE, and the original diagnosis
   was wrong.**  The plan here was to confirm the FP64-atomic hypothesis with
   `rocprof` and then remove the atomics via coloring or an E-vector
   restriction.  Ablation refuted the hypothesis (§9): the atomics cost
   nothing.  The action's real cost was forming the material tangent, and
   replacing that with a directional derivative gave **3.17× on the kernel
   and 2.07× end-to-end** on torsion-QS GPU AMG.  What is left of this item:
   - the residual/assembled-stiffness kernels still form the tangent, and
     the assembled path genuinely needs it — but `FEC.stiffness` builds it
     once per element rather than once per action, so it is far less
     exposed;
   - state-carrying models (J2 plasticity) still take the form-and-contract
     path, because seeding duals through a return map needs the state
     containers to carry duals (§9);
   - the `SolidMechanics{CM.LinearElastic}` specialization re-evaluates a
     tangent that does not depend on `∇u` at every quadrature point; it
     wants hoisting, which needs an FEC interface that evaluates once per
     element.
   Prior art, still relevant to items 6 and 8: Pazner, Kolev & Camier,
   "End-to-end GPU acceleration of low-order-refined preconditioning"
   (IJHPCA 2023); Brown *et al.*, "Performance portable solid mechanics via
   matrix-free p-multigrid" (Ratel); and hypre's GPU BoomerAMG guidance
   (PMIS coarsening, ℓ1-Jacobi/Chebyshev smoothers, never Gauss–Seidel).
2. **The remaining implicit-kernel wins**, now that the action is 9.50 ms
   against a 2.38 ms gather/scatter/launch floor.  The 7.1 ms above that floor
   is arithmetic, and a FLOP budget puts it in perspective: the RX 7600's FP64
   peak is ~21.75 TFLOP/s ÷ 32 ≈ **0.68 TFLOP/s**, so 7.1 ms buys ~3,800
   FLOP/quadrature-point, against an estimated ~1,600–1,900 actually needed
   (geometry ~350, two gradient interpolations ~290, one-partial dual stress
   ~500–750, dense 24×9 `G·dP` ~430).  **The kernel is therefore already at
   roughly 40–50% of FP64 peak, and FP64 throughput — not bandwidth, not the
   scatter — is plausibly the binding ceiling.**  That reframes what is left:
   the wins are in doing fewer FP64 flops, or in not doing them in FP64.
   In rough payoff order:
   - **(a) FP32 action inside the preconditioner.**  The V-cycle's smoothing
     and residuals do not need FP64; wrapping an FP32 action in flexible CG
     keeps the outer solve in FP64.  Arithmetic gets ~32× cheaper on this
     part, so the kernel would be bounded below by its 2.38 ms memory floor —
     up to ~4.0× on the action, and ~2× on the AMG solve phase (of which
     ~57 ms of each 74.2 ms iteration is fine actions).  Largest single item
     on this list.
   - **(b) Hand-written analytic directional derivative** for NeoHookean,
     removing the ~2–3× dual-number overhead on the stress evaluation
     (~1.5–2 ms).  Costs generality — it is per-model, and the Hencky work
     would need its own — so measure (a) first.
   - **(c) Precompute per-quadrature-point `∇N_X`/`JxW`** instead of
     recomputing the Jacobian inverse every action: ~102 MB of extra traffic
     replacing ~1 ms of arithmetic.  **Whether this pays depends entirely on
     the achieved read bandwidth, and the margin is thin.**  At the 57 GB/s
     the scatter path sustains, 102 MB costs 1.79 ms — a net *loss*.  It only
     wins if the read runs far closer to peak, which it plausibly does: this
     is a coalesced streaming read of a contiguous per-quadrature-point array,
     not the indirect gather/scatter that limits `mass_action`, so ~0.35–0.5 ms
     near the 288 GB/s peak is the target.  Measure the achieved bandwidth of
     that read before implementing.  Trading flops for bytes is the right
     direction on a 1/32-rate part — the opposite of the usual matrix-free
     instinct — but not at any price.
   - **(d) Exploit `G`'s structure** in the final contraction: `discrete_gradient`
     builds a 24×9 matrix with 3 nonzeros per row and it is then multiplied
     densely (~430 → ~120 FLOP/qp).
   - **(e) Profile** for occupancy and register pressure, which ablation
     cannot reach.  See the note on `rocprofv3` availability in §9.
3. **Ablate the explicit kernel the way §9 ablated the implicit one.**  §8's
   "~13× headroom" claim is withdrawn because it divided achieved into peak
   *bandwidth* for a kernel that is not bandwidth-bound.  The honest
   replacement is a measurement, not a revised estimate: time an explicit step
   against an arm whose `pk1_stress` is replaced by a synthetic stress, which
   bounds the constitutive share exactly as arm D did for the tangent.
   Explicit evaluates stress rather than a tangent, so there is no 73% to
   reclaim — but the size of what *is* there is currently unknown, and this is
   the cheapest open measurement in this list.
4. **Re-run the scaling sweep** (§3): the 823k and 1.57M rows predate §9, so
   the GPU columns there understate current performance.  The conclusions are
   unaffected in direction, but the table should not stay stale.
5. **Host-memory remediations** (§3): Int32 assembler pattern indices
   (halves the 25–35 GB COO pattern), dedup-to-CSR at construction (removes
   the per-element-entry triplets), and a pattern-free `asm_cpu` mode for GPU
   runs that never assemble on the host.  These, not device limits, gate
   multi-M-DOF problems on 60 GB-class hosts.
6. **Smoother tuning**: CPU AMG's Gauss–Seidel converges in 787 iterations
   where the device's damped Jacobi needs 951–980 (§2) — ~20% headroom via
   Chebyshev-polynomial smoothing (machinery exists), ν/cycle-shape tuning,
   or l1-Jacobi.
7. **Newmark large-dt regime**: AMG should win Newmark once dt grows enough
   that c_M stops conditioning the system; the crossover dt was not mapped.
8. **CUDA validation**: the implementation is KernelAbstractions-portable and
   contains no ROCm-specific paths, but only ROCm was exercised; a CUDA run
   of the GPU test suite would close the portability claim.
9. **Upstream candidates**: the `_slab_galerkin` memory fix (AlgebraicMultigrid.jl
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
- **Both paths sit well below the *bandwidth* roofline — but that is the
  wrong roofline.**  Using the same byte accounting as §4 (~856 B/element),
  explicit moves ~21 GB/s at 40.2 ns/element — ~7% of the card's 288 GB/s,
  against the implicit action's ~1.8%.  (A floor: explicit also touches
  velocity, acceleration and state, so true traffic is higher and the
  achieved fraction better.)  The original reading of that 4× gap — the same
  24-atomic scatter, amortized over a constitutive update instead of a
  `Bᵀ C B v` product — is **refuted for the implicit side by §9**, where
  removing the atomics changed nothing and the constitutive work turned out
  to be 92% of the cost.  The same explanation therefore cannot be assumed
  for explicit either.  That the explicit kernel is *also* FP64-arithmetic-
  bound is a plausible **hypothesis, not a measurement** — it has never been
  ablated, and §6 item 3 is the experiment that would settle it.  Stating it
  as fact here would repeat exactly the error §9 corrects.
- **Headroom.**  This report previously claimed explicit was "leaving ~13× on
  the table relative to roofline."  **Withdraw that number.**  It divides
  achieved bandwidth into peak bandwidth for a kernel that is not
  bandwidth-bound, so it measures arithmetic intensity, not waste.  The
  binding ceiling on this part is FP64 throughput (1/32 rate), and no
  equivalent ablation has been run on the explicit kernel to establish how
  much of its 40.2 ns/element is reducible.  What §9 does establish is that
  the implicit action had ~3× of genuine headroom and it has now been taken;
  whether explicit has a comparable constitutive-side win is untested and is
  the natural next experiment (§6 item 3).  On a datacenter GPU
  (full-rate FP64, 1.6–5.3 TB/s) both ceilings move by an order of magnitude.

## 9. The action kernel: ablation, and a 3.17× rewrite

Follow-up item 1 named the matrix-free stiffness action as the top
optimization target and the FP64 atomic scatter as its prime suspect.  The
target was right.  The suspect was wrong.  This section records how that was
established and what replaced it.  Raw arm-by-arm output, including the exact
source patch behind each arm, is in `benchmark/evidence/action_ablation.txt`;
the driver is `benchmark/action_bench.jl`.

**Method.**  Fedora 44 packages only `rocprofiler-register` and `roctracer`,
with no `rocprof`/`rocprofv3` binary in its repositories, so no profiler was
available *from the distribution*.  (It is obtainable by other routes — AMD
ships `rocprofv3` in its own ROCm repositories, usable from a container on this
host, and there are pip wheels; nothing here should be read as "profiling is
impossible on this machine," only that it was not on hand.  Occupancy and
register pressure in particular cannot be established by ablation and will need
one.)  The diagnosis was therefore made by ablation: patch out one
suspected cost at a time, accept that the results become wrong, and keep only
the timing.  Each arm builds the torsion-QS problem on device, then times
`assemble_matrix_free_action!` in isolation (5 warmup reps discarded, 50 timed,
median), so solver convergence cannot perturb the measurement.  A checksum over
the output storage detects whether an arm actually changed the arithmetic.  The
baseline arm reproduces §4's independently measured ~191 ns/element at
185 ns/element, which is what licenses comparing these numbers to §4's.

**Arms** (530k DOF, 160k HEX8, median of 50):

| arm | ms/action | ns/elem | what it isolates |
|-----|----------:|--------:|------------------|
| A  baseline (`Atomix.@atomic`)       | 29.67 | 185.4 | reference |
| B  atomics → plain `+=` (racy)       | 30.05 | 187.8 | **cost of the FP64 atomics** |
| C  `mass_action`, unmodified         |  2.38 |  14.9 | gather + scatter + launch floor |
| D  material tangent not formed       |  8.10 |  50.7 | cost of forming ∂P/∂∇u |
| —  **after the rewrite below**       |  **9.50** | **59.4** | |
| B2 atomics → plain `+=`, *post-rewrite* | 9.64 | 60.2 | atomics at the new operating point |

Arm B2 exists because arm B was measured when the scatter was 8% of a 30 ms
kernel; after the rewrite it is ~25% of a 9.5 ms one, so "atomics cost
nothing" had to be re-established rather than assumed to carry over.  It
does: 9.64 vs 9.50 ms, again ~1.4% slower and again inside the band, with the
checksum moved (1.906160e6 → 1.901813e6).  The dismissal holds at both
operating points — but note it is a statement about *these* two points, not a
proof that a conflict-free scatter can never pay.

`mass_action` (arm C) is the control that makes this readable: identical
connectivity gather, identical 24-atomic scatter, identical element count and
launch geometry, differing *only* in arithmetic per element.  It was timed in
three separate processes at 2.385 / 2.374 / 2.377 ms, a 0.5% spread, so
cross-run comparison here is sound.

**Result — one baseline action (30.12 ms) decomposes as:**

| component | ms | share | from |
|-----------|---:|------:|------|
| forming the 4th-order material tangent | 22.0 | **73%** | C − D |
| geometry + interpolation + contraction |  5.7 | 19% | D − mass |
| gather + scatter + launch              |  2.4 |  8% | mass |
| FP64 atomics                           | ~0   | **0%** | A − B |

The 19% row is a **bundle these arms cannot separate**, not the contraction
alone.  Arm D removes only the tangent; it still runs `map_interpolants` (the
per-quadrature-point Jacobian inverse), the `∇u` interpolation,
`discrete_gradient` (forming the 24×9 `G`), the 9×9 contraction, and the final
`G·dP`.  `mass_action` has none of the gradient machinery, so the difference
covers all of it.  Splitting this row needs its own arms.

Arm B is the refutation: removing *every* FP64 atomic left the kernel 1.3%
slower — inside the ±1.6 ms run-to-run band — while moving the checksum
(1.906160e6 → 1.903398e6), which proves the race really happened and the
ablation really took effect.  Element coloring and the libCEED/MFEM E-vector
restriction, both proposed by the original item 1, would have bought nothing
here.  Arm C then shows 92% of the kernel is arithmetic, and `mass_action`
sustaining 57 GB/s on the same access pattern shows this gather/scatter shape
is not what limits the machine.

**The fix: never form the tangent.**  A matrix-free action needs the tangent's
action on *one* direction, not the tangent.  The kernel was calling
`CM.material_tangent` to build all 81 components of ∂P/∂∇u and then contracting
them against a single vector.  `Carina/src/physics.jl` now takes the derivative
along ∇v directly —

    dP = ∂P/∂∇u : ∇v

— via one forward-mode dual pass over `pk1_stress` (`_pk1_jvp`), seeding ∇u
with ∇v as the single partial direction.  `A_v` is never built, and the 9×9
contraction disappears with it.

**But the rewrite is not free, and the table above says so.**  The JVP kernel
lands at 9.50 ms — **1.40 ms *above* arm D**, which still performs the
contraction the rewrite eliminates.  The dual-number pass plus the second
gradient interpolation for ∇v therefore cost ~1.4 ms *more* than the
contraction they displace.  Net saving is 30.12 − 9.50 = **20.6 ms, less than
the 22.0 ms attributed to tangent formation alone**.  The correct summary is:
the rewrite removes the tangent formation and pays ~1.4 ms for the privilege —
not that it removes "the 73% and most of the 19%."  That ~1.4 ms is itself a
target (§6 item 2(b): a hand-written analytic directional derivative would avoid
the dual overhead).

This is the standard matrix-free Jacobian application used by
libCEED/MFEM/Ratel.  Cost is about two stress evaluations regardless of model,
so it does not depend on any model having a hand-written tangent — though
NeoHookean's already does (`ConstitutiveModels`
`src/modules/hyperelasticity/NeoHookean.jl:71`), which is why the 73% is
inherent to building a 4th-order tensor rather than an artifact of AD.

Applied to stateless models (every `Hyperelastic`, NS = 0).  State-carrying
models keep the form-and-contract path: `pk1_stress` runs a return map against
Float64 state containers that cannot hold dual numbers, so J2 plasticity is
correct but not yet faster.

**Correctness.**  The action is unchanged as an operator:
- **At finite strain, on the device:** with the problem first solved one load
  step (|U|_max = 3.14e-4), the pre- and post-rewrite kernels produce a
  **bit-identical** checksum — 1.906165947916275e+06 from both, all 16
  significant digits — at 31.24 vs 9.72 ms.  Printed to full precision
  deliberately: `sum(abs, ·)` over 5.3e5 entries compresses a localized
  discrepancy hard, so the 7-digit form this report first quoted left only a
  factor-of-a-few margin on the geometric term.  At full precision there is no
  margin question.  This is the binding evidence.  An initial-state checksum is **not** — at U = 0 the
  geometric part of ∂P/∂∇u vanishes identically, so two kernels differing only
  there would agree anyway.  (The U = 0 checksums do also match; that fact
  simply proves less than it appears to.)
- `test/matrix-free-operators.jl` asserts `‖K·v − action(v)‖/‖K·v‖ < 1e-12`
  against the *assembled* stiffness, which still forms the full analytic
  tangent, at a converged ~1e-3-strain state.  It passes — so the dual-number
  derivative and the hand-written analytic tangent agree to roundoff.  This is
  a host-side check.
- Per-solve CG iteration *lists* match the baseline records run for run (AMG's
  951-list; Newmark's [181, 205, 193, …]).  Note the QS Jacobi row's
  "16,000 → 16,000" is **vacuous** as evidence — both sides hit the 1000-
  iteration cap, so equality there is guaranteed regardless of the operator.
- Full suite: 810/810, unchanged.

**End-to-end (torsion-QS, 530k DOF; `benchmark/results/jvp.jsonl`).**  Nothing
about convergence moved — same CG totals, same Newton counts per step, same
VRAM — which is the signature of an operator that is mathematically identical
and merely cheaper:

| variant | before (s) | after, n=1 (s) | speedup | CG iters | Newton |
|---------|-----------:|---------------:|--------:|---------:|--------|
| GPU CG+AMG, total       | 291.06 (n=4: 290–314) | **140.75** | **2.07×** | 951 → 951       | [3,3,3,3] both |
| GPU CG+AMG, solve phase | 194.71 |  **70.57** | **2.76×** | — | — |
| GPU CG+Jacobi, total    | 576.96 (n=3: 576–581) | **204.30** | **2.82×** | 16,000 → 16,000 | [4,4,4,4] both |
| GPU Newmark CG+Jacobi   | 169.01 (n=1) | **120.97** | **1.40×** | 2,300 → 2,300   | [3,3,3,3] both |

**Every "after" cell is a single run.**  The "before" AMG spread is 8%
(290/291/299/314, §1), so 2.07× is really ~2.0–2.2×; the chosen baseline
(291.06 s) is one of the two fast reps, so the ratio understates rather than
inflates.  The AMG comparison is also like-for-like on the 951-vs-980
nondeterminism §1 documents — both the 291.06 s baseline and the 140.75 s
rep took the 951-iteration trajectory.  Repeats would firm up the third
significant figure; nothing here depends on it.

The solve-phase ratio (2.76×) is the honest one for the kernel change; the
total is diluted by ~40 s of JIT and ~11 s of host-side AMG setup, neither of
which this touches.  Newmark gains least because its effective operator also
applies `mass_action`, which was already at the arm-C floor.

One caveat on the iteration-count argument: matching counts are strong
evidence for AMG (951 exactly, and the per-solve lists agree) and for Newmark
([3,3,3,3] with matching per-solve lists), but **vacuous for QS Jacobi** —
both sides hit the 1000-iteration cap, so 16,000 → 16,000 was guaranteed
regardless of what the operator did.

**What this changes upstream in this report.**  §2's headline comparison was
"CPU AMG and GPU AMG are tied at this size" (302 s vs 299 s).  They are no
longer tied: GPU AMG is now **roughly 2.0–2.2× faster than CPU AMG**
(140.75 s vs 302.06 s, single run against the 8% before-spread), and GPU
CG+Jacobi alone (204 s) now beats CPU AMG too.  The CPU
figures are unchanged and were not re-run, because the CPU path solves
assembled and never calls `stiffness_action`.  §4's efficiency discussion and
§8's headroom claim are corrected in place.

**Closing note — the untested twin.**  The explicit kernel
(`FEC.residual` → `pk1_stress`) evaluates stress, not a tangent, so it has no
equivalent 73% to reclaim; but no ablation has been run on it, and §8's
withdrawn "~13×" claim should not be replaced with a guess.  Timing an explicit
step against a synthetic-stress arm is the cheap way to find out, and it is now
the top open item in §6.
