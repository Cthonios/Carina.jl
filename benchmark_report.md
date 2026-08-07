# Carina solver performance: measured results

What Carina's solvers cost, on CPU and GPU, across quasi-static, implicit
dynamic, and explicit dynamic problems from 39k to 7.8M degrees of freedom.

Every number here traces to a raw record in `benchmark/results/`.
`benchmark/README.md` explains how to reproduce any of them.

---

## Executive summary

### What the GPU buys

| Problem class | Speedup over the best CPU option | Notes |
|---|---|---|
| **Quasi-static implicit** | **2.2× total, 3.0–5.0× in the solve phase** | Grows with problem size. At 1.57M DOF the GPU is the only device that runs at all — CPU AMG exhausts host memory. |
| **Implicit dynamics** (small Δt) | **1.3–1.4×** | Modest by nature: the mass term already conditions the system, so there is less for a better solver to win. |
| **Explicit dynamics** | **3.4×** | Saturates at ~3.4× from ~500k DOF upward; that is ~80% of the memory-bandwidth ratio between the two devices, which is the ceiling. |

The solve-phase figures are the honest measure of solver work; totals include
~40 s of Julia compilation charged identically to every run, which dilutes the
ratio on short problems.

These are measured on a **consumer** GPU — a Radeon RX 7600 (288 GB/s, FP64 at
1/32 rate) against a 12-core Ryzen 9 9900X. The peak-bandwidth ratio between
the two is only ~3–4×, which caps any bandwidth-bound solver here regardless of
software quality. A datacenter GPU (MI250X/MI300, A100/H100) has 6–18× more
bandwidth and full-rate FP64; the algorithmic conclusions carry over, the
absolute ratios do not.

### Recommended solver by problem

| Problem | Device | Use | Why |
|---|---|---|---|
| **Quasi-static** | GPU | `cg` + `amg` | Fastest measured at every size. The only GPU preconditioner whose linear systems actually converge. |
| Quasi-static, small–medium | CPU | `direct` | No tuning, always converges. |
| Quasi-static, large | CPU | `cg` + `amg` | Iteration count independent of problem size. |
| **Implicit dynamics, small Δt** | GPU | `cg` + `jacobi` | The `c_M` mass shift conditions the system; AMG's iteration reduction does not repay its per-application cost. |
| Implicit dynamics, small Δt | CPU | `cg` + `jacobi` | Same reason. |
| Implicit dynamics, large Δt | either | `cg` + `amg` | As Δt grows the mass shift weakens and Jacobi degrades. The crossover Δt has not been mapped. |
| **Explicit dynamics** | GPU | — | No linear solve. Matrix-free throughout; runs to 7.8M DOF on 8 GB. |

Two solvers to avoid as defaults despite good individual numbers:

- **L-BFGS** is the fastest Newmark option on GPU (109 s vs 118 s for
  CG+Jacobi, an 8% margin) but **fails outright on quasi-static, on both
  devices** — it stalls seven orders short of tolerance. Not a device problem
  and not fixable by tuning; see §2. It cannot be a general default.
- **Chebyshev** converges its linear systems and is respectable (221 s
  quasi-static) but is beaten by both AMG and plain Jacobi. Its polynomial
  costs more per iteration than the iterations it saves.

**Never run `cg` with no preconditioner** on a real mesh. It is valid and very
slow.

### Practical limits

- **GPU memory is not the constraint.** The largest implicit case measured
  (1.57M DOF) uses 2.0 GB of the card's 8 GB; explicit runs to 7.8M DOF.
- **Host memory is.** The CPU assembler keeps its sparsity pattern in
  per-element-entry (COO) form — 25 GB at 1.57M DOF, ~35 GB at 3.09M — so a
  60 GB host runs out during setup at ~3M DOF. This affects GPU runs too,
  because the AMG hierarchy is still built host-side. This, not the device, is
  what gates multi-million-DOF implicit problems today.
- Julia defaults to **one thread**. `bin/carina deck.yaml` without `--threads`
  runs the CPU path serially and overstates the GPU advantage by ~10×. Use
  `--threads 24` on this class of machine.

---

## 1. Method

**Hardware.** AMD Ryzen 9 9900X (12 cores / 24 threads, dual-channel DDR5,
~90 GB/s theoretical, 50–90 GB/s achievable) and an AMD Radeon RX 7600
(Navi 33, RDNA3, 32 CU, 8 GB GDDR6, 288 GB/s peak, FP64 at 1/32 rate).
Julia 1.12.6, ROCm through AMDGPU.jl. CPU runs use 24 threads.

**Measurement.** One (case, variant) per fresh Julia process — no JIT or VRAM
carryover. Total wall time therefore includes ~40 s of compilation identically
in every cell; per-step walls and solve-phase sums come from the run log and
exclude it. Iteration counts are parsed from Carina's own log, VRAM from
`AMDGPU.memory_stats().live` at end of run. The harness asserts non-failure on
every run.

**Repeats and spread.** Headline cells are repeated where marked `n=`;
single-run cells carry the observed spread of their class, which is 0.8% for
GPU Jacobi, ~2% across GPU AMG repeats, and 2.3% for CPU Jacobi. Third
significant figures are not meaningful.

**Problems.**

- `torsion-qs` — cylinder (R 25 mm × L 1 m), 530,523 DOF, neo-Hookean
  (E = 1 GPa, ν = 0.25), base clamped, 0.05 rad twist over 4 load steps.
  Newton (abs 1e-6 / rel 1e-10), CG rtol 1e-8, itmax 1000.
- `torsion-newmark` — same mesh, free-free, rigid-torsion initial velocity
  (a = 8000 s⁻¹), Newmark β = 0.25, Δt = 5e-5, 4 steps.
- `cube64-qs` / `cube80-qs` — generated 64³ / 80³ HEX8 cubes (823,875 and
  1,594,323 DOF), uniaxial stretch over 2 load steps.
- `explicit-torsion` — the same bar at seven refinements, 39k to 7.81M DOF,
  central difference, CFL number held fixed across sizes.

Table labels are mesh-total DOF; the JSONL `n_dofs` field records free DOFs
after boundary-condition elimination (torsion-qs 527,877; cube64 806,975;
cube80 1,568,079).

**Correctness.** All variants of a case solve the identical nonlinear problem
with identical tolerances. GPU-vs-CPU solution agreement is a checked-in test:
`test/mechanics-gpu-device.jl` solves a 16³ cube with device CG+AMG and
requires agreement with CPU direct at rtol 1e-7 (ROCm output recorded in
`benchmark/evidence/gpu_amg_test_rocm.txt`).

---

## 2. Quasi-static — 530k DOF torsion

"CG iters" is the total across all Newton iterations of all steps. "Capped"
means the 1000-iteration CG limit was hit, i.e. those linear systems did **not**
converge to rtol 1e-8 — reported as measured.

| variant | total (s) | solve phase (s) | CG iters | linear conv. | VRAM (GB) |
|---|---:|---:|---:|---|---:|
| **GPU CG+AMG** | **138** (n=2) | **70.5** | 951 | yes | 0.65 |
| GPU CG+Jacobi | 203 | 145.4 | 16,000 | **capped** | 0.22 |
| GPU CG+Chebyshev | 221 | 159.9 | 3,556 | yes | 0.22 |
| CPU CG+AMG | 302 | 211.0 | 787 | yes | — |
| CPU CG+Jacobi | 372 (n=2) | 268.0 | 16,000 | **capped** | — |
| CPU direct | 443 | 360.0 | — | (direct) | — |
| CPU CG+IC | 1,011 | 920.5 | 8,226 | yes | — |
| L-BFGS (CPU *and* GPU) | **fails** | — | — | stalls; step failure | — |

- **GPU AMG is 2.2× faster than CPU AMG** (138 vs 302) and 3.2× faster than
  the CPU direct solver. In the solve phase alone it is 3.0× (70.5 vs 211.0).
- It is also the only GPU variant that converges its linear systems. Jacobi
  hits the iteration cap on every solve and needs an extra Newton iteration per
  step (4 vs 3) as a result.
- Chebyshev shows the cost/iteration trade clearly: 4.5× fewer iterations than
  Jacobi, yet slower overall.
- The CPU's Gauss–Seidel smoother converges in fewer iterations than the
  device's parallel-friendly damped Jacobi (787 vs 951) — the GPU wins on
  throughput per iteration, not on iteration count.
- AMG's VRAM cost is the device hierarchy: 0.65 GB against 0.22 GB
  unpreconditioned, i.e. ~0.43 GB, inside the 0.3–0.6× of the never-formed
  fine matrix (~0.9 GB) that smoothed aggregation predicts.
- **L-BFGS fails here on both devices, for an algorithmic reason.** It builds
  an inverse-tangent model from its last 10 secant pairs, and quasi-static has
  no mass term to condition that operator — the same system needs 787 AMG
  iterations, and 10 rank-one updates cannot represent it. `|r|` drops to ~0.1
  within 13 iterations, then stalls at ~1.2e-3 against a 1e-10 target while the
  line search collapses to 8–10 Armijo backtracks and steps of ~1e-3. It
  exhausts its 500-iteration cap on the *first* of four load steps. The CPU and
  GPU runs agree digit-for-digit over the first eleven iterations, which is
  what rules the device out. Trajectories and mechanism:
  `benchmark/evidence/torsion_qs_lbfgs_failure.txt`.

---

## 3. Implicit dynamics (Newmark) — 530k DOF, Δt = 5e-5

| variant | total (s) | solve phase (s) | CG iters | VRAM (GB) |
|---|---:|---:|---:|---:|
| **GPU L-BFGS** | **109** | — | — | 0.33 |
| **GPU CG+Jacobi** | **118** | 29.5 | 2,300 | 0.26 |
| GPU CG+Chebyshev | 127 | 33.2 | 561 | 0.27 |
| CPU CG+Jacobi | 150 | — | 2,558 | — |
| CPU CG+AMG | 176 | — | 264 | — |
| CPU CG+IC | 183 | — | 672 | — |
| CPU direct | 551 | — | — | — |

- The GPU advantage here is **1.27×** (118 vs 150) for like-for-like Jacobi,
  1.38× comparing each device's best.
- **AMG does not pay at small Δt.** At mass-dominated time steps the `c_M`
  shift conditions the system and cheap preconditioners win: AMG's 10×
  iteration reduction (264 vs 2,558 on CPU) costs more than it saves. The
  design predicted this; the measurement confirms it.
- L-BFGS leads by 8%, but it does no linear solve at all — it converges by many
  cheap nonlinear iterations (284–312 per step). What makes it work here is the
  mass shift `c_M = 1/(βΔt²) ≈ 1.6e9`, which leaves the effective operator
  strongly diagonally dominant and therefore easy to model at low rank. Remove
  it and the same solver stalls (§2).

---

## 4. Scaling

Quasi-static uniaxial stretch on generated cubes (2 load steps, so absolute
times are not comparable to the torsion rows above — the trend is the point).

| size | GPU AMG | GPU Jacobi | CPU AMG | CPU Jacobi |
|---|---:|---:|---:|---:|
| 530k (torsion) | **138** s / 951 | 203 s / 16,000c | 302 s / 787 | 372 s / 16,000c |
| 823k (cube64) | **97** s / 118 | 104 s / 2,670 | 164 s / 100 | 169 s / 2,670 |
| 1.57M (cube80) | **161** s / 136 | 183 s / 3,320 | **out of memory** | 312 s / 3,320 |

("c" = CG iteration cap hit; linear systems not converged to tolerance.)

Solve phase only, which removes the fixed compilation cost:

| size | GPU AMG | best CPU | ratio |
|---|---:|---:|---:|
| 530k | 70.5 s | 211.0 s (AMG) | **3.0×** |
| 823k | 17.9 s | 76.2 s (AMG) | **4.3×** |
| 1.57M | 34.5 s | 172.2 s (Jacobi) | **5.0×** |

- **The GPU advantage grows with problem size**, from 3.0× to 5.0× in solver
  work across this range.
- **AMG iteration counts stay flat** (118 → 136) while Jacobi's grow
  (2,670 → 3,320). That h-independence is the property AMG exists for, and it
  holds on the device.
- **At 1.57M DOF the GPU is the only option that runs.** Stock
  AlgebraicMultigrid setup is OOM-killed on this host; the implementation here
  survives because its fine-level Galerkin product is evaluated in column slabs
  (`_slab_galerkin`, peak transient = one slab). Excerpts:
  `benchmark/evidence/cube80_oom_excerpts.txt`,
  `benchmark/evidence/sa_mem_probe_cube80.txt`.
- Newmark at 1.57M: GPU Jacobi 131 s vs CPU Jacobi 171 s (1.31×). AMG remains
  unnecessary there.

### Host-memory walls

Two distinct limits, both measured:

1. **Assembler pattern.** FEC keeps sparsity in per-element-entry (COO) form —
   576 entries per HEX8 — so `asm_cpu` alone is 25.3 GB at 1.57M DOF and
   ~35 GB at 3.09M. The planned 3.09M cube exhausts the 60 GB host during setup
   (44 GB RSS observed mid-setup) before the solver ever runs.
2. **Stock Galerkin product.** AlgebraicMultigrid's fine-level `A*P`
   preallocates for the worst case and is OOM-killed at 1.57M. Fixed here for
   the GPU hierarchy by slab evaluation.

Both are host-side and affect GPU runs too, because the hierarchy is built on
the host. They — not device memory — are what stands between Carina and
multi-million-DOF implicit problems.

---

## 5. Explicit dynamics

Same torsion bar at seven refinements, central difference, CPU (24 threads)
against GPU. The time step scales as 1/N so the CFL number is fixed and only
the cost per step varies. Each run is two equal control intervals — the first
absorbs warm-up and kernel compilation, the second is measured.

| DOF | elements | CPU ms/step | GPU ms/step | ratio |
|---:|---:|---:|---:|---:|
| 39k | 10,240 | 1.84 | 1.42 | 1.30× |
| 122k | 34,560 | 5.62 | 1.86 | 3.02× |
| 530k | 160,000 | 23.55 | 6.63 | 3.55× |
| 1.42M | 439,040 | 63.0 | 17.67 | 3.57× |
| 2.96M | 933,120 | 131.1 | 37.63 | 3.49× |
| 5.35M | 1,703,680 | 231.4 | 68.8 | 3.36× |
| 7.81M | 2,500,000 | 341.5 | 100.5 | 3.40× |

- **The ratio saturates at ~3.4×.** It rises to ~3.55× by 530k DOF, then flat.
  Small problems underuse the device (1.30× at 39k) — launch overhead and
  insufficient parallelism dominate.
- **3.4× is close to the ceiling on this hardware**, not a software shortfall:
  the memory-bandwidth ratio between these two devices is ~3–4×, so explicit is
  running at roughly 80% of the achievable ratio.
- **7.81M DOF fits in 8 GB** — 2.5M elements, 100.5 ms/step.
- CPU thread scaling at 530k DOF, which is why the `--threads` note matters:

  | threads | ms/step | speedup | parallel efficiency |
  |---:|---:|---:|---:|
  | 1 | 232.9 | 1.0× | — |
  | 4 | 66.3 | 3.51× | 88% |
  | 8 | 37.9 | 6.14× | 77% |
  | 12 | 27.5 | 8.47× | 71% |
  | 24 | 23.55 | 9.89× | 41% (SMT) |

  A single-threaded CPU baseline would report the GPU at ~35× rather than
  3.55×. Note the 1-thread row is not simply "the same code, serialized":
  `fec_atomic_add!` skips the atomic entirely when `Threads.nthreads() == 1`,
  so the serial path is cheaper per element and the 9.9× scaling is if anything
  conservative.

---

## 6. Where the time goes

The matrix-free stiffness action is the kernel every GPU implicit variant rests
on — CG applies it once per iteration, the AMG V-cycle about six times. It was
profiled by ablation (patch out one suspected cost, accept wrong results, keep
the timing), because Fedora 44 packages no `rocprof` binary. Arms, patches and
raw output: `benchmark/evidence/action_ablation.txt`; driver
`benchmark/action_bench.jl`.

At 530k DOF the action costs **9.50 ms**, of which:

| component | ms | share |
|---|---:|---:|
| constitutive derivative + geometry + interpolation + contraction | 7.1 | 75% |
| gather + scatter + kernel launch | 2.4 | 25% |
| FP64 atomics specifically | ~0 | 0% |

Three findings worth carrying forward:

- **The kernel is FP64-compute-bound, not bandwidth-bound.** It moves ~15 GB/s
  against the card's 288 GB/s, but that low fraction is arithmetic intensity,
  not waste: a matrix-free method is meant to trade bytes for flops. A FLOP
  budget puts it at **roughly 40–50% of this card's FP64 peak** (~0.68 TFLOP/s
  at the 1/32 rate). FP64 throughput is the binding ceiling.
- **The FP64 atomic scatter costs nothing.** Replacing every atomic with a
  plain non-atomic add — wrong results, valid timing — changes the kernel time
  by 1.4%, inside the noise, at both the pre- and post-optimization operating
  points. Element coloring and conflict-free E-vector restriction, the standard
  remedies for atomic contention, have nothing to buy here.
- **Forming the material tangent was 73% of the original cost.** The action
  built all 81 components of ∂P/∂∇u and contracted them with a single vector.
  It now takes the directional derivative `dP = ∂P/∂∇u : ∇v` directly, via one
  forward-mode dual pass over `pk1_stress` — the standard matrix-free Jacobian
  application used by libCEED/MFEM/Ratel. That gave **3.17× on the kernel and
  2.1× end-to-end**, with the operator unchanged: bit-identical device output at
  finite strain, and the 1e-12 assembled-vs-action test still passes.

  This applies to stateless models — every `Hyperelastic`. Models carrying state
  (J2 plasticity) still form the tangent, because `pk1_stress` runs a return map
  against Float64 containers that cannot hold dual numbers.

---

## 7. Open work, in priority order

1. **FP32 action inside the preconditioner.** The V-cycle's smoothing and
   residuals do not need FP64; an FP32 action wrapped in flexible CG keeps the
   outer solve in FP64. Arithmetic becomes ~32× cheaper on that part, so the
   kernel would bottom out at its 2.4 ms memory floor — bounded ~4× on the
   action and ~2× on the AMG solve phase. Given §6 this is much the largest
   remaining item.
2. **Analytic directional derivative** for NeoHookean, removing the dual-number
   overhead (~1.5–2 ms of the 9.5). Costs generality — it is per-model — so
   measure item 1 first.
3. **Cheaper geometry.** Precompute per-quadrature-point `∇N_X`/`JxW` rather
   than recomputing the Jacobian inverse every action (~102 MB of traffic for
   ~1 ms of arithmetic — only pays if that coalesced read runs near peak
   bandwidth, so measure before implementing); and exploit the structural
   sparsity of `discrete_gradient`, which builds a 24×9 matrix with 3 nonzeros
   per row and then multiplies it densely.
4. **Host-memory remediations** (§4): Int32 assembler pattern indices (halves
   the 25–35 GB pattern), dedup-to-CSR at construction, and a pattern-free
   `asm_cpu` mode for GPU runs that never assemble on the host. These gate
   multi-million-DOF problems.
5. **Ablate the explicit kernel** the way §6 ablated the implicit one. Explicit
   evaluates stress rather than a tangent, so it has no equivalent 73% to
   reclaim, but its constitutive share has never been measured.
6. **Smoother tuning**: the device's damped Jacobi needs 951 iterations where
   CPU Gauss–Seidel needs 787 — ~20% via Chebyshev-polynomial smoothing (the
   machinery exists), ν/cycle-shape tuning, or ℓ1-Jacobi.
7. **Map the Newmark large-Δt crossover.** AMG should win once Δt grows enough
   that `c_M` stops conditioning the system; that Δt is unknown.
8. **CUDA validation.** The implementation is KernelAbstractions-portable and
   contains no ROCm-specific paths, but only ROCm has been exercised.
9. **Profile** for occupancy and register pressure, which ablation cannot reach.
   `rocprofv3` is not packaged by Fedora but ships in AMD's own ROCm
   repositories and is usable from a container.
10. **Upstream candidates**: the `_slab_galerkin` memory fix
    (AlgebraicMultigrid.jl would benefit directly) and a KernelAbstractions.jl
    issue for the `KA.@index` qualified-macro CPU-backend miscompilation.

---

## 8. What is implemented

Pure Julia throughout: AlgebraicMultigrid.jl for hierarchy setup,
KernelAbstractions kernels for every device operation, Krylov.jl for CG. No
hypre, Trilinos, AmgX, or rocSPARSE, and no vendor-specific code paths — ROCm is
the test vehicle, CUDA follows from KA + Adapt with no code changes.

The GPU AMG preconditioner builds its hierarchy on the host with smoothed
aggregation, near-nullspace = six rigid-body modes evaluated at the **current**
configuration, rebuilt lazily on `c_M` change or CG-iteration growth. The
hierarchy converts to device CSR (Int32 indices) and the V(2,2)-cycle applies
entirely on the device: the fine level smooths through the matrix-free stiffness
action and extracted diagonal, so the fine matrix is never formed on the device;
coarse levels are assembled CSR; the coarsest is a dense `pinv` matvec. The
apply path allocates nothing. Symmetry and positivity of the cycle are verified
numerically in `test/gpu-amg-vcycle.jl`, and a checked-in test asserts that 20
repeated V-cycle applications leave live VRAM exactly unchanged.

Design rationale and rejected alternatives: `benchmark/design.md`.

---

## 9. Provenance

The GPU AMG implementation and its original benchmark passed a three-round
adversarial review in August 2026; the stiffness-action work in §6 and the
current numbers passed a further two-round review. Both reviews' findings are in
the git history.

Two conclusions from earlier versions of this report have since been overturned
by measurement, noted here because they circulated:

- GPU AMG was reported as **tied** with CPU AMG (299 s vs 302 s). That was
  before the action rewrite; it is now 2.2× faster.
- The FP64 atomic scatter was named as the reason the action underused the
  device. Ablation refuted it (§6); forming the material tangent was the real
  cost.

Raw records: `benchmark/results/current.jsonl` holds the implicit numbers quoted
here, `explicit-scaling.jsonl` the explicit sweep, `threadcheck.jsonl` the thread
scaling. Earlier tags (`baseline`, `proposed`, `scaling2`, `variance`, `detail`,
`bisect`, `nbuilds-check`, `jvp`) are kept so the history stays auditable.
