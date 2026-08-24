# Cross-code torsion benchmark: Carina, Norma.jl, Albany/LCM

Where Carina stands against two established solid-mechanics codes on the same
problem, the same mesh, and the same machine.

Everything here is reproducible: `python3 run_crosscode.py`. Raw records in
`results.jsonl`, per-run logs in `logs/`.

---

## 1. The problem is identical in all three codes

| | Carina | Norma.jl | Albany/LCM |
|---|---|---|---|
| mesh | `torsion.g` | same file | same file |
| | 160,000 HEX8 / 530,523 DOF | | |
| material | neo-Hookean | neo-Hookean | `Model Name: Neohookean` |
| E, ν, ρ | 1e9, 0.25, 1000 | same | same |
| initial condition | `a=8000; -a*y*z` … | same | `About Linear Z, [8000.0]` |
| integrator | Newmark β=0.25, γ=0.5 | same | `Newmark Implicit a-Form`, same |
| Δt | 5e-5 | same | same |
| Newton | abs 1e-6, rel 1e-10 | same | same |

Norma's own torsion example is the same problem at 160 elements, generated from
the same `torsion.jou`, so block and node-set names already matched and Carina's
mesh dropped straight in. Albany reads it through `Method: Ioss`; the MPI runs
read the `decomp`-produced `torsion.g.<nranks>.<rank>` files.

**Each code keeps its own linear solver** — Carina CG+Jacobi/AMG/IC/direct,
Norma Hessian-minimizer/full-Newton, Albany Belos GMRES + Ifpack2 ILUT. Forcing
one code's solver onto another would measure something other than the codes.

**Timing method.** Every configuration is run at 4 *and* 8 steps and the
per-step cost reported as `(T_8 - T_4)/4`. Fixed costs here differ by more than
the quantity being measured — Julia pays ~40 s of JIT per process that Albany
does not — and the difference cancels all of them. Machine: Ryzen 9 9900X
(12 cores / 24 threads), Radeon RX 7600, Julia 1.12.6, Albany with OpenMPI 5.0.9.

---

## 2. Results

Carina's CPU numbers below are **after** the three fixes in §3, which the
first version of this benchmark surfaced. The "before" column is what the
cross-code comparison originally measured.

| code | solver | device | ways | per-step | before §3 |
|---|---|---|---:|---:|---:|
| **Carina** | CG+Jacobi | GPU | — | **7.07 s** | 7.07 |
| Carina | CG+Chebyshev | GPU | — | 8.60 s | 8.60 |
| Carina | L-BFGS | GPU | — | 9.74 s | 9.74 |
| **Carina** | CG+Jacobi | CPU | 24 thr | **10.79 s** | 18.53 |
| **LCM** | Belos GMRES + ILUT | CPU | 12 rank | **14.40 s** | — |
| Carina | CG+Jacobi | CPU | 1 thr | 14.99 s | 23.08 |
| LCM | Belos GMRES + ILUT | CPU | 24 rank | 16.27 s | — |
| Carina | CG+AMG | CPU | 24 thr | 21.29 s | 26.08 |
| Norma | Hessian min / Newton | CPU | 24 thr | 22.70 s | — |
| Carina | direct | CPU | 24 thr | 23.08 s | 114.43 |
| Carina | CG+IC | CPU | 24 thr | 24.32 s | 26.92 |
| Norma | Hessian min / Newton | CPU | 1 thr | 39.78 s | — |
| LCM | Belos GMRES + ILUT | CPU | 1 rank | 66.40 s | — |

- **Carina GPU is still the fastest configuration measured**, 7.07 s/step at
  the commit this table snapshots.  (§4: an A100 initially did barely
  better — 6.71 — the reason was Carina's kernel, not the hardware, and
  fixing it brought the A100 to 4.76 s/step and the RX 7600 to 6.17; the
  operator-fusion round that followed brought the A100 to 2.96 and the
  RX 7600 to 5.52, device-resident CG brought the A100 to 2.05, and taking
  the per-step Exodus write out of the loop — a cost Norma's decks never
  paid — brought it to 1.37 against a best CPU of 9.85 measured the same
  way: 7.2×.)
- **Carina's best CPU now beats LCM's best**, 10.79 against 14.40 s/step
  (1.33×), where it was 1.29× behind before §3.
- **Core for core Carina leads by more**: 14.99 s/step against Norma's 39.78
  (2.65×) and LCM's 66.40 (4.43×).
- **A single Carina CPU thread is now within 4% of LCM's twelve ranks**
  (14.99 vs 14.40), which is the sharpest way to state what the fixes did.
- LCM at 24 ranks is *slower* than at 12: the box has 12 physical cores, so the
  second hardware thread per core buys nothing here.
- The direct solver moved most in relative terms, 114.43 → 23.08 s/step, and is
  now competitive with the iterative variants rather than five times worse.
- Norma is untouched by any of this and is the honest control: it and Carina's
  CPU path were within 1.22× before, and are 2.1× apart after.

---

## 3. What the comparison found in Carina

The first run of this benchmark showed Carina's CPU implicit path scaling only
1.25× over 24 threads, against 9.9× for the explicit kernel
(`benchmark_report.md` §5). Chasing that gap found three defects. None was a
parallelism problem; two were wasted work and one was a stale premise repeated
in three places.

### The measurement that started it

`benchmark/cpu_step_profile.jl` accounted for only 10.90 s of a step the run log
measured at 19.50 s. The missing 8.6 s was the first defect.

### 1. The same sparse matrix built three times

`setup_jacobian!` passed `FEC.stiffness(asm)` separately into each of three
preconditioner updates. At most one does work — the others resolve to
`::Preconditioner` no-op fallbacks — but Julia evaluates arguments eagerly, so
the COO → CSC conversion ran three times and two results were discarded. At
530k DOF that conversion is 509 ms over 40.2M nonzeros: ~1.0 s wasted per
Newton iteration, ~16% of the step. The quasi-static path had the same shape.

A sweep for the pattern elsewhere found no other instances. The repeated
`FEC.diagonal(asm)` pairs are *not* this bug — each follows a different
`assemble_diagonal!` — and the matrix-free `_update_*_precond_*!` family passes
references rather than computed values.

### 2. A serial operator where a parallel one was available

`SparseArrays.mul!` on a `SparseMatrixCSC` is single-threaded and cannot easily
be otherwise: CSC walks columns and scatters into `y`, so parallel columns
collide. CSR walks rows and reduces into `y[i]`, which is conflict-free — the
reason Tpetra gives LCM a parallel SpMV and Julia's stdlib does not.

| | 1 thread | 24 threads | scaling |
|---|---:|---:|---:|
| `Symmetric(S,:L)` mul! (what CG applied) | — | 15.31 ms | — |
| `SparseArrays` CSC mul! | 14.47 ms | 14.20 ms | 1.0× |
| threaded CSR mul! | 12.71 ms | **9.43 ms** | 1.35× |

Switching cost nothing to build: `_csr_mul!` already existed in `gpu_amg.jl` as
a KernelAbstractions kernel, and the CPU backend threads it.

### 3. One stale claim, three decisions

The code asserted in three places that FEC's assembly is "~1e-7 asymmetric (AD
material tangent)". It is not. Measured asymmetry is **1.0e-16**, in
quasi-static (`c_M = 0`) as much as Newmark. The likely origin is that
`issymmetric(K)` returns `false` — it tests exact equality, so a 1e-16
perturbation trips it. That one misread predicate propagated into:

- symmetrizing `(K+K')/2` per Newton iteration, at 626 ms a call;
- forcing `Symmetric((K+K')/2, :L)` as the Krylov operator, which also blocks
  the CSR form above, since a symmetric matrix's CSC arrays *are* its CSR
  arrays and no transpose is needed;
- choosing `lu` over `cholesky` in the direct solver — 37.8 s against 6.7 s a
  factorization, and Cholesky is *more* accurate (4.91e-15 vs 1.67e-14).

### Matrix-free is not the fix

An earlier draft of this file asserted the serial SpMV was essentially the whole
step and that switching CPU to the matrix-free path was a one-line fix. Both
claims were wrong.

The first came from timing a `sprand` matrix of matching size and density rather
than the real one: 33 ms against the true 13.8 ms. Random sparsity has far worse
locality than an assembled FE operator, so it overstated the cost by 2.4× and
inflated the SpMV's share from ~48% to ~100%.

The second is refuted by measurement. Matrix-free threads well — 10.8×, in line
with the explicit kernel — but starts 25.7× behind and is still **2.32× slower
than the SpMV at 24 threads**. Matrix-free wins on the GPU because FP64 SpMV
bandwidth is the scarce resource and arithmetic is nearly free; on a CPU with 24
threads against a well-ordered 40M-nonzero matrix the trade runs the other way.

### What remains, and is a real ceiling

Thread scaling improved from 1.25× to 1.39×, not to LCM's 4.6×, and it will not
go much further. The SpMV is **memory-bandwidth bound**: 9.43 ms moves ~490 MB,
about 52 GB/s against the 50–90 GB/s this box achieves. One core nearly
saturates dual-channel DDR5, so core count was never the lever. The next gain
would have to move fewer bytes — Int32 column indices would cut ~160 MB per
apply, which is the one place the host-memory work and the speed work coincide.

### A note on how the Cholesky change nearly went wrong

The first attempt at defect 3 called `cholesky` per Newton iteration. Each call
allocates a fresh ~6.3 GB supernodal factor, and CHOLMOD allocates outside
Julia's heap — Julia sees a small wrapper, feels no pressure, and never
collects. The run reached 59.5 GB and was OOM-killed at step 6 of 8, taking the
session with it. The single-factorization benchmark that justified the change
measured time and never memory, which is the dimension that broke.

The shipped version builds the factor once and refactorizes in place with
`cholesky!`, valid because values change every Newton iteration while the
sparsity pattern does not. RSS is then flat across repetitions — the property
that matters — and it is faster still, since symbolic analysis is not repeated.

## 4. Cross-vendor check: the same problem on an A100

Run 2026-08-22 on ascicgpu073 (2x NVIDIA A100-PCIE-40GB, CUDA 13.0, 2x Xeon
Gold 6348), same repos at the same commits, same decks with `device: cuda`
(`run_crosscode.py --gpu-device cuda` rewrites the line).  The A100 offers
5.4x the memory bandwidth (~1555 vs 288 GB/s) and roughly 40x the FP64
throughput of the RX 7600.  What it bought:

| variant | RX 7600 | A100 | A100 advantage |
|---|---:|---:|---:|
| GPU CG+Jacobi | 7.07 s | **6.71 s** | 1.05x |
| GPU CG+Chebyshev | 8.60 s | 8.95 s | 0.96x |
| GPU L-BFGS | 9.74 s | 8.32 s | 1.17x |

Correctness carried over exactly: the same 4,455 CG iterations over the
8-step run, `|U|_max = 3.98e-02` in every variant, and the stiffness-action
checksum agrees with the RX 7600 to all 16 printed digits.

**Why the A100 barely wins.**  A CUPTI profile shows one monolithic kernel --
the matrix-free stiffness action -- is over 76% of device time, and the
isolated action timings are 9.36 ms (RX 7600) vs 7.17 ms (A100): 1.31x from
hardware that should give several times that.  The SASS explains it: the
kernel uses **253 registers per thread** (the architectural cap is 255), so a
256-thread block consumes 64,768 of an SM's 65,536 registers -- exactly one
block resident per SM, **12.5% occupancy** -- plus a ~4 KB stack frame per
thread with 475 local-memory spill instructions.  Only ~1,100 of its
instructions are FP64 arithmetic, so it is nowhere near flop-bound, and at
19 GB/s effective it is nowhere near bandwidth-bound.  Both GPUs are pinned
by the same kernel-internal wall: occupancy and spill latency.

Two consequences.  First, the earlier estimate that the action runs at
"40-50% of FP64 peak" on the RX 7600 (benchmark_report.md par.6) was an
inference from an assumed flop count, and the cross-vendor result refutes it
as the binding constraint: if FP64 rate bound the kernel, full-rate FP64
would have made it dramatically faster, and it did not.  Second, reducing
per-thread live state in that kernel -- splitting per quadrature point,
capping registers at launch, or staging through shared memory -- is now the
top GPU lever, it is worth potentially several x, and it pays on both vendors
at once.

**A measurement trap worth recording.**  The first A100 batch benchmarked
from the NFS-mounted home directory and reported 10.36 / 8.99 / 9.45 s/step.
Each 8-step run writes a 1.6 GB Exodus file, and on a shared NAS that cost is
neither constant nor reliably cancelled by the difference method -- write-back
caching absorbed Chebyshev's output entirely (8.99 -> 8.95 on local disk) but
charged Jacobi ~3.6 s/step (10.36 -> 6.71).  Both row sets are in
`results.jsonl` (`workdir: scratch` marks the local-disk rows, which are the
canonical ones); on shared-filesystem machines the benchmark must run from
node-local storage.


### The wall, removed (2026-08-22, commits `053b490` + `85928c5`)

The register diagnosis above was actionable, and an ablation ladder ran the
same day.  (1) A `maxregs` recompilation sweep on the A100 was flat from 96
to 255 registers and 3-5x worse below 64: forcing occupancy never pays, the
live state itself had to shrink.  (2) A closed-form NeoHookean directional
derivative replaced the ForwardDiff dual pass -- and the *form* mattered more
than the fact of being analytic: staged tensor temporaries ran 2.3x slower
than the dual pass on CUDA at near-identical static SASS, while the same
derivative collected into `dP = c1 F^-T + c2 W + c3 dF + c4 F` beats the dual
pass on both vendors.  (3) The real win: the discrete-gradient operator G --
a 3Nx9 matrix, 216 doubles for HEX8, built per quadrature point and used
once -- was eliminated in favor of a direct contraction.  Even the mass
kernel, whose only arithmetic is N'N*v, hit the 255-register cap: the
element machinery, not the constitutive math, was spending the registers.

On the A100 the spill frame collapsed from ~4 KB to 248 bytes per thread and
the action went **7.17 -> 2.88 ms** (within 2x of the mass-action memory
floor).  On the RX 7600 the action barely moved (9.36 -> 8.66 ms): RDNA3 was
never spill-bound -- its 1/32-rate FP64 arithmetic is its wall, and only
reduced precision can move it further.  One change, two vendors, two
different binding constraints revealed.  End to end, same 4,455 CG
iterations and `|U|_max` on every row:

| variant (per-step) | RX 7600 | A100 |
|---|---:|---:|
| GPU CG+Jacobi | 7.07 -> **6.17 s** | 6.71 -> **4.76 s** |
| GPU CG+Chebyshev | 8.60 -> 7.89 s | 8.95 -> 6.63 s |
| GPU L-BFGS | 9.74 -> 8.21 s | 8.32 -> 5.06 s |

The A100 now leads the RX 7600 by 1.30x on the best variant -- still far
from the 5.4x bandwidth ratio, because the two cards are limited by
different resources -- and Carina's best configuration is **4.76 s/step**,
2.3x its own best CPU and 3.0x LCM's best.  Rows in `results.jsonl` carry
`commit: 85928c5`.

### A third card as falsification test: V100 (2026-08-23, at `b5326a1`)

The "now memory-bound" claim above makes a prediction on other full-rate-FP64
hardware, so the same decks ran unchanged on a V100-PCIE-32GB
(ascicgpu24, driver 580, CUDA.jl auto-selects the 12.9 runtime for sm_70 --
CUDA 13 dropped Volta).  Pure DRAM-bandwidth scaling (1555/900 GB/s)
predicts the A100 kernel times x1.73; the V100 measured:

| kernel (min over 50 reps) | A100 | V100 | ratio |
|---|---:|---:|---:|
| stiffness action | 2.88 ms | 6.07 ms | 2.11x |
| mass action | 1.49 ms | 3.46 ms | 2.32x |

No Volta-specific register cliff: the action sits at 1.74x its own
mass-kernel floor (A100: 1.93x), and both checksums match the other two
cards to the last digit.  But *both* kernels -- including mass, which does
almost no arithmetic -- run 20-35% worse than bandwidth scaling.  The
missing factor is L2 capacity: 40 MB on the A100 against 6 MB on the V100,
and the dominant traffic in both kernels is the redundant 24-DOF
connectivity gather, whose reuse the A100 catches in L2 and the V100 sends
to HBM.  So the diagnosis survives, refined: the kernels are bound by the
memory system, not just the DRAM pins, and the shared-gather work item
gains a second motivation.

End to end (same invariants: 4,455 CG iterations, `|U|_max = 3.98e-02`):

| variant (per-step) | RX 7600 | V100 | A100 |
|---|---:|---:|---:|
| GPU CG+Jacobi | 6.17 s | 10.46 s | 4.76 s |
| GPU CG+Chebyshev | 7.89 s | 9.06 s | 6.63 s |
| GPU L-BFGS | 8.21 s | **7.09 s** | 5.06 s |

Two things worth keeping.  First, the variant ordering flips on the V100:
CG+Jacobi, the winner on both other cards, is worst here, and L-BFGS wins.
Jacobi leans hardest on the action kernel (4,455 CG iterations per 8
steps), so a 2.1x kernel slowdown costs it most; the falsification test
doubles as a reminder that the best solver variant is hardware-dependent.
Second, the V100 loses to the $270 RX 7600 on two of three variants despite
a 1.4x faster action kernel.  The gap is bigger than any kernel time
explains: everything the host still drives inside the step loop -- kernel
launches at thousands of CG iterations per step, line search, per-iteration
bookkeeping -- runs on 2.1 GHz Skylake cores here against the 5.7 GHz Zen 5
driving the 7600.  Untangling launch overhead from host sections needs a
profile, but either way it is an argument for keeping the CPU out of the
step loop.  Rows in `results.jsonl` carry `host: ascicgpu24`.

### Where a step actually goes: CUPTI profile of one production step (2026-08-23)

`CUDA.@profile` around one full Newmark step on the V100 (3 Newton
iterations, 579 CG iterations, one Exodus write; 10.33 s trace, GPU busy
70.5%):

| component | time | share |
|---|---:|---:|
| stiffness action, 579 x 5.88 ms | 3.41 s | 33% |
| mass action, 583 x 3.31 ms | 1.93 s | 19% |
| Jacobi diagonal assembly, 3 x 585 ms | 1.76 s | 17% |
| all other kernels combined | 0.15 s | 1.5% |
| GPU idle (host: Exodus write + per-iteration sync logic) | 3.05 s | 30% |

Three conclusions, one refutation:

1. **Launch overhead is a non-issue.**  All 11,200 launches cost 55 ms
   combined; the slow-host penalty conjectured above is *not* launch
   dispatch.  It is the ~30% GPU-idle bucket: 1,215 device-to-host scalar
   copies (a convergence-check synchronization every CG iteration) plus
   the per-step Exodus write, all serialized on 2.1 GHz cores.
2. **The mass action is a separate full element kernel every CG
   iteration.**  Newmark applies `A v = c M v + K v` as two kernels that
   walk the same connectivity, gather the same 24 DOF, and scatter to the
   same nodes.  The mass kernel is nearly pure gather/scatter traffic --
   fusing it into the stiffness action should recover most of its 19% on
   every card.
3. **The Jacobi preconditioner costs 585 ms per Newton iteration** --
   3.7 us/element, ~100x the action kernel per element.  It computes the
   full 24x24 element matrix (an assembled-path site that still builds G)
   and keeps 24 numbers of it.  A diagonal-only kernel is ~10x less work.

The refutation closes the cross-vendor cache story.  The RX 7600's mass
action measures 2.47 ms -- *faster than the V100's 3.31* on a third of the
bandwidth -- because Navi 33 carries a 32 MB Infinity Cache.  The
effective-cache ordering A100 (40 MB L2) > RX 7600 (32 MB IC) > V100
(6 MB L2) exactly matches the mass-kernel ordering 1.49 < 2.47 < 3.46 ms:
the gather-dominated kernels are cache-capacity-bound, and the V100 is the
outlier because it is the only card without a large last-level cache.
(The stiffness action does not follow the same ordering because on the
7600 it is 70% FP64-arithmetic-bound -- two limits, one kernel, which card
you run picks which limit binds.)

### Both targets removed (2026-08-23, commits `b4712e6` + `247586f`)

The profile's two targets fell the same day.  `NewmarkAction(c_M)` applies
`(K + c_M·M)·v` in one element pass -- one gather, one quadrature loop, one
atomic scatter, and one launch+sync per CG iteration where there were two of
everything.  Form mattered once again: the first version composed the two
existing kernels through dispatch and NVPTX failed to fold the duplicated
element-Jacobian inversion (8.76 ms on the V100, worse than the 5.95 ms
stiffness term should dominate); an explicit body sharing the mapped cell
lands at 6.50 ms, the whole mass term costing +0.55 ms.  On the A100 the
fused kernel is 2.45 ms -- the mass term now costs +0.16 ms against 1.57 ms
as a separate kernel.  Second, `NewmarkDiagonal(c_M)` returns the
per-quadrature-point diagonal of `K + c_M·M` directly (diag(JxW·G·A·G')
needs only the tangent entries pairing a component with itself), so the
preconditioner update never forms a 24×24 element matrix: 587 → 12.4 ms on
the V100 (47×), 290 → 6.5 ms on the A100 (45×), 44 → 21 ms on the 7600
(2.1× -- there the tangent FLOPs, common to both, dominate).  Fused-vs-
two-pass agreement is 3e-16; every run below holds the invariants (4,455 CG
iterations, `|U|_max = 3.98e-02`).

| variant (per-step) | RX 7600 | V100 | A100 |
|---|---:|---:|---:|
| GPU CG+Jacobi | 6.17 -> 5.52 s | 10.46 -> **5.40 s** | 4.76 -> **2.96 s** |
| GPU CG+Chebyshev | 7.89 -> 7.20 s | 9.06 -> 7.11 s | 6.63 -> 3.01 s |
| GPU L-BFGS | 8.21 -> 8.47 s | 7.09 -> 7.88 s | 5.06 -> 4.96 s |

Carina's best configuration is now **2.96 s/step** (A100, CG+Jacobi), 3.6x
its own best CPU -- and 2.3x faster than the same hardware ran two days
earlier.  The gains land exactly where the profile said the costs were:
the V100, whose step was 19% mass kernel + 17% diagonal assembly, nearly
halves on Jacobi and now ties the RX 7600 (5.40 vs 5.52) despite its host
handicap; the A100's Chebyshev, which applies the operator inside the
smoother polynomial too, gains 2.2x.  L-BFGS, which uses neither fix, is
unchanged within its run-to-run scatter (the V100 measured 7.09 / 8.62 /
7.88 s across three runs of identical code -- that variant's difference is
host-dominated and noisy on GPFS-homed machines).  Rows in `results.jsonl`
carry `commit: 247586f` (`b4712e6` for L-BFGS).

### The host leaves the loop: device-resident CG (commits `5f9ec61` + `df8ff62`)

Re-profiling after the fusion showed the A100's step had inverted: device
busy only 36% of the trace, with 2,430 `cuStreamSynchronize` calls -- every
CG iteration read 3-4 scalars back through blocking copies that drain the
pipeline, and the faster the kernels got, the larger that share became.
`_device_pcg!` replaces Krylov.jl on the matrix-free path: recurrence
scalars live in 1-element device arrays (on-device two-stage tree-reduction
dots, 1-element broadcasts for α and β), and the host reads back one number
per convergence-check block.  A fixed 8-iteration block overran expensively
for Chebyshev, whose iterations each apply a five-matvec smoother
(7.20 → 7.62 s on the 7600), so the block is sized from the measured
contraction rate -- on the A100 the predictor lands on exactly Krylov's
4,455 iterations, zero overrun.

| variant (per-step) | RX 7600 | V100 | A100 |
|---|---:|---:|---:|
| GPU CG+Jacobi | 5.52 -> 5.57 s | 5.40 -> **4.71 s** | 2.96 -> **2.05 s** |
| GPU CG+Chebyshev | 7.20 -> 7.19 s | 7.11 -> 6.82 s | 3.01 -> 3.10 s |

The pattern is the diagnosis confirmed a third way: the win scales with the
per-iteration host gap.  The A100 (2.7 ms gap per 2.6 ms of kernel) gains
1.44x; the V100 (0.8 ms gap) gains 1.15x; the RX 7600, driven by a 5.7 GHz
host with nothing to reclaim, is a wash on both variants -- as is Chebyshev
on the A100 (samples 3.10/3.28 vs Krylov's 3.01/3.09), whose few heavy
iterations never paid much sync tax per second.  Invariants hold on every
row.  Carina's best configuration is now **2.05 s/step** (A100, CG+Jacobi):
5.3x its own best CPU, 7.0x LCM's best, and 3.3x what the same A100 ran
before this two-day sequence of profile-directed fixes.  Rows carry
`commit: df8ff62`.

### The tax nobody was counting: per-step Exodus writes (commit `37f0bea`)

With the CG loop device-resident, the next-largest term in the A100's step
was not compute at all.  Every Carina benchmark row so far included a
per-step Exodus write -- displacement plus per-quadrature-point stress and
deformation gradient (15 element variables x 8 qp), recomputed on the host
each step -- while Norma's decks in this very comparison ran with
`Exodus output interval: 0` and wrote nothing.  Carina's top-level
`output interval` key (a time period in seconds, as in Norma; the
integrator already subcycles between output stops) makes the cost
avoidable: `output interval: 1.0` on a 4e-4 s run writes exactly two
frames, initial and final.  The same commit fixes a truncation bug -- a
period that did not divide the span used to silently end the run early
(`round` in the stop count; interval 3e-4 on a 4e-4 s run stopped at
3e-4 s) -- and rejects non-positive or non-numeric intervals loudly.

Removing the writes also broke the measurement method: the per-step signal
(3-14 s across an n4/n8 pair) sank below the setup jitter of the
GPFS-backed hosts (about +/-1 s per run, amplified 4x by the difference).
The estimator therefore moved from whole-process walls to the `[STOP]`
line wall Carina logs per output stop, which starts after setup and whose
single final write cancels between the pair.  Two samples per GPU
reproduce to ~3%:

| GPU CG+Jacobi (per-step) | RX 7600 | V100 | A100 | CPU 24 thr |
|---|---:|---:|---:|---:|
| with per-step writes | 5.57 s | 4.71 s | 2.05 s | 10.79 s |
| `output interval: 1.0` (2 frames) | 5.00 s | 3.56 s | **1.37 s** | 9.85 s |
| implied write cost per step | 0.57 s | 1.15 s | 0.68 s | 0.94 s |

(Samples: A100 1.34/1.41, V100 3.52/3.60, RX 7600 4.90/5.10; CPU one
sample.)  The implied write cost is host-serial work and sits in the same
0.6-1.2 s band on all four hosts, which is the cross-check that the new
estimator and the old rows are measuring the same thing.  Invariants hold
on every run (`|U|_max = 3.98e-02`).  Carina's best configuration is now
**1.37 s/step** (A100, CG+Jacobi, output at start and end only): 7.2x its
own best CPU measured the same way (9.85), and 4.9x what the same A100 ran
at the start of this sequence (6.71).  Rows carry `commit: 37f0bea`,
`method: stop_wall`.

### Per-qp threading falsified: the matvec is at its memory-system bound (commit `97083b3`)

With writes gone, a CUPTI profile of a write-free step showed the step IS
the fused stiffness matvec: 67% of device time in one kernel, ~557 CG
iterations x 2.36 ms accounting for essentially the whole 1.37 s, host
API time almost entirely waiting.  The obvious device-side lever was the
lgrtk two-phase pattern -- one thread per (element, qp) writing disjoint
staging (8x the threads, 1/8th the per-thread state, no atomics), then a
node-parallel gather over an inverse adjacency.  Implemented behind an
off-by-default switch (`src/two_phase_action.jl`), matching the fused
kernel to 1e-12 at a deformed state, and measured on all three cards:

| NewmarkAction matvec (median) | A100 | RX 7600 | V100 |
|---|---:|---:|---:|
| fused one-thread-per-element | 3.04 ms | 9.89 ms | 6.59 ms |
| two-phase per-(element, qp) | 3.80 ms | 12.57 ms | 10.05 ms |
| ratio | 0.80x | 0.79x | 0.66x |

It loses everywhere, and the loss ordering is the §4 cache story again:
the V100's 6 MB L2 absorbs least of the 8x-duplicated element gathers.
Combined with the earlier flat atomic ablation, this pins the diagnosis:
the fused kernel is memory-system-bound on the GATHER side -- not
occupancy-bound, not atomic-bound -- and a thread-mapping change fixes
neither cost while adding a ~500 MB staging round trip per matvec.  The
remaining per-iteration levers are data layout (element-blocked field
storage, a much larger change) or simply taking fewer iterations: GPU
AMG on the 4,455-iteration count, which was the founding goal all along.

### GPU AMG: falsified for implicit dynamics, 2x for quasi-static steady state

With the matvec at its memory-system bound, the last lever is taking fewer
of them.  The GPU AMG preconditioner (host-built smoothed-aggregation
hierarchy, fully device-resident V(2,2)-cycle, matrix-free fine level) won
the earlier quasi-static campaign but had never been tried on the Newmark
operator, and had never met the device-resident CG.  Both were measured at
`0e7441d`; no code changed.

On the **Newmark** benchmark AMG loses on both cards -- A100 1.37 -> 2.01
s/step, RX 7600 5.00 -> 6.75 -- because the mass-dominated
K + c_M*M at dt = 5e-5 is already so well-conditioned that Jacobi's ~557
CG iterations per step sit below the ~6x cost of a V-cycle per iteration.
This closes, with two-architecture measurements, the code comment that
implicit dynamics "defaults to Jacobi with no evidence this path would
pay": the default is now evidence-backed.  The 1.37 s/step Jacobi record
stands.

On **quasi-static** torsion, AMG's home ground, the A100 shows the split
this benchmark's 4-step span hides poorly: steady-state steps run
**5.1-5.3 s against Jacobi's 10.0-10.8** (2x, identical |U|_max at every
stop), but the one-time host-side hierarchy build costs 20-25 s, so the
4-step TOTALS tie (130-136 s both).  `--threads 24` trims the build only
24.6 -> 20.3 s: the time is AMG.jl's serial setup (strength, aggregation,
stdlib SpGEMM Galerkin products) plus a dense pinv, not the threaded FEC
assembly.  On any production-length run the build amortizes away and the
2x stands; making short runs win too means a threaded or device-side
SpGEMM for the setup, which is its own project.

## 5. Caveats

- One problem, one size, one machine (plus the A100 and V100 cross-checks
  of §4).
  Nothing here says anything about
  scaling, about other physics, or about these codes on a cluster.
- Albany/LCM is doing more work per step in a general sense: it is a
  multiphysics framework carrying Phalanx/Tempus machinery that Carina does not
  have. Read the rows as "these codes as configured for this problem", not as
  a statement about implementation quality.
- The GMRES+ILUT preconditioner LCM ships for this test is not the same
  algorithm as Carina's CG+Jacobi. Iteration counts are not comparable across
  codes; only wall time per step is.
- Only Carina's CPU rows changed between the two columns in §2. Norma and LCM
  were measured once and are untouched, which makes them the control: if the
  Carina improvements were a measurement artifact rather than real, the
  unchanged codes would have drifted too, and they did not.
- Until `37f0bea` every Carina row included a per-step Exodus write
  (displacement plus per-QP stress and F, recomputed on the host), while
  Norma's decks ran with `Exodus output interval: 0` — no output at all.
  The `-nowrite` rows remove that asymmetry; earlier rows stand as
  measured.
- Quasi-static and explicit regimes are not covered. Norma has `QuasiStatic`
  and `CentralDifference` and LCM has an explicit Tempus input, so both are
  reachable; the implicit case was done first because all three codes already
  had a matching deck for it.

## 6. Reproducing

```
python3 run_crosscode.py                       # everything
python3 run_crosscode.py --only lcm --ways 12  # one cell
python3 run_crosscode.py --only carina --variants gpu-cg-jacobi --warmup
python3 run_crosscode.py --only carina --variants gpu-cg-jacobi \
        --warmup --gpu-device cuda            # on an NVIDIA host
```

`--warmup` runs and discards one extra run before the timed pair. It exists
because the first configuration of the first batch returned a *negative*
per-step: cold ROCm kernel compilation landed entirely on its 4-step run, so
the 8-step run finished faster. Rows carry `warmup: true/false`; superseded
cold-cache rows are kept in `results.jsonl` rather than deleted.

MPI runs need the decomposed mesh:

```
~/LCM/trilinos-install-serial-gcc-release/bin/decomp --processors 12 torsion.g
```
