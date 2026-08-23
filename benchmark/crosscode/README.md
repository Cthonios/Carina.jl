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
  fixing it brought the A100 to 4.76 s/step and the RX 7600 to 6.17.)
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
