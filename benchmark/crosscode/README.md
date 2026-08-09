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

| code | solver | device | ways | per-step |
|---|---|---|---:|---:|
| **Carina** | CG+Jacobi | GPU | — | **7.07 s** |
| Carina | CG+Chebyshev | GPU | — | 8.60 s |
| Carina | L-BFGS | GPU | — | 9.74 s |
| **LCM** | Belos GMRES + ILUT | CPU | 12 rank | **14.40 s** |
| LCM | Belos GMRES + ILUT | CPU | 24 rank | 16.27 s |
| Carina | CG+Jacobi | CPU | 12 thr | 18.14 s |
| **Carina** | CG+Jacobi | CPU | 24 thr | **18.53 s** |
| Carina | CG+Jacobi | CPU | 4 thr | 19.91 s |
| Norma | Hessian min / Newton | CPU | 24 thr | 22.70 s |
| Carina | CG+Jacobi | CPU | 1 thr | 23.08 s |
| Carina | CG+AMG | CPU | 24 thr | 26.08 s |
| Carina | CG+IC | CPU | 24 thr | 26.92 s |
| Norma | Hessian min / Newton | CPU | 1 thr | 39.78 s |
| LCM | Belos GMRES + ILUT | CPU | 1 rank | 66.40 s |
| Carina | direct | CPU | 24 thr | 114.43 s |

- **Carina GPU is the fastest configuration measured**, 7.07 s/step: 2.6× its
  own best CPU and 2.0× LCM's best. That is the GPU campaign's thesis holding
  against two mature codes on identical input.
- **Core for core Carina leads**: 23.08 s/step against Norma's 39.78 (1.72×)
  and LCM's 66.40 (2.88×).
- **LCM's best beats Carina's best on CPU**, 14.40 vs 18.53 s/step (1.29×).
  Not because its per-core work is faster — it is 2.9× slower per core — but
  because it converts 12 ranks into 4.6× while Carina converts 24 threads into
  1.25×. §3 is about that.
- LCM at 24 ranks is *slower* than at 12: the box has 12 physical cores, so the
  second hardware thread per core buys nothing here.
- Norma and Carina CPU are closer than either is to the GPU — 22.70 vs 18.53,
  a 1.22× gap, smaller than Carina's own spread across preconditioners.

---

## 3. Carina's CPU implicit path does not thread

| threads | per-step | speedup | parallel efficiency |
|---:|---:|---:|---:|
| 1 | 23.08 s | 1.00× | — |
| 4 | 19.91 s | 1.16× | 29% |
| 12 | 18.14 s | 1.27× | 11% |
| 24 | 18.53 s | 1.25× | 5% |

A monotone curve that plateaus almost immediately — not noise. Set against
`benchmark_report.md` §5, which measures **9.9×** at 24 threads for the explicit
kernel on this same mesh, the contrast localises the cause precisely.

The two paths differ in exactly one thing. `src/input_parsing.jl:977` reads

```julia
assembled = backend isa KA.CPU
```

so a CPU run always assembles a sparse matrix and applies it with
`SparseArrays.mul!`, while GPU and explicit runs use the matrix-free action.
Julia's `SparseMatrixCSC` SpMV is single-threaded. Measured on the actual
K_eff this problem builds (530,523 DOF, 40.2M nonzeros) by
`benchmark/cpu_operator_bench.jl`:

| | 1 thread | 24 threads | scaling |
|---|---:|---:|---:|
| assembled SpMV | 13.78 ms | 14.12 ms | **1.0×** |
| matrix-free action | 353.85 ms | 32.75 ms | **10.8×** |

The two agree to `3.7e-16` — the same K_eff applied to the same vector — so
this is a like-for-like timing, and also a correctness check on the CPU
matrix-free path.

`benchmark_report.md` §3 records 2,558 CG iterations over 4 steps for this
variant, ~640 per step, so the serial SpMV is **~9.0 s of a measured 18.5 s
step, about half**.

### Matrix-free is not the fix

An earlier draft of this file asserted the SpMV was essentially the whole step
and that switching CPU to the matrix-free path was a one-line fix. Both claims
were wrong, and the measurement above is what corrects them.

The first came from timing a `sprand` matrix of the same size and density
rather than the real one: 33 ms against the true 13.8 ms. A random sparsity
pattern has far worse locality than an assembled FE operator, so it
overstated the cost by 2.4× and inflated the SpMV's share to ~100%.

The second is refuted outright. Matrix-free threads well — 10.8×, in line with
the explicit kernel's 9.9× — but it starts 25.7× behind and is still **2.32×
slower than the SpMV at 24 threads**. Flipping `assembled` on CPU would make
this path slower, not faster. Matrix-free wins on the GPU because FP64 SpMV
bandwidth is the scarce resource there and arithmetic is nearly free; on a CPU
with 24 threads against a well-ordered 40M-nonzero matrix, that trade runs the
other way.

### What the fix would actually be

A **threaded SpMV**, not a different operator. The row-parallel CSR form is
standard and is what Tpetra gives LCM; Julia's `SparseArrays` simply does not
provide it. That is a bounded piece of work against a known 14 ms kernel.

Note also that Amdahl caps the payoff: with the SpMV at ~48% of the step and
everything else perfectly parallel, the ceiling is ~2.0×, and the measured
scaling is 1.25×. So the SpMV is the largest serial block but not the only one
— the remaining ~9.5 s/step (preconditioner application, CG vector operations,
residual assembly, Newton overhead) has not been profiled, and some of it is
evidently serial too. Threading the SpMV alone would not reach LCM's 4.6×.

---

## 4. Caveats

- One problem, one size, one machine. Nothing here says anything about
  scaling, about other physics, or about these codes on a cluster.
- Albany/LCM is doing more work per step in a general sense: it is a
  multiphysics framework carrying Phalanx/Tempus machinery that Carina does not
  have. Read the rows as "these codes as configured for this problem", not as
  a statement about implementation quality.
- The GMRES+ILUT preconditioner LCM ships for this test is not the same
  algorithm as Carina's CG+Jacobi. Iteration counts are not comparable across
  codes; only wall time per step is.
- Quasi-static and explicit regimes are not covered. Norma has `QuasiStatic`
  and `CentralDifference` and LCM has an explicit Tempus input, so both are
  reachable; the implicit case was done first because all three codes already
  had a matching deck for it.

## 5. Reproducing

```
python3 run_crosscode.py                       # everything
python3 run_crosscode.py --only lcm --ways 12  # one cell
python3 run_crosscode.py --only carina --variants gpu-cg-jacobi --warmup
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
