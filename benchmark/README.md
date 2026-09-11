# GPU solver benchmark suite

Everything behind `benchmark_report.md` (repo root): input decks, mesh
generation, the measurement harness, sweep scripts, and the raw results.

## Layout

| Path | What it is |
| --- | --- |
| `inputs/` | Standalone YAML decks, one per (case, variant) pair of the implicit study — tracked and directly runnable, see below |
| `inputs/generated/` | Decks written by `explicit_sweep.jl`, whose content depends on the step count passed to it — gitignored scratch, regenerated on demand |
| `cases.jl` | Single source of truth for case/variant definitions (decks and harness both derive from it) |
| `write_inputs.jl` | Regenerates `inputs/` from `cases.jl` |
| `harness.jl` | Measurement harness: one (case, variant) per fresh process, appends a JSON-lines record to `results/<tag>.jsonl` |
| `action_bench.jl` | Times the matrix-free stiffness action against the mass action in isolation, on device — the microbenchmark behind the kernel analysis in the report.  Takes `[nreps] [initial\|deformed] [auto\|rocm\|cuda]` |
| `explicit_sweep.jl` | Explicit-dynamics CPU-vs-GPU harness: one (size, device) per fresh process, same JSON-lines contract |
| `meshgen.jl` | Structured HEX8 cube mesh generator for the large cases |
| `torsiongen.jl` | Structured HEX8 torsion-bar generator at arbitrary refinement (`N=20` reproduces `torsion.g`) |
| `run_baselines.sh`, `run_round2.sh`, `run_scaling.sh`, `run_scaling2.sh` | The sweep scripts the implicit study ran |
| `run_explicit_scaling.sh` | The explicit CPU-vs-GPU size sweep (report §8) |
| `MATRIX.md` | The rendered performance matrix — every device across the three regimes, commit-matched. Consult this first |
| `matrix.jl` | Drives the harnesses below over the full matrix on one machine and normalizes their records into `results/matrix/` |
| `matrix_report.jl` | Renders `results/matrix/` into `MATRIX.md` |
| `results/matrix/` | The current performance record: one schema, mandatory provenance, one file per (host, device) |
| `results/archive/` | Raw records of the optimization rounds — every number in the report traces to these |
| `evidence/` | Log excerpts and ablation arms backing specific report claims (OOMs, L-BFGS failure, ROCm test output, the action ablation, the FEC block-size sweep, the inexact-Newton A/B) |
| `design.md` | Proposed solution, design rationale, rejected alternatives |

## Performance matrix

The current, commit-matched record of every device across the three regimes
is `MATRIX.md`, rendered by `matrix_report.jl` from `results/matrix/`.  It is
the answer to "how do these GPUs compare for explicit dynamics, implicit
dynamics and quasi-statics", and it is the table to consult first; everything
below it in this file describes the individual harnesses the matrix drives.

```sh
julia --project=. benchmark/matrix.jl --part all       # measure this machine
julia benchmark/matrix_report.jl --write               # re-render MATRIX.md
```

The matrix has three parts.  The *spine* runs one mesh (`torsion.g`, 530k DOF)
through all three time integrators, so a device's three numbers differ only by
the integrator.  The *ladder* is the explicit refinement sequence, N = 8 to 64.
The *size* cases are the implicit analog, cube64 and cube80.  Every record
carries host, vendor, GPU model, commit, Julia version and thread count, and
the device memory in use before the point started; a failed point is recorded
with its reason rather than skipped.

## Meshes

The 530k-DOF torsion mesh is tracked at `examples/meshes/torsion/torsion.g`
(Cubit journal alongside), as is the 81-DOF smoke cube.  The large cube
meshes are too big for git and are regenerated deterministically:

```sh
julia --project=. benchmark/meshgen.jl 64  benchmark/meshes/cube64.g    # 823k DOF, 16 MB
julia --project=. benchmark/meshgen.jl 80  benchmark/meshes/cube80.g    # 1.57M DOF, 30 MB
julia --project=. benchmark/meshgen.jl 100 benchmark/meshes/cube100.g   # 3.09M DOF, 59 MB (exceeds 60 GB hosts — report §3)
```

## Running a single case

Each deck in `inputs/` is self-contained (mesh path, BCs, solver, and a
`device:` key pinning the backend the study used — `rocm` for `gpu-*`
decks).  Run through the CLI launcher, which owns the GPU vendor packages:

```sh
bin/carina benchmark/inputs/torsion-qs-gpu-cg-amg.yaml            # as studied
bin/carina benchmark/inputs/torsion-qs-gpu-cg-amg.yaml --device cpu  # override backend
```

Outputs (`.e`, `.log`) land next to the deck and are gitignored.  The
`cube-qs-*` decks are the 81-DOF smoke case — seconds, good for checking a
setup.  Both `--device rocm` and `--device cuda` are exercised: ROCm on the
RX 7600, CUDA on the V100, A100 and L4.

For measured runs (iteration counts, phase timings, VRAM) use the harness
instead; it runs each combination in a fresh process and appends to
`results/<tag>.jsonl`:

```sh
julia --project=. benchmark/harness.jl torsion-qs cpu-cg-amg mytag   # CPU variants
JULIA_LOAD_PATH="$PWD:$PWD/bin:@stdlib" \
  julia benchmark/harness.jl torsion-qs gpu-cg-amg mytag             # GPU variants
```

## Explicit CPU-vs-GPU sweep

Separate from the implicit study: the same torsion bar at seven refinements
(39k to 7.8M DOF), central difference, CPU against GPU.

```sh
benchmark/run_explicit_scaling.sh mytag        # full sweep
julia --project=. benchmark/explicit_sweep.jl 20 rocm mytag 800 24   # one point
```

Meshes are generated on demand by `torsiongen.jl` into `meshes/`.  The time
step scales as 1/N so the CFL number is fixed across sizes and only the cost
per step varies.  Each run is two equal control intervals — the first absorbs
warm-up and device kernel compilation, the second is the measured one — and
output is stripped to nodal displacement with no recovery so the single Exodus
write inside the measured interval is negligible.  The CPU baseline uses 24
threads, which is faster than 12 at N=20 (23.7 vs 27.5 ms/step).

On a machine whose home directory is NFS/GPFS (the ascicgpu hosts), set
`CARINA_BENCH_SCRATCH=/scratch/...` so meshes, decks, outputs, and result
records land on a local disk.

### Cross-card reference (2026-08-25, commit `2b827db`)

The same ladder on the Sandia V100 (32 GB) and A100 (40 GB), extended past
the RX 7600's 8 GB capacity cap; records in
`results/archive/explicit-ascicgpu{24,073}.jsonl`.  Per-step milliseconds, with the
original sweep's CPU baseline (the 5.7 GHz desktop host, 24 threads — the
fastest CPU measured per core) alongside:

| N | DOF | desktop CPU | Rigel 48T | RX 7600 | L4 | V100 | A100 | A100 / CPU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 39k | 1.84 | 1.67 | 1.42 | 0.58 | 0.69 | 0.43 | 4.3x |
| 12 | 122k | 5.62 | 2.97 | 1.86 | 1.34 | 1.29 | 0.90 | 6.2x |
| 20 | 531k | 23.6 | 12.2 | 6.63 | 4.68 | 4.68 | 2.45 | 9.6x |
| 28 | 1.42M | 63.0 | 34.4 | 17.7 | 12.6 | 12.0 | 5.93 | 10.6x |
| 36 | 2.96M | 131.1 | 75.4 | 37.6 | 27.9 | 25.4 | 12.8 | 10.3x |
| 44 | 5.35M | 231.4 | 128.0 | 68.8 | 51.6 | 46.8 | 23.8 | 9.7x |
| 50 | 7.81M | 341.5 | 181.3 | 100.5 | 78.3 | 71.0 | 34.5 | **9.9x** |
| 64 | 16.2M | — | 368.0 | 220.0† | 175.0 | 148.7 | 73.7 | — |
| 72 | 23.0M | — | — | — | — | 219.0 | — | — |
| 80 | 31.5M | — | 708.7 | — | — | — | 158.7 | — |
| 100 | 61.2M | — | 1806 | — | — | — | — | — |

The L4 column is a 72 W inference card added to Rigel on 2026-09-08; records in
`results/archive/explicit-rigel-l4.jsonl`.  It matches the V100 within 10% from N=20 to
N=50.

† This table is the August record at commit `2b827db` and is superseded by
`MATRIX.md`, which re-measured every device at one commit in September.  One
entry changed in kind: the RX 7600 was out of memory at N=64 in August and
completes it in September at 220 ms/step, 16.2M DOF in 8 GB.  The July
`as_matrix_free` change cut a run's device footprint from 5.83 to 0.245 GB,
and the August sweep predates its reaching this path.  The same card also runs
both large implicit cases (cube64 at 1.06 GB, cube80 at 2.19 GB); the memory
that limits cube80 is the host's, during the AMG hierarchy build, not the
device's.

Rigel is a dual EPYC 9634 (168 cores / 336 threads, 1.5 TB); records in
`results/archive/explicit-rigel{,-threads}.jsonl`.  Its column is the machine's
measured optimum, which is **48 threads** — thread scaling INVERTS above
that (N=20: 17.6 ms at 24T, 12.0 at 48T, 24.7 at 84T, 138.8 at 168T,
545.6 at 336T).  The threaded CPU scatter uses atomic adds, and past ~48
threads across 8 NUMA domains the contention dominates; more elements per
thread softens it (at N=64, 168T reaches parity with 48T at 387 vs 368 ms)
but never pays.  The known cure is the two-phase scatter already
implemented for the implicit action (`src/two_phase_action.jl`, a loss on
GPUs where atomics were free) — worth porting to the CPU path only if
big-CPU-node explicit becomes a real target.

- **Per-element cost is flat everywhere** once past launch overhead:
  ~13.5–15.5 ns/elem (A100), ~27–29 (V100), ~29–33 (L4), ~40 (RX 7600),
  ~70–80 (Rigel at 48T), ~135–145 (desktop CPU).  No scaling cliff up to 31.5M
  DOF; memory capacity, not bandwidth, is the ceiling (~1.0–1.3 KB/DOF
  on every card measured).
- **Against the fastest CPU measured, the saturated ratios are A100 ~10x,
  V100 ~5x, RX 7600 3.4x** — stable across the size range because the CPU
  is flat per element too.  The CPU ladder stops at N=50; the A100 runs
  4x that problem.
- **Placement**: every GPU is faster than every CPU node measured (even the
  168-core Rigel runs 1.8x slower than the RX 7600 and 5.3x slower than
  the A100 at N=50).  The margin depends on the pairing: the L4 is only
  2.1–2.7x faster than the Rigel CPU beside it, against 7.4x for the A100
  over its host, because Rigel pairs a 72 W inference card with the fastest
  CPU node measured.  Rigel's niche is capacity: it runs 61.2M DOF at
  1.8 s/step — 2x past the A100's 40 GB ceiling and 3.8x past the L4's —
  so CPU nodes are for problems that do not fit a GPU, not for speed.
- **The explicit ordering is A100 > V100 ≈ L4 > RX 7600.** An earlier version
  of this file attributed that ordering to FP64 throughput, on the strength of
  the RX 7600's 1/32 rate.  The L4 refutes it: its FP64 peak is *lower* than
  the RX 7600's (0.489 vs 0.68 TFLOP/s) and it is 1.35x faster per element,
  and it has 1/16 the V100's FP64 peak while tying it.  Bandwidth does not
  explain the ordering either — it accounts for A100 over V100 (1.73x
  bandwidth, 1.93x speed) and fails on the L4 (0.33x the V100's bandwidth at
  parity).  What actually binds this kernel is open; the same question on the
  implicit action is worked in `evidence/action_cross_vendor.txt`, which shows
  the arithmetic/memory split is itself vendor-dependent (71.5% arithmetic on
  the RX 7600, 48.9% on the L4, measured at one commit).
- **GPU-vs-same-host-CPU at N=20**: 7.4x on ascicgpu073 (2.45 vs 18.1 ms,
  24 threads), 7.2x on ascicgpu24 (4.68 vs 33.6) — against 3.6x for the
  RX 7600 over the desktop host.  "Fast CPU" means fast per core: at 24
  threads the ascicgpu073 server host is faster than the desktop (18.1 vs 23.6 ms),
  so the original sweep's saturated 3.4x was a host-pairing statement,
  not a property of Carina's explicit kernels.

## Reproducing the sweeps

`run_baselines.sh` is the 530k-DOF baseline sweep (report §2),
`run_scaling2.sh` the cube64/cube80 scaling sweep (§3).  Both serialize
runs — the large cases need most of a 60 GB host to themselves.  Results
land in `results/<tag>.jsonl`; the committed files are the study's records,
so pick a fresh tag to avoid appending to them.  The tags that matter:

| Tag | What it holds |
| --- | --- |
| `current` | The numbers the report quotes today — every GPU variant re-measured after the stiffness-action rewrite |
| `baseline`, `proposed`, `scaling2`, `variance`, `detail`, `bisect`, `nbuilds-check`, `jvp` | The original campaign and the rewrite's first measurements, kept so the history is auditable |

`current` is the one to compare a new change against.

## Editing cases

Change `cases.jl`, then regenerate the checked-in decks:

```sh
julia benchmark/write_inputs.jl
```

Decks in `inputs/` are generated files — never edit them by hand. They come
from `write_inputs.jl` and are tracked, so they are a stable record of what the
implicit study ran. The explicit sweep's decks are different: their content
encodes the step count passed on the command line, so a run at a different
`nsteps` rewrites them. They go to the gitignored `inputs/generated/` for that
reason, and the sweep's reproducible record is the JSONL in `results/`, which
is tracked and appended to rather than overwritten.
