# GPU solver benchmark suite

Everything behind `benchmark_report.md` (repo root): input decks, mesh
generation, the measurement harness, sweep scripts, and the raw results.

## Layout

| Path | What it is |
| --- | --- |
| `inputs/` | Standalone YAML decks, one per (case, variant) pair of the study — directly runnable, see below |
| `cases.jl` | Single source of truth for case/variant definitions (decks and harness both derive from it) |
| `write_inputs.jl` | Regenerates `inputs/` from `cases.jl` |
| `harness.jl` | Measurement harness: one (case, variant) per fresh process, appends a JSON-lines record to `results/<tag>.jsonl` |
| `action_bench.jl` | Times the matrix-free stiffness action in isolation, on device — the microbenchmark behind the kernel analysis in the report |
| `explicit_sweep.jl` | Explicit-dynamics CPU-vs-GPU harness: one (size, device) per fresh process, same JSON-lines contract |
| `meshgen.jl` | Structured HEX8 cube mesh generator for the large cases |
| `torsiongen.jl` | Structured HEX8 torsion-bar generator at arbitrary refinement (`N=20` reproduces `torsion.g`) |
| `run_baselines.sh`, `run_round2.sh`, `run_scaling.sh`, `run_scaling2.sh` | The sweep scripts the implicit study ran |
| `run_explicit_scaling.sh` | The explicit CPU-vs-GPU size sweep (report §8) |
| `results/*.jsonl` | Raw records of the study — every number in the report traces to these |
| `evidence/` | Log excerpts and ablation arms backing specific report claims (OOMs, L-BFGS failure, ROCm test output, the action ablation) |
| `design.md` | Proposed solution, design rationale, rejected alternatives |

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
setup.  Note `--device cuda` is untested (report §7).

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
threads, which beats 12 at N=20 (23.7 vs 27.5 ms/step).

On a machine whose home directory is NFS/GPFS (the ascicgpu hosts), set
`CARINA_BENCH_SCRATCH=/scratch/...` so meshes, decks, outputs, and result
records land on a local disk.

### Cross-card reference (2026-08-25, commit `2b827db`)

The same ladder on the Sandia V100 (32 GB) and A100 (40 GB), extended past
the RX 7600's 8 GB capacity cap; records in
`results/explicit-ascicgpu{24,073}.jsonl`.  Per-step milliseconds, with the
original sweep's CPU baseline (the 5.7 GHz desktop host, 24 threads — the
fastest CPU measured per core) alongside:

| N | DOF | desktop CPU | Rigel 48T | RX 7600 | V100 | A100 | A100 / CPU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 39k | 1.84 | 1.67 | 1.42 | 0.69 | 0.43 | 4.3x |
| 12 | 122k | 5.62 | 2.97 | 1.86 | 1.29 | 0.90 | 6.2x |
| 20 | 531k | 23.6 | 12.2 | 6.63 | 4.68 | 2.45 | 9.6x |
| 28 | 1.42M | 63.0 | 34.4 | 17.7 | 12.0 | 5.93 | 10.6x |
| 36 | 2.96M | 131.1 | 75.4 | 37.6 | 25.4 | 12.8 | 10.3x |
| 44 | 5.35M | 231.4 | 128.0 | 68.8 | 46.8 | 23.8 | 9.7x |
| 50 | 7.81M | 341.5 | 181.3 | 100.5 | 71.0 | 34.5 | **9.9x** |
| 64 | 16.2M | — | 368.0 | — (OOM) | 148.7 | 73.7 | — |
| 72 | 23.0M | — | — | — | 219.0 | — | — |
| 80 | 31.5M | — | 708.7 | — | — | 158.7 | — |
| 100 | 61.2M | — | 1806 | — | — | — | — |

Rigel is a dual EPYC 9634 (168 cores / 336 threads, 1.5 TB); records in
`results/explicit-rigel{,-threads}.jsonl`.  Its column is the machine's
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
  ~13.5–15.5 ns/elem (A100), ~27–29 (V100), ~40 (RX 7600), ~70–80
  (Rigel at 48T), ~135–145 (desktop CPU).  No scaling cliff up to 31.5M
  DOF; memory capacity, not bandwidth, is the ceiling (~1.0–1.3 KB/DOF
  on all three cards).
- **Against the fastest CPU measured, the saturated ratios are A100 ~10x,
  V100 ~5x, RX 7600 3.4x** — stable across the size range because the CPU
  is flat per element too.  The CPU ladder stops at N=50; the A100 runs
  4x that problem.
- **Placement**: every GPU beats every CPU node measured (even the
  168-core Rigel runs 1.8x slower than the RX 7600 and 5.3x slower than
  the A100 at N=50).  Rigel's niche is capacity: it runs 61.2M DOF at
  1.8 s/step — 2x past the A100's 40 GB ceiling with room for far more —
  so CPU nodes are for problems that do not fit a GPU, not for speed.
- **The explicit ordering is A100 > V100 > RX 7600** — unlike the implicit
  gather kernel, where the RX 7600's Infinity Cache put it ahead of the
  V100.  The explicit internal-force kernel follows FP64 throughput
  (the 7600 runs FP64 at 1/32 rate), not cache capacity.
- **GPU-vs-same-host-CPU at N=20**: 7.4x on ascicgpu073 (2.45 vs 18.1 ms,
  24 threads), 7.2x on ascicgpu24 (4.68 vs 33.6) — against 3.6x for the
  RX 7600 over the desktop host.  "Fast CPU" means fast per core: at 24
  threads the ascicgpu073 server host beats the desktop (18.1 vs 23.6 ms),
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

Decks in `inputs/` are generated files — never edit them by hand.
