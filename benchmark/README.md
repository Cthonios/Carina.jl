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
| `explicit_sweep.jl` | Explicit-dynamics CPU-vs-GPU harness: one (size, device) per fresh process, same JSON-lines contract |
| `meshgen.jl` | Structured HEX8 cube mesh generator for the large cases |
| `torsiongen.jl` | Structured HEX8 torsion-bar generator at arbitrary refinement (`N=20` reproduces `torsion.g`) |
| `run_baselines.sh`, `run_round2.sh`, `run_scaling.sh`, `run_scaling2.sh` | The sweep scripts the implicit study ran |
| `run_explicit_scaling.sh` | The explicit CPU-vs-GPU size sweep (report §8) |
| `results/*.jsonl` | Raw records of the study — every number in the report traces to these |
| `evidence/` | Log excerpts backing specific report claims (OOMs, L-BFGS failure, ROCm test output) |
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
setup.  Note `--device cuda` is untested (report §6).

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

## Reproducing the sweeps

`run_baselines.sh` is the 530k-DOF baseline sweep (report §2),
`run_scaling2.sh` the cube64/cube80 scaling sweep (§3).  Both serialize
runs — the large cases need most of a 60 GB host to themselves.  Results
land in `results/<tag>.jsonl`; the committed files are the study's records
(tags: `baseline`, `proposed`, `scaling2`, `variance`, `detail`, `bisect`,
`nbuilds-check`), so pick a fresh tag to avoid appending to them.

## Editing cases

Change `cases.jl`, then regenerate the checked-in decks:

```sh
julia benchmark/write_inputs.jl
```

Decks in `inputs/` are generated files — never edit them by hand.
