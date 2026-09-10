# Archived records: the optimization-round evidence trail

These files are **not** a performance record. Each one is the before or after
of a specific decision taken during the GPU solver campaign, and
`../../../benchmark_report.md` cites several of them as the justification for a
claim. They are kept so those claims stay auditable.

For the current device-by-regime measurements, see `../matrix/` and the
"Performance matrix" section of `../../README.md`.

## Why they are not a performance record

Three things make them unsuitable for comparing devices:

* **Three schemas.** `harness.jl` (implicit and quasi-static),
  `explicit_sweep.jl` (the explicit ladder) and
  `crosscode/run_crosscode.py` each emit a different record shape, and even the
  degree-of-freedom count is spelled `n_dofs` in one and `n_dof` in another.
* **Missing provenance.** 80 of the 127 records name neither the host nor the
  GPU vendor. Every `gpu-*` record written before 2026-09-11 is a Radeon
  RX 7600 on `sirius`, because `harness.jl` was ROCm-only until then, but
  nothing in the data says so.
* **Mixed commits.** The rows span roughly eight commits of a campaign whose
  whole purpose was to change performance, so two rows are generally not
  comparable to each other.

## What each tag was

| file | what it records |
| --- | --- |
| `baseline.jsonl` | The state before the campaign: CPU variants and the first GPU CG runs. |
| `current.jsonl` | The implicit numbers quoted in `benchmark_report.md` §2. |
| `proposed.jsonl`, `scaling2.jsonl` | Intermediate rounds of the AMG work. |
| `jvp.jsonl` | The closed-form directional derivative replacing the ForwardDiff dual pass. |
| `fp32-csr.jsonl` | Reduced precision on the CSR coarse levels — rejected; the traffic is fine-sized. |
| `fp32-smoother.jsonl`, `fp64-baseline.jsonl` | The back-to-back pair behind §2's top two rows. |
| `nbuilds-check.jsonl` | Confirming the AMG hierarchy is rebuilt as rarely as intended. |
| `bisect.jsonl`, `detail.jsonl`, `variance.jsonl` | Diagnostics: a regression hunt, a per-phase breakdown, and a run-to-run spread check. |
| `threadcheck.jsonl` | CPU thread scaling at 530k DOF. |
| `explicit-scaling.jsonl` | The original explicit ladder, RX 7600 against its desktop host. |
| `explicit-rigel.jsonl`, `explicit-rigel-threads.jsonl` | Rigel's CPU ladder and its thread sweep, which found the 48-thread optimum. |
| `explicit-ascicgpu073.jsonl`, `explicit-ascicgpu24.jsonl` | The A100 and V100 explicit ladders. |
| `explicit-rigel-l4.jsonl` | The L4 explicit ladder, 2026-09-08. |
| `fec-block-size-*.jsonl` | The FiniteElementContainers GPU block-size sweep; the 256 default stood. |
