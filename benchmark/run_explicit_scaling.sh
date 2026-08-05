#!/bin/bash
# Explicit-dynamics CPU-vs-GPU scaling sweep on the torsion bar (report §8).
#
# One fresh process per (size, device); sizes are interleaved so a failure at
# large N still leaves paired data at every smaller size.  Step counts are
# chosen to keep the measured interval at roughly 10 s of CPU time, so the
# per-step numbers are not dominated by the single Exodus write inside it.
#
# CPU baseline uses 24 threads: measured at N=20, 24 threads beats 12
# (23.7 vs 27.5 ms/step), so this is the CPU's best showing.
#
# Usage: benchmark/run_explicit_scaling.sh [tag]
set -u
cd "$(dirname "$0")/.."
TAG="${1:-explicit-scaling}"

# N:nsteps — nsteps must be even (two equal control intervals)
SIZES=(8:8000 12:3000 20:800 28:300 36:160 44:100 50:80)

for entry in "${SIZES[@]}"; do
    N="${entry%%:*}"
    STEPS="${entry##*:}"
    for DEV in cpu rocm; do
        echo "=== N=$N device=$DEV steps=$STEPS ==="
        julia --project=. benchmark/explicit_sweep.jl "$N" "$DEV" "$TAG" "$STEPS" 24 \
            2>&1 | grep -E "STOP|Setup complete|Recorded|measured|per_step_ms|ok =|ERROR"
        echo
    done
done
echo "=== sweep complete: benchmark/results/$TAG.jsonl ==="
