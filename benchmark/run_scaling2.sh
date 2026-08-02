#!/usr/bin/env bash
# Serial scaling sweep: cube64 (823k) + cube80 (1.57M DOF), one process at a
# time — the 3.09M cube100 exhausts host RAM (assembler pattern + SA build)
# on this 60 GB machine and is deliberately excluded.
set -u
cd "$(dirname "$0")/.."
STACKED="$PWD:$PWD/bin:@stdlib"
TAG="${1:-scaling}"
LIMIT=2700

run() {
    local case="$1" variant="$2"
    echo "=== [$(date +%H:%M:%S)] $case $variant (mem: $(free -g | awk 'NR==2{print $7}')G avail) ==="
    if [[ "$variant" == gpu-* ]]; then
        JULIA_LOAD_PATH="$STACKED" timeout "$LIMIT" julia benchmark/harness.jl "$case" "$variant" "$TAG"
    else
        timeout "$LIMIT" julia --project=. benchmark/harness.jl "$case" "$variant" "$TAG"
    fi
    [ $? -ne 0 ] && echo "!!! $case $variant failed (124 = timeout)"
}

# Quasistatic scaling (the AMG story), then Newmark reference points,
# then instrumented torsion detail re-runs for the phase breakdown.
run cube64-qs gpu-cg-amg
run cube64-qs cpu-cg-amg
run cube64-qs cpu-cg-jacobi
run cube80-qs gpu-cg-amg
run cube80-qs gpu-cg-jacobi
run cube80-qs cpu-cg-amg
run cube80-qs cpu-cg-jacobi
run cube80-newmark gpu-cg-jacobi
run cube80-newmark cpu-cg-jacobi
run torsion-qs gpu-cg-amg
run torsion-qs gpu-cg-jacobi

echo "=== sweep complete: benchmark/results/$TAG.jsonl ==="
