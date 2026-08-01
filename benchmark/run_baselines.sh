#!/usr/bin/env bash
# Sequential baseline sweep for the GPU solver campaign.
# GPU variants need AMDGPU, which lives in bin/'s environment — stack it.
set -u
cd "$(dirname "$0")/.."

STACKED="$PWD:$PWD/bin:@stdlib"
TAG="${1:-baseline}"
LIMIT=5400   # seconds per run; a hung run must not stall the sweep

run() {
    local case="$1" variant="$2" env="$3"
    echo "=== [$(date +%H:%M:%S)] $case $variant ==="
    if [ "$env" = gpu ]; then
        JULIA_LOAD_PATH="$STACKED" timeout "$LIMIT" \
            julia benchmark/harness.jl "$case" "$variant" "$TAG"
    else
        timeout "$LIMIT" \
            julia --project=. benchmark/harness.jl "$case" "$variant" "$TAG"
    fi
    local rc=$?
    [ $rc -ne 0 ] && echo "!!! $case $variant exited with $rc (124 = timeout)"
}

# GPU first (matrix-free path), then CPU (assembled path).
for case in torsion-newmark torsion-qs; do
    run "$case" gpu-cg-jacobi    gpu
    run "$case" gpu-cg-chebyshev gpu
    run "$case" gpu-lbfgs        gpu
done

for case in torsion-newmark torsion-qs; do
    run "$case" cpu-cg-jacobi cpu
    run "$case" cpu-cg-ic     cpu
    run "$case" cpu-cg-amg    cpu
    run "$case" cpu-direct    cpu
done

echo "=== sweep complete: benchmark/results/$TAG.jsonl ==="
