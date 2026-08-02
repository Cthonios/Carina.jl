#!/usr/bin/env bash
# cube100 (3.09M DOF) scaling runs: the GPU-vs-CPU and AMG-vs-Jacobi verdict.
set -u
cd "$(dirname "$0")/.."
STACKED="$PWD:$PWD/bin:@stdlib"
TAG="${1:-scaling}"
LIMIT=5400

run_gpu() {
    echo "=== [$(date +%H:%M:%S)] $1 $2 ==="
    JULIA_LOAD_PATH="$STACKED" timeout "$LIMIT" julia benchmark/harness.jl "$1" "$2" "$TAG"
    [ $? -ne 0 ] && echo "!!! $1 $2 failed (124 = timeout)"
}
run_cpu() {
    echo "=== [$(date +%H:%M:%S)] $1 $2 ==="
    timeout "$LIMIT" julia --project=. benchmark/harness.jl "$1" "$2" "$TAG"
    [ $? -ne 0 ] && echo "!!! $1 $2 failed (124 = timeout)"
}

run_gpu cube100-qs      gpu-cg-amg
run_gpu cube100-qs      gpu-cg-jacobi
run_gpu cube100-newmark gpu-cg-amg
run_gpu cube100-newmark gpu-cg-jacobi
run_cpu cube100-qs      cpu-cg-amg
run_cpu cube100-newmark cpu-cg-jacobi

echo "=== sweep complete: benchmark/results/$TAG.jsonl ==="
