#!/usr/bin/env bash
# Critic round-2 evidence runs: variance repeats of the headline QS cells
# plus an instrumented CPU-Jacobi leg.  Strictly serial.
set -u
cd "$(dirname "$0")/.."
STACKED="$PWD:$PWD/bin:@stdlib"
LIMIT=2700
g() { JULIA_LOAD_PATH="$STACKED" timeout $LIMIT julia benchmark/harness.jl "$@"; }
c() { timeout $LIMIT julia --project=. benchmark/harness.jl "$@"; }
echo "=== variance repeats ==="
g torsion-qs gpu-cg-amg    variance
g torsion-qs gpu-cg-jacobi variance
c torsion-qs cpu-cg-jacobi detail
echo "=== round2 runs complete ==="
