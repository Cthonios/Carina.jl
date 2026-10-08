#!/bin/bash
# Composite tetrahedron (Sierra/SM) on the Taylor bar, one run.
#
#   sierra/run-ct.sh <h> <ranks> [alpha0|off]
#
# From benchmark/tet15-p1/taylor.  Writes sierra/ct-h<h>[-<variant>]/ with the
# deck, the log, the joined results and history.tsv.  Variants: "alpha0" sets
# the VEM stabilization parameter to 0 (the bulk parameter keeps its default
# 1.0e-6, as in the deck of the paper, whose alpha only set the first); "off"
# sets both to 0 (no stabilization).
set -euo pipefail
h=$1; np=$2; variant=${3:-}
SIERRA=$HOME/sierra/code/bin
MPI=/usr/lib64/openmpi
export PATH=$MPI/bin:$PATH LD_LIBRARY_PATH=$MPI/lib:${LD_LIBRARY_PATH:-} OMP_NUM_THREADS=1
here=$(pwd)
name=ct-h$h${variant:+-$variant}
dir=$here/sierra/$name
mkdir -p "$dir"
vem=""
[ "$variant" = alpha0 ] && vem="    vem stabilization parameter = 0.0"
[ "$variant" = off ] && vem="    vem stabilization parameter = 0.0\n    vem bulk stabilization parameter = 0.0"
# IOSS rejects the node set and side set both named "impact": Sierra reads a
# copy of the mesh with the side sets renamed (rename-sidesets.jl).
mesh=$here/sierra/taylor-h$h-tet10-sierra.g
[ -f "$mesh" ] || (cd "$here/../../.." && julia --project=. benchmark/tet15-p1/taylor/sierra/rename-sidesets.jl \
    "$here/meshes/taylor-h$h-tet10.g" "$mesh")
sed -e "s|{MESH}|$mesh|" -e "s|{OUT}|taylor.e|" \
    -e "s|{HB}|taylor.hb|" -e "s|^{VEM}$|$vem|" sierra/ct-template.i > "$dir/$name.i"
cd "$dir"
start=$(date +%s)
mpirun -np "$np" --bind-to none "$SIERRA/adagio" -i "$name.i" > "$name.log" 2>&1 || echo "adagio exit $?" >> "$name.log"
end=$(date +%s)
if [ "$np" -gt 1 ]; then
  "$SIERRA/epu" -auto "taylor.e.$np.0" >> "$name.log" 2>&1 && rm -f taylor.e.$np.*
fi
echo "wall $((end - start)) s" >> "$name.log"
cd "$here/../../.."
julia --project=. benchmark/tet15-p1/taylor/history.jl "$dir/taylor.e" "$dir/sierra-$name-history.tsv"
