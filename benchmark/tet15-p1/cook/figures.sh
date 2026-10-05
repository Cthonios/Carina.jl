#!/bin/bash
# Pressure figures of the Cook's membrane benchmark.
#
#   benchmark/tet15-p1/cook/figures.sh <out-dir> [<extra-data-dir>]
#
# Stages, in <out-dir>/data, the front faces of the three meshes
# (surface_faces.jl) and, for every run under runs/ that has them, the files
# pressure_nodes.csv and displacement.csv of extract_pressure.jl; files of
# runs made elsewhere are taken from <extra-data-dir>, named
# <case>-<element>-h<h>-{pressure_nodes,displacement}.csv.  It then writes,
# per case, the six elements at h = 2 (<case>-h2.png) and each element over
# h = 8, 4, 2 (<case>-<element>-refine.png).  One color scale per case, from
# the elements whose pressure does not oscillate (TET15-P1, TET15-P0 and the
# composite tetrahedron).  Run from the Carina root.

set -euo pipefail
out=$1
extra=${2:-}
here=benchmark/tet15-p1/cook
data=$out/data
mkdir -p "$data"

for h in 8 4 2; do
    [[ -f $data/faces-h$h.csv ]] ||
        julia --project=. $here/surface_faces.jl $here/meshes/cook-h$h.g $data/faces-h$h.csv
done
for d in $here/runs/*; do
    r=$(basename "$d")
    for f in pressure_nodes displacement; do
        [[ -f $d/$f.csv ]] && cp "$d/$f.csv" "$data/$r-$f.csv"
    done
done
if [[ -n $extra ]]; then
    cp "$extra"/*-pressure_nodes.csv "$extra"/*-displacement.csv "$data"/ 2>/dev/null || true
fi

render() { python3 $here/render_pressure.py "$data" "$@"; }
for c in elastic plastic; do
    lim=auto:$c:tet15-p1,tet15-p0,lcm-ct
    render "$out/$c-h2.png" $lim 3 \
        "TET10\n(Carina)=$c-tet10-h2" "TET10\n(Albany)=$c-lcm-tet10-h2" \
        "TET15=$c-tet15-h2" "TET15-P1=$c-tet15-p1-h2" "TET15-P0=$c-tet15-p0-h2" \
        "Composite tetrahedron\n(Albany)=$c-lcm-ct-h2"
    # Node counts: 647, 3208, 22 059 on the TETRA10 meshes (TET10, composite
    # tetrahedron); 1721, 9127, 66 813 on the TETRA15 meshes.
    for e in tet10 tet15 tet15-p1 tet15-p0 lcm-ct; do
        case $e in
            tet10|lcm-ct) n8=647; n4=3208; n2="22 059" ;;
            *)            n8=1721; n4=9127; n2="66 813" ;;
        esac
        render "$out/$c-$e-refine.png" $lim 3 "h = 8\n327 elements\n$n8 nodes=$c-$e-h8" \
            "h = 4\n1860 elements\n$n4 nodes=$c-$e-h4" "h = 2\n14 475 elements\n$n2 nodes=$c-$e-h2"
    done
done
