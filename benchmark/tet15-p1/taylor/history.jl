# Radius of the impact face and length of the Taylor bar at every output
# frame of an Exodus file (Carina or Sierra/SM output, one file; join the
# per-rank files of a parallel run with epu first).
#
#   julia --project=. benchmark/tet15-p1/taylor/history.jl <output.e> <history.tsv>
#
# Radius: maximum over the nodes of node set "impact" of the distance from the
# axis x = y = 0.  Length: maximum over the nodes of the free face (largest
# reference z) of the current z.  Units of the file (m); written as time,
# radius, length.  The displacement is displ_x/y/z (Carina) or
# displacement_x/y/z (Sierra/SM).

using Exodus
using Printf

function taylor_history(out_file)
    e = ExodusDatabase(out_file, "r")
    X = read_coordinates(e)
    names = read_names(e, NodeSet)
    k = findfirst(==("impact"), names)
    k === nothing && error("$out_file: node set \"impact\" missing")
    impact = Int.(read_sets(e, NodeSet)[k].nodes)
    zmax = maximum(X[3, :])
    top = findall(n -> X[3, n] > zmax - 1e-9, axes(X, 2))
    vn = read_names(e, NodalVariable)
    base = "displ_x" in vn ? "displ" : "displacement_x" in vn ? "displacement" :
           error("$out_file: no displacement variable among $vn")
    times = read_times(e)
    rows = map(eachindex(times)) do i
        ux, uy, uz = (read_values(e, NodalVariable, i, "$(base)_$c") for c in ("x", "y", "z"))
        r = maximum(hypot(X[1, n] + ux[n], X[2, n] + uy[n]) for n in impact)
        len = maximum(X[3, n] + uz[n] for n in top)
        (times[i], r, len)
    end
    close(e)
    return rows
end

function write_history(path, rows)
    open(path, "w") do io
        println(io, "time\tradius\tlength")
        foreach(r -> @printf(io, "%.6e\t%.6e\t%.6e\n", r...), rows)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    rows = taylor_history(ARGS[1])
    write_history(ARGS[2], rows)
    t, r, len = rows[end]
    @printf("%s: t = %.3e s, radius %.4f mm, length %.4f mm\n", ARGS[1], t, 1e3 * r, 1e3 * len)
end
