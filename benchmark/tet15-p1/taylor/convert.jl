# Taylor bar meshes: from the smoothed four-node mesh of Norma to TETRA10,
# in meters.
#
#   julia --project=. benchmark/tet15-p1/taylor/convert.jl <tet4.g> <smoothed.e | -> <tet10.g>
#
# <tet4.g> is the Cubit mesh (mm) with its blocks and sets; <smoothed.e> is the
# output of Norma's mesh smoothing on it, whose last frame holds the smoothing
# displacement (disp_x, disp_y, disp_z), or "-" for no smoothing.  The TETRA10
# mesh has the smoothed vertex positions and one node at the midpoint of every
# edge; a midpoint on the lateral surface (both ends on the side set
# "lateral") is moved radially onto the cylinder of radius R = 3.2 mm.
# Coordinates are scaled from mm to m.  Node sets gain the midpoints whose
# two ends are in the set; side sets are kept.  The script prints the shape
# quality q = 12 (3V)^(2/3) / sum(l_i^2) of the four-node elements before and
# after smoothing (1 for the regular tetrahedron, 0 for a flat one): minimum,
# mean and the fraction below 0.3.

using Exodus
using LinearAlgebra
using Printf

const R_MM  = 3.2
const SCALE = 1.0e-3
const EDGES = ((1, 2), (2, 3), (3, 1), (1, 4), (2, 4), (3, 4))
const FACES = ((1, 2, 4), (2, 3, 4), (1, 4, 3), (1, 3, 2))   # Exodus side order

function quality(X, conn)
    q = Vector{Float64}(undef, size(conn, 2))
    for e in axes(conn, 2)
        x = [X[:, conn[i, e]] for i in 1:4]
        V = abs(dot(x[2] - x[1], cross(x[3] - x[1], x[4] - x[1]))) / 6
        s = sum(norm(x[b] - x[a])^2 for (a, b) in EDGES)
        q[e] = 12 * (3V)^(2 / 3) / s
    end
    return q
end

report(tag, q) = @printf("%-9s q: min %.3f  mean %.3f  below 0.3: %.4f%%\n", tag,
                         minimum(q), sum(q) / length(q), 100 * count(<(0.3), q) / length(q))

function main(tet4, smoothed, out)
    exo = ExodusDatabase(tet4, "r")
    X = Matrix{Float64}(read_coordinates(exo))
    blocks = read_sets(exo, Block)
    bnames = read_names(exo, Block)
    nsets = read_sets(exo, NodeSet); nsnames = read_names(exo, NodeSet)
    ssets = read_sets(exo, SideSet); ssnames = read_names(exo, SideSet)
    close(exo)
    length(blocks) == 1 || error("one block expected")
    conn = Matrix{Int}(blocks[1].conn)
    report("cubit", quality(X, conn))
    if smoothed != "-"
        s = ExodusDatabase(smoothed, "r")
        n = read_number_of_time_steps(s)
        U = vcat((read_values(s, NodalVariable, n, "disp_$c")' for c in ("x", "y", "z"))...)
        close(s)
        size(U) == size(X) || error("smoothed mesh does not match the Cubit mesh")
        X .+= U
        report("smoothed", quality(X, conn))
    end

    # Lateral nodes, from the lateral side set.
    lateral = Set{Int}()
    il = findfirst(==("lateral"), ssnames)
    il === nothing && error("side set \"lateral\" missing")
    for (e, f) in zip(ssets[il].elements, ssets[il].sides)
        foreach(v -> push!(lateral, conn[v, e]), FACES[f])
    end
    r_dev = maximum(abs(hypot(X[1, n], X[2, n]) - R_MM) for n in lateral)
    @printf("lateral vertices: largest distance from the cylinder %.2e mm\n", r_dev)

    nv = size(X, 2)
    xs = [X[:, n] for n in 1:nv]
    mid = Dict{Tuple{Int, Int}, Int}()
    c10 = zeros(Int, 10, size(conn, 2))
    for e in axes(conn, 2)
        c10[1:4, e] .= conn[:, e]
        for (k, (a, b)) in enumerate(EDGES)
            p, q = conn[a, e], conn[b, e]
            key = p < q ? (p, q) : (q, p)
            c10[4 + k, e] = get!(mid, key) do
                x = (xs[p] .+ xs[q]) ./ 2
                if p in lateral && q in lateral
                    r = hypot(x[1], x[2])
                    x = [x[1] * R_MM / r, x[2] * R_MM / r, x[3]]
                end
                push!(xs, x)
                length(xs)
            end
        end
    end
    nn = length(xs)
    Xo = Matrix{Float64}(undef, 3, nn)
    for n in 1:nn
        Xo[:, n] .= SCALE .* xs[n]
    end
    new_nsets = map(nsets) do ns
        m = Set{Int}(Int.(ns.nodes))
        vcat(Int.(ns.nodes), sort!([n for (k, n) in mid if k[1] in m && k[2] in m]))
    end

    isfile(out) && rm(out)
    init = Initialization{Int32}(3, nn, size(c10, 2), 1, length(nsets), length(ssets))
    o = ExodusDatabase{Int32, Int32, Int32, Float64}(out, "w", init)
    try
        write_coordinates(o, Xo)
        write_names(o, Block, String.(bnames))
        write_block(o, Int(blocks[1].id), "TETRA10", Int32.(c10))
        for (i, ns) in enumerate(nsets)
            write_set(o, NodeSet(Int32(ns.id), Int32.(new_nsets[i])))
        end
        write_names(o, NodeSet, String.(nsnames))
        for ss in ssets
            write_set(o, SideSet(Int32(ss.id), Int32.(ss.elements), Int32.(ss.sides), Int32[], Int32[]))
        end
        write_names(o, SideSet, String.(ssnames))
    finally
        close(o)
    end
    @printf("%s: %d TETRA10, %d nodes\n", out, size(c10, 2), nn)
end

main(ARGS...)
