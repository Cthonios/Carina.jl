# Punch meshes: from the Cubit four-node mesh to TETRA10, in meters.
#
#   julia --project=. benchmark/tet15-p1/punch/convert.jl <tet4.g> <tet10.g>
#
# <tet4.g> is the Cubit mesh (mm) with its blocks and sets, smoothed by
# Cubit's condition-number smoother only.  The TETRA10 mesh has one node at
# the midpoint of every edge.  A midpoint on the lateral surface (both ends on
# the side set "lateral") is moved radially onto the cylinder of radius
# R = 4 mm; a midpoint of an edge of the footprint circle (an edge shared by
# a face of the side set "punch" and a face of the side set "top") is moved
# radially onto the circle of radius a = 1 mm in the plane z = H = 4 mm.
# Coordinates are scaled from mm to m.  Node sets gain the midpoints whose two
# ends are in the set; side sets are kept.  Two one-node sets are added for the
# point constraints that remove the rigid-body modes the base plane leaves:
# "axis-base", the base node on the axis (0, 0, 0), and "pin-base", the base
# node on the half-line y = 0 at r = R, (R, 0, 0).  The script prints the shape
# quality q = 12 (3V)^(2/3) / sum(l_i^2) of the four-node elements (1 for the
# regular tetrahedron, 0 for a flat one) and checks that the footprint circle
# nodes lie on the circle and the footprint nodes on the top plane.

using Exodus
using LinearAlgebra
using Printf

const R_MM  = 4.0
const A_MM  = 1.0
const H_MM  = 4.0
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

# The vertex edges of the faces of a side set, as sorted node pairs.
function face_edges(conn, ss)
    E = Set{Tuple{Int, Int}}()
    for (e, f) in zip(ss.elements, ss.sides)
        v = [conn[i, e] for i in FACES[f]]
        for (a, b) in ((1, 2), (2, 3), (3, 1))
            p, q = v[a], v[b]
            push!(E, p < q ? (p, q) : (q, p))
        end
    end
    return E
end

function side_set(ssets, ssnames, name)
    i = findfirst(==(name), ssnames)
    i === nothing && error("side set \"$name\" missing")
    return ssets[i]
end

function main(tet4, out)
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

    lateral = Set{Int}()
    for (e, f) in zip(side_set(ssets, ssnames, "lateral").elements, side_set(ssets, ssnames, "lateral").sides)
        foreach(v -> push!(lateral, conn[v, e]), FACES[f])
    end
    circle = intersect(face_edges(conn, side_set(ssets, ssnames, "punch")),
                       face_edges(conn, side_set(ssets, ssnames, "top")))
    isempty(circle) && error("no edges shared by the side sets \"punch\" and \"top\"")
    cnodes = Set{Int}(n for k in circle for n in k)
    @printf("lateral vertices: largest distance from the cylinder %.2e mm\n",
            maximum(abs(hypot(X[1, n], X[2, n]) - R_MM) for n in lateral))
    @printf("footprint circle: %d edges, %d vertices, largest |r - a| %.2e mm, |z - H| %.2e mm\n",
            length(circle), length(cnodes), maximum(abs(hypot(X[1, n], X[2, n]) - A_MM) for n in cnodes),
            maximum(abs(X[3, n] - H_MM) for n in cnodes))

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
                if key in circle
                    r = hypot(x[1], x[2])
                    x = [x[1] * A_MM / r, x[2] * A_MM / r, x[3]]
                elseif p in lateral && q in lateral
                    r = hypot(x[1], x[2])
                    x = [x[1] * R_MM / r, x[2] * R_MM / r, x[3]]
                end
                push!(xs, x)
                length(xs)
            end
        end
    end
    @printf("footprint circle midpoints: largest |r - a| %.2e mm, |z - H| %.2e mm\n",
            maximum(abs(hypot(xs[mid[k]][1], xs[mid[k]][2]) - A_MM) for k in circle),
            maximum(abs(xs[mid[k]][3] - H_MM) for k in circle))
    nn = length(xs)
    Xo = Matrix{Float64}(undef, 3, nn)
    for n in 1:nn
        Xo[:, n] .= SCALE .* xs[n]
    end
    new_nsets = map(nsets) do ns
        m = Set{Int}(Int.(ns.nodes))
        vcat(Int.(ns.nodes), sort!([n for (k, n) in mid if k[1] in m && k[2] in m]))
    end
    point(x) = (d = [norm(X[:, n] .- x) for n in 1:nv]; i = argmin(d);
                d[i] < 1.0e-9 || error("no vertex at \$x"); i)
    points = ["axis-base" => point([0.0, 0.0, 0.0]), "pin-base" => point([R_MM, 0.0, 0.0])]
    @printf("point sets: axis-base node %d, pin-base node %d\n", points[1][2], points[2][2])
    next_id = maximum(Int(ns.id) for ns in nsets)
    nsets = vcat(nsets, [NodeSet(Int32(next_id + i), Int32[n]) for (i, (_, n)) in enumerate(points)])
    nsnames = vcat(String.(nsnames), first.(points))
    new_nsets = vcat(new_nsets, [[n] for (_, n) in points])
    ip = findfirst(==("punch"), nsnames)
    ip === nothing || @printf("node set punch: %d nodes, largest |z - H| %.2e mm, largest r %.6f mm\n",
                              length(new_nsets[ip]), maximum(abs(xs[n][3] - H_MM) for n in new_nsets[ip]),
                              maximum(hypot(xs[n][1], xs[n][2]) for n in new_nsets[ip]))

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
