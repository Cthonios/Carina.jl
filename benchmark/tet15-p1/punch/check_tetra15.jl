# Check of a punch TETRA15 mesh (m): the vertex, edge and face nodes of the
# footprint faces (side set "punch") lie on the top plane z = H, the nodes of
# node set "punch" lie in r <= a, and the edge nodes of the footprint circle
# lie on the circle r = a, and the nodes of node sets "symm-x" and "symm-y"
# lie on x = 0 and y = 0.  Prints the largest deviations and the counts.
#   julia --project=. benchmark/tet15-p1/punch/check_tetra15.jl <tet15.g>
using Exodus, Printf
const A = 1.0e-3; const H = 4.0e-3
const FACES = ((1, 2, 4), (2, 3, 4), (1, 4, 3), (1, 3, 2))
const FEDGE = ((5, 9, 8), (6, 10, 9), (8, 10, 7), (7, 6, 5))   # TETRA10 edge nodes of each side
function main(f)
    exo = ExodusDatabase(f, "r")
    X = Matrix{Float64}(read_coordinates(exo))
    conn = Matrix{Int}(read_sets(exo, Block)[1].conn)
    ss = read_sets(exo, SideSet); ssn = read_names(exo, SideSet)
    ns = read_sets(exo, NodeSet); nsn = read_names(exo, NodeSet)
    close(exo)
    sp = ss[findfirst(n -> startswith(n, "punch"), ssn)]
    dz = 0.0; circ = 0.0; ncirc = 0; nface = 0
    for (e, s) in zip(sp.elements, sp.sides)
        v = conn[collect(FACES[s]), e]; m = conn[collect(FEDGE[s]), e]
        c = sum(X[:, v], dims = 2) ./ 3
        # the face node of this side: the one of nodes 11-15 nearest the vertex centroid
        fn = conn[11:15, e][argmin([sum(abs2, X[:, n] .- c) for n in conn[11:15, e]])]
        nface += 1
        dz = max(dz, maximum(abs.(X[3, vcat(v, m, fn)] .- H)))
        for (k, (a, b)) in enumerate(((1, 2), (2, 3), (3, 1)))
            ra, rb = hypot(X[1, v[a]], X[2, v[a]]), hypot(X[1, v[b]], X[2, v[b]])
            if abs(ra - A) < 1e-12 && abs(rb - A) < 1e-12
                ncirc += 1; circ = max(circ, abs(hypot(X[1, m[k]], X[2, m[k]]) - A))
            end
        end
    end
    pn = ns[findfirst(==("punch"), nsn)].nodes
    @printf("%s: %d TETRA15, %d nodes; %d footprint faces, largest |z - H| of their 7 nodes %.2e m\n",
            f, size(conn, 2), size(X, 2), nface, dz)
    @printf("  footprint circle edges seen from the faces %d, largest |r - a| of their edge nodes %.2e m\n", ncirc, circ)
    @printf("  node set punch %d nodes, largest r - a %.2e m, largest |z - H| %.2e m\n", length(pn),
            maximum(hypot(X[1, n], X[2, n]) for n in pn) - A, maximum(abs(X[3, n] - H) for n in pn))
    for (name, c) in (("symm-x", 1), ("symm-y", 2))
        i = findfirst(==(name), nsn)
        i === nothing && continue
        nodes = ns[i].nodes
        @printf("  node set %s %d nodes, largest |%s| %.2e m\n", name, length(nodes), c == 1 ? "x" : "y",
                maximum(abs(X[c, n]) for n in nodes))
    end
end
main(ARGS[1])
