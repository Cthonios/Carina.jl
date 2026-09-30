# Pressure on the surface of the Taylor bar near the impact face, for the
# figures of render_pressure.py, from a Carina run with --stress or a
# Sierra/SM run with the element variable stress:
#
#   julia --project=. benchmark/tet15-p1/taylor/pressure.jl runs/<element>-h<h>-stress ...
#
# For each run directory, reads the last frame of taylor.e and writes
#   <dir>/surface.vtp  the boundary faces of the deformed bar whose nodes all
#                      lie within ZMAX of the wall, each face subdivided into
#                      NSUB^2 triangles, with the pressure p = -tr(σ)/3
#                      (positive in compression, Pa) at their vertices
#   <dir>/edges.vtp    the edges of those faces, as polylines of NSUB segments
# The pressure in each element is the L2 projection of its quadrature-point
# values onto the quadratic polynomials (as in ../cook/extract_pressure.jl),
# restricted to the face; the face geometry is the quadratic face of the
# ten-node tetrahedron (vertices and edge midpoints) at the deformed
# positions X + u.

using Exodus
using LinearAlgebra
using Printf
import ReferenceFiniteElements as RFE

const ZMAX = 8.0e-3   # m from the wall
const NSUB = 4

const RULES = Dict(n => RFE.cell_quadrature_points_and_weights(RFE.Tet{RFE.Lagrange, 1}(),
                                                               RFE.GaussLegendre(d))
                   for (n, d) in ((1, 1), (4, 2), (14, 5)))
const P2_NODES = ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                  [0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.0],
                  [0.0, 0.0, 0.5], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5])
monomials(ξ, degree) = degree == 1 ? [1.0, ξ[1], ξ[2], ξ[3]] :
    [1.0, ξ[1], ξ[2], ξ[3], ξ[1]^2, ξ[2]^2, ξ[3]^2, ξ[1] * ξ[2], ξ[1] * ξ[3], ξ[2] * ξ[3]]

# Values at the ten nodes of the quadratic tetrahedron of the polynomial
# fitted to the quadrature values P[q][e].
function node_values(P)
    nq = length(P)
    nq == 1 && return [fill(v, 10) for v in P[1]]
    ξ, w = RULES[nq]
    degree = nq == 14 ? 2 : 1
    A = reduce(vcat, (monomials(ξ[:, q], degree)' for q in 1:nq))
    E = reduce(vcat, (monomials(x, degree)' for x in P2_NODES))
    L = E * ((A' * Diagonal(w) * A) \ (A' * Diagonal(w)))
    return [L * [P[q][e] for q in 1:nq] for e in eachindex(P[1])]
end

# Faces of the ten-node tetrahedron (Exodus side order): three vertices and
# the three edge midpoints opposite to vertex 3, 1, 2 of the face, i.e. on
# edges (a,b), (b,c), (c,a).
const FACES = (((1, 2, 4), (5, 9, 8)), ((2, 3, 4), (6, 10, 9)),
               ((1, 4, 3), (8, 10, 7)), ((1, 3, 2), (7, 6, 5)))

# Quadratic Lagrange functions on a triangle at barycentric (l1, l2, l3), in
# the order vertices 1-3, midpoints of edges 12, 23, 31.
quad6(l1, l2, l3) = (l1 * (2l1 - 1), l2 * (2l2 - 1), l3 * (2l3 - 1),
                     4l1 * l2, 4l2 * l3, 4l3 * l1)

function extract(dir)
    file = first(filter(f -> isfile(joinpath(dir, f)), ["taylor.e", "taylor_output.e", "taylor.exo"]))
    exo = ExodusDatabase(joinpath(dir, file), "r")
    step = read_number_of_time_steps(exo)
    t = read_time(exo, step)
    X = read_coordinates(exo)
    vn = read_names(exo, NodalVariable)
    base = any(==("displ_x"), vn) ? "displ" : any(==("displacement_x"), vn) ? "displacement" : "disp"
    x = X .+ reduce(vcat, (read_values(exo, NodalVariable, step, "$(base)_$c")'
                           for c in ("x", "y", "z")))
    enames = read_names(exo, ElementVariable)
    # Carina: sigma_xx_<point> at every quadrature point.  Sierra/SM: stress_xx,
    # one value per element (the element mean; for the composite tetrahedron
    # with its volume-averaged J the pressure is constant on the element).
    carina = any(startswith("sigma_xx_"), enames)
    nq = carina ? count(startswith("sigma_xx_"), enames) : Int(any(==("stress_xx"), enames))
    nq > 0 || error("no stress output in $dir; rerun with --stress (Carina) or " *
                    "element variables = stress (Sierra/SM)")
    comp(c, q) = carina ? "sigma_$(c)$(c)_$q" : "stress_$(c)$(c)"
    conn = Matrix{Int}(undef, 10, 0)
    vals = Vector{Vector{Float64}}()
    for b in read_sets(exo, Block)
        name = read_names(exo, Block)[findfirst(==(b.id), read_ids(exo, Block))]
        P = [-sum(read_values(exo, ElementVariable, step, name, comp(c, q))
                  for c in ("x", "y", "z")) ./ 3 for q in 1:nq]
        conn = hcat(conn, Int.(b.conn[1:10, :]))
        append!(vals, node_values(P))
    end
    close(exo)

    # Boundary faces: faces whose sorted vertex triple occurs once.
    count_of = Dict{NTuple{3, Int}, Int}()
    for e in axes(conn, 2), (v, _) in FACES
        k = Tuple(sort([conn[i, e] for i in v]))
        count_of[k] = get(count_of, k, 0) + 1
    end
    pts = Vector{NTuple{4, Float64}}()      # x, y, z, p
    tris = Vector{NTuple{3, Int}}()
    lines = Vector{Vector{Int}}()
    edge_pts = Vector{NTuple{3, Float64}}()
    seen_edges = Set{NTuple{2, Int}}()
    for e in axes(conn, 2), (v, m) in FACES
        k = Tuple(sort([conn[i, e] for i in v]))
        count_of[k] == 1 || continue
        nodes = (conn[v[1], e], conn[v[2], e], conn[v[3], e], conn[m[1], e], conn[m[2], e], conn[m[3], e])
        all(x[3, n] <= ZMAX for n in nodes) || continue
        pv = (vals[e][v[1]], vals[e][v[2]], vals[e][v[3]], vals[e][m[1]], vals[e][m[2]], vals[e][m[3]])
        at(l1, l2, l3) = begin
            N = quad6(l1, l2, l3)
            (sum(N[i] * x[1, nodes[i]] for i in 1:6), sum(N[i] * x[2, nodes[i]] for i in 1:6),
             sum(N[i] * x[3, nodes[i]] for i in 1:6), sum(N[i] * pv[i] for i in 1:6))
        end
        idx = Dict{NTuple{2, Int}, Int}()
        for i in 0:NSUB, j in 0:(NSUB - i)
            push!(pts, at(i / NSUB, j / NSUB, 1 - (i + j) / NSUB))
            idx[(i, j)] = length(pts) - 1
        end
        for i in 0:(NSUB - 1), j in 0:(NSUB - 1 - i)
            push!(tris, (idx[(i, j)], idx[(i + 1, j)], idx[(i, j + 1)]))
            i + j < NSUB - 1 && push!(tris, (idx[(i + 1, j)], idx[(i + 1, j + 1)], idx[(i, j + 1)]))
        end
        for (a, b) in ((1, 2), (2, 3), (3, 1))
            key = minmax(nodes[a], nodes[b])
            key in seen_edges && continue
            push!(seen_edges, key)
            line = Int[]
            for s in 0:NSUB
                l = zeros(3); l[a] = 1 - s / NSUB; l[b] = s / NSUB
                push!(edge_pts, at(l...)[1:3]); push!(line, length(edge_pts) - 1)
            end
            push!(lines, line)
        end
    end
    write_vtp(joinpath(dir, "surface.vtp"), [p[1:3] for p in pts], [p[4] for p in pts], tris)
    write_vtp(joinpath(dir, "edges.vtp"), edge_pts, nothing, lines)
    p = [q[4] for q in pts]
    @printf("%s: t = %.3e s, %d faces, p in [%.3e, %.3e] Pa\n", dir, t,
            length(tris) ÷ NSUB^2, minimum(p), maximum(p))
end

function write_vtp(path, xyz, p, cells)
    polys = eltype(cells) <: Tuple
    open(path, "w") do io
        println(io, """<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">
<PolyData><Piece NumberOfPoints="$(length(xyz))" NumberOfPolys="$(polys ? length(cells) : 0)" NumberOfLines="$(polys ? 0 : length(cells))">""")
        if p !== nothing
            println(io, """<PointData Scalars="pressure"><DataArray type="Float64" Name="pressure" format="ascii">""")
            foreach(v -> @printf(io, "%.6e\n", v), p)
            println(io, "</DataArray></PointData>")
        end
        println(io, """<Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">""")
        foreach(q -> @printf(io, "%.9e %.9e %.9e\n", q...), xyz)
        println(io, "</DataArray></Points>")
        tag = polys ? "Polys" : "Lines"
        println(io, "<$tag><DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">")
        foreach(c -> println(io, join(c, " ")), cells)
        println(io, "</DataArray><DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">")
        println(io, join(cumsum(length.(cells)), " "))
        println(io, "</DataArray></$tag>")
        println(io, "</Piece></PolyData></VTKFile>")
    end
end

foreach(extract, ARGS)
