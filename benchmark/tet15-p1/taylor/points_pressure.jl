# Pressure at the integration points of the elements with a face on the impact
# face of the Taylor bar, from a Sierra/SM run with the element variable
# hydrostatic_stress (one value per integration point):
#
#   julia --project=. benchmark/tet15-p1/taylor/points_pressure.jl <run dir> ...
#
# For each run directory, reads the last frame of taylor.e and writes
#   <dir>/points.vtp   the integration points of those elements at their
#                      current positions x = X + u, with the pressure
#                      p = -tr(σ)/3 = -hydrostatic_stress (positive in
#                      compression, Pa) at each point
# and prints the element size h (mean edge length of those faces), which
# render_points.py uses to size the dots.  The points are the ones whose
# pressure enters the internal force: the 4 points of the degree-2 Gauss rule
# for TET10 and CT10; the 14 points of the degree-5 rule for TET15 and
# TET15-P1; and, for TET15-SR4, its 4 pressure points (the degree-2 Gauss
# rule), which follow its 15 full points in the output.  The positions use
# the quadratic basis of the ten-node tetrahedron or the nodal basis of the
# fifteen-node tetrahedron (quadratic plus face and interior bubbles).
# The edges of the face are written by pressure.jl (edges.vtp).

using Exodus
using Printf

const A2 = (5 - sqrt(5)) / 20
const B2 = 1 - 3A2
const GAUSS4 = ([A2, A2, A2], [B2, A2, A2], [A2, B2, A2], [A2, A2, B2])
const (A5, B5) = (0.0927352503108912264023239137370306, 0.7217942490673263207930282587889082)
const (C5, D5) = (0.3108859192633006097973457337634578, 0.0673422422100981706079627987096265)
const (E5, F5) = (0.0455037041256496376438588448192628, 0.4544962958743503623561411551807372)
const KEAST14 = ([A5, A5, A5], [B5, A5, A5], [A5, B5, A5], [A5, A5, B5],
                 [C5, C5, C5], [D5, C5, C5], [C5, D5, C5], [C5, C5, D5],
                 [E5, E5, F5], [E5, F5, E5], [F5, E5, E5], [E5, F5, F5], [F5, E5, F5], [F5, F5, E5])

# Faces of the ten-node tetrahedron (Exodus side order): vertex triples.
const FACES = ((1, 2, 4), (2, 3, 4), (1, 4, 3), (1, 3, 2))
# Fifteen-node tetrahedron: face nodes 11-14 on the vertex triples below, in
# that order, and the centroid node 15 (TETRA15 = TETRA14 + centroid).
const FACE_VERTICES = ((1, 3, 2), (2, 3, 4), (1, 4, 3), (1, 2, 4))
const EDGE_VERTICES = ((1, 2), (2, 3), (3, 1), (1, 4), (2, 4), (3, 4))

function shape10(ξ)
    l = (1 - ξ[1] - ξ[2] - ξ[3], ξ[1], ξ[2], ξ[3])
    N = zeros(10)
    for i in 1:4
        N[i] = l[i] * (2l[i] - 1)
    end
    for (e, (a, b)) in enumerate(EDGE_VERTICES)
        N[4 + e] = 4l[a] * l[b]
    end
    return N
end

function shape15(ξ)
    l = (1 - ξ[1] - ξ[2] - ξ[3], ξ[1], ξ[2], ξ[3])
    p = prod(l)
    B = 256p
    F = [27 * l[a] * l[b] * l[c] - 108p for (a, b, c) in FACE_VERTICES]
    N = zeros(15)
    for i in 1:4
        N[i] = l[i] * (2l[i] - 1) + B / 8 +
               sum(F[f] / 9 for f in 1:4 if i in FACE_VERTICES[f])
    end
    for (e, (a, b)) in enumerate(EDGE_VERTICES)
        N[4 + e] = 4l[a] * l[b] - B / 4 -
                   sum(4F[f] / 9 for f in 1:4 if a in FACE_VERTICES[f] && b in FACE_VERTICES[f])
    end
    N[11:14] .= F
    N[15] = B
    return N
end

function extract(dir)
    exo = ExodusDatabase(joinpath(dir, "taylor.e"), "r")
    step = read_number_of_time_steps(exo)
    t = read_time(exo, step)
    X = read_coordinates(exo)
    vn = read_names(exo, NodalVariable)
    base = any(==("displ_x"), vn) ? "displ" : any(==("displacement_x"), vn) ? "displacement" : "disp"
    x = X .+ reduce(vcat, (read_values(exo, NodalVariable, step, "$(base)_$c")'
                           for c in ("x", "y", "z")))
    enames = sort(filter(startswith("hydrostatic_stress_"), read_names(exo, ElementVariable)))
    nq = length(enames)
    nq > 0 || error("no hydrostatic_stress output in $dir")
    # TET15-SR4: 15 full points, then the 4 pressure points.
    used = nq == 19 ? enames[16:19] : enames
    ξs = length(used) == 4 ? GAUSS4 : length(used) == 14 ? KEAST14 :
         error("$(length(used)) integration points: no rule known")
    pts = Vector{NTuple{4, Float64}}()
    hsum, hcount = 0.0, 0
    for b in read_sets(exo, Block)
        name = read_names(exo, Block)[findfirst(==(b.id), read_ids(exo, Block))]
        conn = Int.(b.conn)
        nn = size(conn, 1)
        shape = nn == 15 ? shape15 : nn == 10 ? shape10 : error("$nn-node elements")
        P = [-read_values(exo, ElementVariable, step, name, v) for v in used]
        N = [shape(ξ) for ξ in ξs]
        # Elements with a face on the impact face z = 0 (u_z = 0 there).
        for e in axes(conn, 2)
            onface = false
            for v in FACES
                if all(abs(X[3, conn[i, e]]) < 1e-9 for i in v)
                    onface = true
                    for (a, c) in ((1, 2), (2, 3), (3, 1))
                        hsum += hypot((X[:, conn[v[a], e]] .- X[:, conn[v[c], e]])...)
                        hcount += 1
                    end
                end
            end
            onface || continue
            for q in eachindex(ξs)
                xq = sum(N[q][a] .* x[:, conn[a, e]] for a in 1:nn)
                push!(pts, (xq[1], xq[2], xq[3], P[q][e]))
            end
        end
    end
    close(exo)
    open(joinpath(dir, "points.vtp"), "w") do io
        println(io, """<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">
<PolyData><Piece NumberOfPoints="$(length(pts))" NumberOfVerts="$(length(pts))">
<PointData Scalars="pressure"><DataArray type="Float64" Name="pressure" format="ascii">""")
        foreach(p -> @printf(io, "%.6e\n", p[4]), pts)
        println(io, """</DataArray></PointData>
<Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">""")
        foreach(p -> @printf(io, "%.9e %.9e %.9e\n", p[1], p[2], p[3]), pts)
        println(io, """</DataArray></Points>
<Verts><DataArray type="Int64" Name="connectivity" format="ascii">""")
        println(io, join(0:(length(pts) - 1), " "))
        println(io, "</DataArray><DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">")
        println(io, join(1:length(pts), " "))
        println(io, "</DataArray></Verts></Piece></PolyData></VTKFile>")
    end
    p = [q[4] for q in pts]
    h = hsum / hcount
    @printf("%s: t = %.3e s, %d points (%d per element), h = %.3e m, p in [%.3e, %.3e] Pa\n",
            dir, t, length(pts), length(ξs), h, minimum(p), maximum(p))
    open(joinpath(dir, "points.txt"), "w") do io
        @printf(io, "%.6e\n", h)
    end
end

foreach(extract, ARGS)
