# Faces of the Cook's membrane mesh on the face z = t (the front face), for
# the pressure figures: one row per triangle, with the id of the element it
# belongs to and the ids and reference x, y of its three vertices.
#
#   julia --project=. benchmark/tet15-p1/cook/surface_faces.jl meshes/cook-h2.g faces-h2.csv

using Exodus
using Printf

function main(mesh_file, out)
    exo = ExodusDatabase(mesh_file, "r")
    X = read_coordinates(exo)
    nmap = Int.(read_id_map(exo, NodeMap))
    emap = Int.(read_id_map(exo, ElementMap))
    zmax = maximum(X[3, :])
    tol = 1e-8 * zmax
    faces = ((1, 2, 4), (2, 3, 4), (1, 4, 3), (1, 3, 2))
    rows = String[]
    offset = 0
    for b in read_sets(exo, Block)
        c = b.conn
        for e in axes(c, 2), f in faces
            v = (c[f[1], e], c[f[2], e], c[f[3], e])
            all(abs(X[3, n] - zmax) < tol for n in v) || continue
            push!(rows, @sprintf("%d,%d,%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g", emap[offset + e],
                                 nmap[v[1]], nmap[v[2]], nmap[v[3]],
                                 X[1, v[1]], X[2, v[1]], X[1, v[2]], X[2, v[2]], X[1, v[3]], X[2, v[3]]))
        end
        offset += size(c, 2)
    end
    close(exo)
    open(out, "w") do io
        println(io, "element,n1,n2,n3,x1,y1,x2,y2,x3,y3")
        foreach(r -> println(io, r), rows)
    end
    println("$out: $(length(rows)) faces on z = $zmax")
end

main(ARGS...)
