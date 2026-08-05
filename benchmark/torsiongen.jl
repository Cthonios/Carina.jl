# Structured HEX8 torsion-bar mesh generator, for the explicit CPU-vs-GPU
# scaling sweep.  Reproduces the geometry of `examples/meshes/torsion/torsion.g`
# (a 0.05 x 0.05 x 1.0 bar centred on the origin, cube elements) at arbitrary
# refinement, so the sweep varies only mesh size.
#
# Usage:  julia --project=. benchmark/torsiongen.jl <N> <out.g>
#   N = elements across the square section; the length gets 20N to keep the
#   elements cubic.  N=20 reproduces torsion.g exactly: 160,000 elements,
#   176,841 nodes, 530,523 DOF.
#
# Block and node-set names match torsion.g ("torsion"; -X/+X/-Y/+Y/-Z/+Z/nsall)
# so the example decks apply unchanged.

import Exodus

const SIDE   = 0.05
const HEIGHT = 1.0

function generate(N::Int, path::String)
    NZ = 20N                      # cube elements: HEIGHT/SIDE = 20
    nx, ny, nz = N + 1, N + 1, NZ + 1
    nnodes = nx * ny * nz
    nelems = N * N * NZ
    @info "Generating $(N)x$(N)x$(NZ) hex torsion bar: " *
          "$nnodes nodes, $nelems elements, $(3nnodes) DOF"

    x = Vector{Float64}(undef, nnodes)
    y = Vector{Float64}(undef, nnodes)
    z = Vector{Float64}(undef, nnodes)
    h = SIDE / N
    idx(i, j, k) = 1 + i + nx * j + nx * ny * k
    for k in 0:NZ, j in 0:N, i in 0:N
        n = idx(i, j, k)
        x[n] = -SIDE / 2 + i * h
        y[n] = -SIDE / 2 + j * h
        z[n] = -HEIGHT / 2 + k * h
    end

    # HEX8 connectivity, standard Exodus node ordering.
    conn = Matrix{Int32}(undef, 8, nelems)
    e = 0
    for k in 0:NZ-1, j in 0:N-1, i in 0:N-1
        e += 1
        conn[1, e] = idx(i,     j,     k)
        conn[2, e] = idx(i + 1, j,     k)
        conn[3, e] = idx(i + 1, j + 1, k)
        conn[4, e] = idx(i,     j + 1, k)
        conn[5, e] = idx(i,     j,     k + 1)
        conn[6, e] = idx(i + 1, j,     k + 1)
        conn[7, e] = idx(i + 1, j + 1, k + 1)
        conn[8, e] = idx(i,     j + 1, k + 1)
    end

    face(sel) = Int32[idx(i, j, k) for k in 0:NZ, j in 0:N, i in 0:N if sel(i, j, k)]
    nsets = [
        ("-X",    face((i, j, k) -> i == 0)),
        ("+X",    face((i, j, k) -> i == N)),
        ("-Y",    face((i, j, k) -> j == 0)),
        ("+Y",    face((i, j, k) -> j == N)),
        ("-Z",    face((i, j, k) -> k == 0)),
        ("+Z",    face((i, j, k) -> k == NZ)),
        ("nsall", Int32.(1:nnodes)),
    ]

    init = Exodus.Initialization{Int32}(3, nnodes, nelems, 1, length(nsets), 0)
    rm(path; force=true)
    exo = Exodus.ExodusDatabase{Int32, Int32, Int32, Float64}(path, "w", init)
    try
        Exodus.write_coordinates(exo, permutedims(hcat(x, y, z)))
        Exodus.write_block(exo, Exodus.Block(Int32(1), nelems, 8, "HEX8", conn))
        Exodus.write_name(exo, Exodus.Block, Int32(1), "torsion")
        for (n, (name, nodes)) in enumerate(nsets)
            Exodus.write_set(exo, Exodus.NodeSet(Int32(n), nodes))
            Exodus.write_name(exo, Exodus.NodeSet, Int32(n), name)
        end
    finally
        Exodus.close(exo)
    end
    @info "Wrote $path ($(round(filesize(path) / 1e6; digits=1)) MB)"
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error("Usage: torsiongen.jl <N> <out.g>")
    generate(parse(Int, ARGS[1]), ARGS[2])
end
