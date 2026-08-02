# Structured HEX8 cube mesh generator for the large benchmark cases.
# Nothing bundled is bigger than torsion's 530k DOF; the campaign needs a
# multi-M-DOF problem, and a unit cube with N^3 elements is the cheapest
# reproducible one.
#
# Usage:  julia --project=. benchmark/meshgen.jl <N> <out.g>
#   N=100 -> 101^3 nodes = 3.09M DOF;  N=128 -> 6.44M DOF
#
# Node sets follow the cube.g naming (nsx-/nsx+/nsy-/nsy+/nsz-/nsz+/nsall)
# so the harness cube-style BC blocks apply unchanged.  No side sets: the
# benchmark cases use node-set BCs only.

import Exodus

function generate(N::Int, path::String)
    nn  = N + 1
    nnodes = nn^3
    nelems = N^3
    @info "Generating $(N)^3 hex cube: $nnodes nodes, $nelems elements, $(3nnodes) DOF"

    # Coordinates, node-major; node id = 1 + i + nn*j + nn^2*k (i,j,k in 0:N).
    x = Vector{Float64}(undef, nnodes)
    y = Vector{Float64}(undef, nnodes)
    z = Vector{Float64}(undef, nnodes)
    h = 1.0 / N
    idx(i, j, k) = 1 + i + nn * j + nn * nn * k
    for k in 0:N, j in 0:N, i in 0:N
        n = idx(i, j, k)
        x[n] = i * h; y[n] = j * h; z[n] = k * h
    end

    # HEX8 connectivity, standard Exodus node ordering.
    conn = Matrix{Int32}(undef, 8, nelems)
    e = 0
    for k in 0:N-1, j in 0:N-1, i in 0:N-1
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

    face(sel) = Int32[idx(i, j, k) for k in 0:N, j in 0:N, i in 0:N if sel(i, j, k)]
    nsets = [
        ("nsx-", face((i, j, k) -> i == 0)),
        ("nsx+", face((i, j, k) -> i == N)),
        ("nsy-", face((i, j, k) -> j == 0)),
        ("nsy+", face((i, j, k) -> j == N)),
        ("nsz-", face((i, j, k) -> k == 0)),
        ("nsz+", face((i, j, k) -> k == N)),
        ("nsall", Int32.(1:nnodes)),
    ]

    init = Exodus.Initialization{Int32}(3, nnodes, nelems, 1, length(nsets), 0)
    rm(path; force=true)
    exo = Exodus.ExodusDatabase{Int32, Int32, Int32, Float64}(path, "w", init)
    try
        Exodus.write_coordinates(exo, permutedims(hcat(x, y, z)))
        Exodus.write_block(exo, Exodus.Block(Int32(1), nelems, 8, "HEX8", conn))
        Exodus.write_name(exo, Exodus.Block, Int32(1), "cube")
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
    length(ARGS) == 2 || error("Usage: meshgen.jl <N> <out.g>")
    generate(parse(Int, ARGS[1]), ARGS[2])
end
