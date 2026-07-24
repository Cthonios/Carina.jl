# Generates two-block.g: a 1 x 1 x 2 m bar of HEX8 elements split into two
# named element blocks along z.
#
# Carina had no multi-block mesh at all (nor does Norma), so per-block materials
# could not be exercised.  This fixture exists specifically to catch the case
# where each block must receive its OWN material -- in particular that materials
# are matched to blocks by mesh order rather than by the order they happen to
# appear in an unordered YAML dict.
#
#   block "lower"  : 0.0 <= z <= 1.0
#   block "upper"  : 1.0 <= z <= 2.0
#
#   node sets:  bottom (z=0), top (z=2), sym_x (x=0), sym_y (y=0)
#
# Run:  julia --project=. examples/meshes/two-block/make_two_block.jl

using Exodus

const NX, NY, NZ = 2, 2, 4        # elements per direction (NZ split 2/2)
const LX, LY, LZ = 1.0, 1.0, 2.0

nnx, nny, nnz = NX + 1, NY + 1, NZ + 1
num_nodes = nnx * nny * nnz

# Node numbering: i fastest, then j, then k (1-based).
nid(i, j, k) = i + (j - 1) * nnx + (k - 1) * nnx * nny

coords = zeros(Float64, 3, num_nodes)
for k in 1:nnz, j in 1:nny, i in 1:nnx
    n = nid(i, j, k)
    coords[1, n] = LX * (i - 1) / NX
    coords[2, n] = LY * (j - 1) / NY
    coords[3, n] = LZ * (k - 1) / NZ
end

# HEX8 connectivity in Exodus ordering: bottom face CCW, then top face CCW.
function hex_conn(i, j, k)
    return Int32[
        nid(i,     j,     k),
        nid(i + 1, j,     k),
        nid(i + 1, j + 1, k),
        nid(i,     j + 1, k),
        nid(i,     j,     k + 1),
        nid(i + 1, j,     k + 1),
        nid(i + 1, j + 1, k + 1),
        nid(i,     j + 1, k + 1),
    ]
end

# Split along z: lower half -> block 1, upper half -> block 2.
kz_split = NZ ÷ 2
lower, upper = Vector{Int32}[], Vector{Int32}[]
for k in 1:NZ, j in 1:NY, i in 1:NX
    push!(k <= kz_split ? lower : upper, hex_conn(i, j, k))
end

conn_lower = reduce(hcat, lower)
conn_upper = reduce(hcat, upper)

# Node sets.
bottom = Int32[nid(i, j, 1)   for j in 1:nny for i in 1:nnx]
top    = Int32[nid(i, j, nnz) for j in 1:nny for i in 1:nnx]
sym_x  = Int32[nid(1, j, k)   for k in 1:nnz for j in 1:nny]
sym_y  = Int32[nid(i, 1, k)   for k in 1:nnz for i in 1:nnx]

out = joinpath(@__DIR__, "two-block.g")
isfile(out) && rm(out)

init = Initialization(
    Int32(3), Int32(num_nodes), Int32(NX * NY * NZ),
    Int32(2),                       # element blocks
    Int32(4),                       # node sets
    Int32(0),                       # side sets
)

exo = ExodusDatabase{Int32, Int32, Int32, Float64}(out, "w", init)

write_coordinates(exo, coords)

write_block(exo, Block(Int32(1), size(conn_lower, 2), 8, "HEX8", conn_lower))
write_block(exo, Block(Int32(2), size(conn_upper, 2), 8, "HEX8", conn_upper))
write_name(exo, Block(exo, 1), "lower")
write_name(exo, Block(exo, 2), "upper")

write_set(exo, NodeSet(Int32(1), bottom))
write_set(exo, NodeSet(Int32(2), top))
write_set(exo, NodeSet(Int32(3), sym_x))
write_set(exo, NodeSet(Int32(4), sym_y))
write_name(exo, NodeSet(exo, 1), "bottom")
write_name(exo, NodeSet(exo, 2), "top")
write_name(exo, NodeSet(exo, 3), "sym_x")
write_name(exo, NodeSet(exo, 4), "sym_y")

close(exo)

println("wrote $out")
println("  nodes    = $num_nodes")
println("  elements = $(NX * NY * NZ)  (lower $(size(conn_lower,2)), upper $(size(conn_upper,2)))")
