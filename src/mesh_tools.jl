# Mesh preprocessing: conversion of a tetrahedral Exodus mesh to TETRA15.
#
# The enriched tetrahedron `Tet{EnrichedLagrange, 2}` of
# ReferenceFiniteElements has fifteen nodes: the ten of TETRA10 followed by
# one node at the centroid of each face and one at the centroid of the
# element.  Meshing tools produce TETRA4 and TETRA10; this file adds the
# remaining nodes to such a mesh and writes the result as a TETRA15 block.
#
# Node order of a TETRA15 element (the convention of the Exodus library,
# from the side-node table in ex_get_side_set_node_list.c):
#   1-4    vertices
#   5-10   edge midpoints of the edges 1-2, 2-3, 1-3, 1-4, 2-4, 3-4
#   11     face 1-3-2 (side 4)
#   12     face 2-3-4 (side 2)
#   13     face 1-4-3 (side 3)
#   14     face 1-2-4 (side 1)
#   15     centroid
#
# The coordinates of the new nodes are the images of the reference face
# centroids and of the reference centroid under the quadratic isoparametric
# map of the TETRA10 element, so a curved face keeps its shape.  For a
# straight-sided element they are the arithmetic means of the vertices.
#
# Node sets receive the new nodes that lie on the set: an edge midpoint
# whose two end nodes are in the set (TETRA4 input), and a face node whose
# six TETRA10 nodes are in the set.  Element centroids are never added.
# Side sets are copied unchanged: an Exodus side set lists elements and
# local side numbers, and neither changes.  Block ids, block names, node and
# element id maps and set names are preserved.

# Exodus TETRA10 edge order (node 4 + k lies on edge k).
const _TET_EDGES = ((1, 2), (2, 3), (1, 3), (1, 4), (2, 4), (3, 4))

# Exodus TETRA15 face nodes 11-14: the vertices of each face and the edges
# (indices into _TET_EDGES) that bound it.
const _TET15_FACES = ((1, 3, 2), (2, 3, 4), (1, 4, 3), (1, 2, 4))

# Edge index of the edge between vertices a and b.
function _tet_edge_index(a::Int, b::Int)
    for (k, (p, q)) in enumerate(_TET_EDGES)
        ((p, q) == (a, b) || (p, q) == (b, a)) && return k
    end
    error("no edge between tetrahedron vertices $a and $b")
end

_sorted_key(v::NTuple{2, Int}) = v[1] < v[2] ? v : (v[2], v[1])
_sorted_key(v::NTuple{3, Int}) = Tuple(sort!(collect(v)))

"""
    tetra15_mesh(input::AbstractString, output::AbstractString)

Read the Exodus mesh `input`, whose element blocks must all be TETRA4 or
TETRA10, add a node at the centroid of every face and of every element, and
write the result to `output` with the blocks typed TETRA15.  Returns the
number of nodes written.  Nodes on faces and edges shared by two elements
are created once.
"""
function tetra15_mesh(input::AbstractString, output::AbstractString)
    isfile(input) || error("tetra15_mesh: input mesh \"$input\" does not exist")
    exo = Exodus.ExodusDatabase(input, "r")
    try
        return _tetra15_mesh(exo, output)
    finally
        Exodus.close(exo)
    end
end

function _tetra15_mesh(exo, output::AbstractString)
    init = exo.init
    Exodus.num_dimensions(init) == 3 ||
        error("tetra15_mesh: the mesh is $(Exodus.num_dimensions(init))-dimensional; " *
              "TETRA15 conversion needs a three-dimensional mesh")
    coords = Exodus.read_coordinates(exo)
    n_old  = size(coords, 2)
    xs = [Vector{Float64}(coords[:, n]) for n in 1:n_old]

    blocks      = Exodus.read_sets(exo, Exodus.Block)
    block_names = Exodus.read_names(exo, Exodus.Block)
    for b in blocks
        t = uppercase(String(b.elem_type))
        t in ("TETRA4", "TETRA", "TET", "TET4", "TETRA10") ||
            error("tetra15_mesh: block $(b.id) has element type \"$t\"; " *
                  "only TETRA4 and TETRA10 blocks can be converted to TETRA15")
    end

    # New nodes, created once per edge and once per face.
    edge_node = Dict{NTuple{2, Int}, Int}()
    # face key => (face node, the three edge-midpoint nodes of the face)
    face_node = Dict{NTuple{3, Int}, Tuple{Int, NTuple{3, Int}}}()
    new_node!(x) = (push!(xs, x); length(xs))

    # Values of the ten TETRA10 shape functions at a reference face centroid
    # (for the three vertices and three edge nodes of that face) and at the
    # reference centroid.
    w_face_vertex, w_face_edge = -1 / 9, 4 / 9
    w_cent_vertex, w_cent_edge = -1 / 8, 1 / 4

    new_conns = Vector{Matrix{Int}}(undef, length(blocks))
    for (ib, b) in enumerate(blocks)
        conn = Matrix{Int}(b.conn)
        ne   = size(conn, 2)
        c15  = zeros(Int, 15, ne)
        for e in 1:ne
            v = conn[1:4, e]
            c15[1:4, e] .= v
            # edge midpoints
            for (k, (p, q)) in enumerate(_TET_EDGES)
                if size(conn, 1) == 10
                    c15[4 + k, e] = conn[4 + k, e]
                else
                    key = _sorted_key((v[p], v[q]))
                    c15[4 + k, e] = get!(edge_node, key) do
                        new_node!((xs[v[p]] .+ xs[v[q]]) ./ 2)
                    end
                end
            end
            # face nodes
            for (f, (a, bb, c)) in enumerate(_TET15_FACES)
                key = _sorted_key((v[a], v[bb], v[c]))
                m = (c15[4 + _tet_edge_index(a, bb), e],
                     c15[4 + _tet_edge_index(bb, c), e],
                     c15[4 + _tet_edge_index(c, a), e])
                c15[10 + f, e] = get!(face_node, key) do
                    x = w_face_vertex .* (xs[v[a]] .+ xs[v[bb]] .+ xs[v[c]]) .+
                        w_face_edge .* (xs[m[1]] .+ xs[m[2]] .+ xs[m[3]])
                    (new_node!(x), m)
                end[1]
            end
            # centroid
            x = w_cent_vertex .* sum(xs[v[i]] for i in 1:4) .+
                w_cent_edge .* sum(xs[c15[4 + k, e]] for k in 1:6)
            c15[15, e] = new_node!(x)
        end
        new_conns[ib] = c15
    end
    n_new = length(xs)

    # Node sets: extend with the new edge and face nodes whose defining nodes
    # are all in the set.
    nset_ids   = Exodus.read_ids(exo, Exodus.NodeSet)
    nset_names = Exodus.read_names(exo, Exodus.NodeSet)
    nsets      = Exodus.read_sets(exo, Exodus.NodeSet)
    new_nsets  = Vector{Vector{Int}}(undef, length(nsets))
    for (i, ns) in enumerate(nsets)
        members = Set{Int}(Int.(ns.nodes))
        added   = Set{Int}()
        for (key, n) in edge_node
            all(k -> k in members, key) && push!(added, n)
        end
        for (key, (n, mids)) in face_node
            all(k -> k in members, key) || continue
            # an edge midpoint created above is in the set when its end
            # nodes are; an existing midpoint (TETRA10 input) must be listed
            all(k -> k in members || k in added, mids) || continue
            push!(added, n)
        end
        new_nsets[i] = vcat(Int.(ns.nodes), sort!(collect(added)))
    end

    sset_ids   = Exodus.read_ids(exo, Exodus.SideSet)
    sset_names = Exodus.read_names(exo, Exodus.SideSet)
    ssets      = Exodus.read_sets(exo, Exodus.SideSet)

    node_map = Int.(Exodus.read_id_map(exo, Exodus.NodeMap))
    elem_map = Int.(Exodus.read_id_map(exo, Exodus.ElementMap))
    next_id  = maximum(node_map; init = 0)
    new_node_map = vcat(node_map, next_id .+ (1:(n_new - n_old)))

    # Write.
    isfile(output) && rm(output)
    n_elems = sum(size(c, 2) for c in new_conns)
    out_init = Exodus.Initialization{Int32}(3, n_new, n_elems, length(blocks),
                                            length(nsets), length(ssets))
    out = Exodus.ExodusDatabase{Int32, Int32, Int32, Float64}(output, "w", out_init)
    try
        X = Matrix{Float64}(undef, 3, n_new)
        for n in 1:n_new
            X[:, n] .= xs[n]
        end
        Exodus.write_coordinates(out, X)
        Exodus.write_names(out, Exodus.Block, String.(block_names))
        for (ib, b) in enumerate(blocks)
            Exodus.write_block(out, Int(b.id), "TETRA15", Int32.(new_conns[ib]))
        end
        for (i, ns) in enumerate(nsets)
            Exodus.write_set(out, Exodus.NodeSet(Int32(ns.id), Int32.(new_nsets[i])))
        end
        isempty(nsets) || Exodus.write_names(out, Exodus.NodeSet, String.(nset_names))
        for ss in ssets
            Exodus.write_set(out, Exodus.SideSet(Int32(ss.id), Int32.(ss.elements),
                                                 Int32.(ss.sides), Int32[], Int32[]))
        end
        isempty(ssets) || Exodus.write_names(out, Exodus.SideSet, String.(sset_names))
        Exodus.write_id_map(out, Exodus.NodeMap, Int32.(new_node_map))
        Exodus.write_id_map(out, Exodus.ElementMap, Int32.(elem_map))
    finally
        Exodus.close(out)
    end
    return n_new
end
