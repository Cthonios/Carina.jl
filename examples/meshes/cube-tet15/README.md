# cube-tet15

The unit cube of `../cube-tet10` (49 TETRA10 elements, 126 nodes) converted
to TETRA15 by

    bin/tetra15 examples/meshes/cube-tet10/cube.g examples/meshes/cube-tet15/cube.g

The conversion adds one node at the centroid of each of the 122 faces and
one at the centroid of each element (297 nodes), keeps the block name
`cube`, the six node sets `ns{x,y,z}{-,+}` (extended by the face nodes that
lie on each set) and the six side sets `ss{x,y,z}{-,+}`.  TETRA15 is the
quadratic tetrahedron with one cubic bubble per face and one quartic
interior bubble, in a nodal basis (`Tet{EnrichedLagrange, 2}` of
ReferenceFiniteElements).
