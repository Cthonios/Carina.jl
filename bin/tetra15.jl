# Converts a TETRA4 or TETRA10 Exodus mesh to TETRA15 (see src/mesh_tools.jl).
#
# Invoked by the `bin/tetra15` shell wrapper:
#   tetra15 <input.g> <output.g>

import Carina

function main(args)
    if length(args) != 2
        println(stderr, "Usage: tetra15 <input.g> <output.g>")
        return 1
    end
    n = Carina.tetra15_mesh(args[1], args[2])
    println("tetra15: wrote $(args[2]) with $n nodes")
    return 0
end

exit(main(ARGS))
