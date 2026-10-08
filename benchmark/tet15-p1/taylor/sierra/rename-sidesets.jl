# Copy of a Taylor mesh for Sierra/SM: IOSS rejects a node set and a side set
# of the same name ("impact"), so the side sets get a "_face" suffix.
# Coordinates, connectivity and the node sets are unchanged.
#   julia --project=. benchmark/tet15-p1/taylor/sierra/rename-sidesets.jl <in.g> <out.g>
using Exodus
cp(ARGS[1], ARGS[2]; force = true)
e = ExodusDatabase(ARGS[2], "rw")
names = read_names(e, SideSet)
write_names(e, SideSet, [n * "_face" for n in names])
println(ARGS[2], ": side sets ", read_names(e, SideSet), ", node sets ", read_names(e, NodeSet))
close(e)
