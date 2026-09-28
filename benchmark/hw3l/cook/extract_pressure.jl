# Pressure and displacement of a Cook's membrane run, at full load, in a
# compact form for the figures (the Exodus files of the h = 2 runs with
# --stress are 100 MB and more).
#
#   julia --project=. benchmark/hw3l/cook/extract_pressure.jl runs/<case>-<element>-h<h> ...
#
# For each run directory, reads the last frame of the output (cook.e, or the
# per-rank files cook.e.<np>.<k> of a parallel Albany run) and writes
#   <dir>/pressure.csv       element id, mean pressure p = -tr(σ)/3 over the
#                            quadrature points of the element
#   <dir>/displacement.csv   node id, u_x, u_y, u_z
# with the global element and node ids of the output's id maps, which equal
# those of the input mesh in both codes.

using Exodus
using Printf

function output_files(dir)
    fs = filter(f -> f == "cook.e" || startswith(f, "cook.e."), readdir(dir))
    parallel = filter(f -> f != "cook.e", fs)
    files = isempty(parallel) ? fs : parallel
    isempty(files) && error("no output file in $dir")
    return joinpath.(dir, sort(files))
end

# Mean over quadrature points of tr(σ), for the element variables of one block.
function mean_trace(exo, step, block, names)
    if any(startswith("sigma_xx_"), names)                 # Carina
        nq = count(startswith("sigma_xx_"), names)
        tr = sum(read_values(exo, ElementVariable, step, block, "sigma_$(c)$(c)_$q")
                 for q in 1:nq for c in ("x", "y", "z"))
        return tr ./ nq
    elseif any(startswith("Cauchy_Stress_"), names)        # Albany: Cauchy_Stress_<row>_<(q-1)*3+col>
        n = count(startswith("Cauchy_Stress_1_"), names)
        nq = n ÷ 3
        w = length(split(first(filter(startswith("Cauchy_Stress_1_"), names)), "_")[end])
        tr = sum(read_values(exo, ElementVariable, step, block,
                             "Cauchy_Stress_$(i)_" * lpad(3 * (q - 1) + i, w, '0'))
                 for q in 1:nq for i in 1:3)
        return tr ./ nq
    end
    error("no stress output; rerun with --stress")
end

function extract(dir)
    p_of = Dict{Int, Float64}()
    u_of = Dict{Int, NTuple{3, Float64}}()
    for f in output_files(dir)
        exo = ExodusDatabase(f, "r")
        step = read_number_of_time_steps(exo)
        enames = read_names(exo, ElementVariable)
        emap = Int.(read_id_map(exo, ElementMap))
        offset = 0
        for b in read_sets(exo, Block)
            name = read_names(exo, Block)[findfirst(==(b.id), read_ids(exo, Block))]
            tr = mean_trace(exo, step, name, enames)
            for k in eachindex(tr)
                p_of[emap[offset + k]] = -tr[k] / 3
            end
            offset += size(b.conn, 2)
        end
        vn = read_names(exo, NodalVariable)
        base = any(==("displ_x"), vn) ? "displ" : "disp"
        ux, uy, uz = (read_values(exo, NodalVariable, step, "$(base)_$c") for c in ("x", "y", "z"))
        for (k, id) in enumerate(Int.(read_id_map(exo, NodeMap)))
            u_of[id] = (ux[k], uy[k], uz[k])
        end
        close(exo)
    end
    open(joinpath(dir, "pressure.csv"), "w") do io
        println(io, "element,p")
        for id in sort(collect(keys(p_of)))
            @printf(io, "%d,%.9e\n", id, p_of[id])
        end
    end
    open(joinpath(dir, "displacement.csv"), "w") do io
        println(io, "node,ux,uy,uz")
        for id in sort(collect(keys(u_of)))
            u = u_of[id]
            @printf(io, "%d,%.9e,%.9e,%.9e\n", id, u...)
        end
    end
    println("$dir: $(length(p_of)) elements, $(length(u_of)) nodes, p in ",
            extrema(values(p_of)))
end

foreach(extract, ARGS)
