# Results of one punch run, from its Exodus output, in the files of the
# benchmark, quarter model (gist README of 2026-10-14 22:52Z, "What to report").
#
#   julia --project=. benchmark/tet15-p1/punch/extract_fields.jl <results.e> <outdir> <prefix> [force.tsv]
#
# <results.e> is the output of a Carina run (stress per quadrature point,
# sigma_<ij>_<q>, and eqps_<q>) or of a Sierra/SM run (element stress
# stress_<ij> and element eqps, one value per element, with the nodal
# reaction and the global punch_force).  The output times are t = 0, 0.05,
# ..., 1 (depth d = 0.2a t); step k is the frame at t = 0.05k.  Files written
# to <outdir>, each named <prefix>-<file>:
#   force.tsv        step, depth (m), full punch force (N, positive in
#                    compression), four times the quarter's: from the global
#                    punch_force of Sierra/SM (the sum of the z reactions on
#                    node set punch of the quarter) times 4, or copied from
#                    the force file of the Carina driver, which applies the
#                    factor itself (the optional fourth argument)
#   fields-s10.csv,  per element: id, reference centroid x, y, z, pressure
#   fields-s20.csv   p = -tr(σ)/3 and equivalent plastic strain eqps, the
#                    element means; for Carina also the 14 quadrature-point
#                    values p_1..p_14 and eqps_1..eqps_14
#   jumps.tsv        over the interior faces whose two elements have reference
#                    centroids in r <= 1.5a, z >= H - 1.5a: the jump
#                    |q_L - q_R| of the element means of p and of eqps,
#                    divided by max |q| over the elements of that region; the
#                    mean, the 99th percentile and the maximum, at steps 10
#                    and 20
#   axisymmetry.tsv  the nodes of the top face (z = H in the reference
#                    configuration), binned by reference radius in bins of
#                    width a/8, closed on the right (r_lo, r_hi], so that
#                    the footprint circle r = a ends a bin: per bin the
#                    count, the mean and the standard deviation over the
#                    angle in [0, π/2] of u_z at step 20
#   profile.tsv      at step 20, the elements with a vertex on the axis
#                    (r = 0), by the z of their centroid, and the elements
#                    with a face on the top face and a vertex on the symmetry
#                    plane y = 0 (node set symm-y), by the r of their
#                    centroid: line (axis | top), element id, r, z, p, eqps

using Exodus
using LinearAlgebra
using Printf
using Statistics

const A = 1.0e-3
const H = 4.0e-3
const FACES = ((1, 2, 4), (2, 3, 4), (1, 4, 3), (1, 3, 2))

frame(times, t) = (k = argmin(abs.(times .- t)); abs(times[k] - t) < 1.0e-6 || error("no frame at t = $t"); k)

function element_values(exo, k, block, enames)
    if any(==("stress_xx"), enames)                                  # Sierra/SM
        p = -(read_values(exo, ElementVariable, k, block, "stress_xx") .+
              read_values(exo, ElementVariable, k, block, "stress_yy") .+
              read_values(exo, ElementVariable, k, block, "stress_zz")) ./ 3
        # eqps is written per integration point (eqps_1, ...) when the section
        # has several; the element value is their mean.
        eq = any(==("eqps"), enames) ? ["eqps"] : filter(n -> occursin(r"^eqps_\d+$", n), enames)
        e = sum(Vector{Float64}(read_values(exo, ElementVariable, k, block, n)) for n in eq) ./ length(eq)
        return [Vector{Float64}(p)], [e]
    end
    nq = count(startswith("sigma_xx_"), enames)                      # Carina
    nq > 0 || error("no stress output")
    P = [-(read_values(exo, ElementVariable, k, block, "sigma_xx_$q") .+
           read_values(exo, ElementVariable, k, block, "sigma_yy_$q") .+
           read_values(exo, ElementVariable, k, block, "sigma_zz_$q")) ./ 3 for q in 1:nq]
    Q = [Vector{Float64}(read_values(exo, ElementVariable, k, block, "eqps_$q")) for q in 1:nq]
    return P, Q
end

mean_of(P) = [sum(P[q][e] for q in eachindex(P)) / length(P) for e in eachindex(P[1])]

function write_fields(path, ids, C, P, Q)
    nq = length(P)
    open(path, "w") do io
        print(io, "id,x,y,z,p,eqps")
        nq > 1 && print(io, join((",p_$q" for q in 1:nq)), join((",eqps_$q" for q in 1:nq)))
        println(io)
        pm, qm = mean_of(P), mean_of(Q)
        for e in eachindex(ids)
            @printf(io, "%d,%.9e,%.9e,%.9e,%.9e,%.9e", ids[e], C[1, e], C[2, e], C[3, e], pm[e], qm[e])
            if nq > 1
                for q in 1:nq; @printf(io, ",%.9e", P[q][e]); end
                for q in 1:nq; @printf(io, ",%.9e", Q[q][e]); end
            end
            println(io)
        end
    end
end

function jump_stats(faces, inregion, q)
    qmax = maximum(abs(q[e]) for e in eachindex(q) if inregion[e]; init = 0.0)
    j = [abs(q[a] - q[b]) for (a, b) in faces]
    qmax > 0 || return (qmax, 0.0, 0.0, 0.0)
    j ./= qmax
    return (qmax, mean(j), quantile(j, 0.99), maximum(j))
end

function main(file, outdir, prefix, forcefile = "")
    mkpath(outdir)
    out(name) = joinpath(outdir, "$prefix-$name")
    exo = ExodusDatabase(file, "r")
    X = Matrix{Float64}(read_coordinates(exo))
    times = read_times(exo)
    blocks = read_sets(exo, Block)
    length(blocks) == 1 || error("one block expected")
    bname = read_names(exo, Block)[1]
    conn = Matrix{Int}(blocks[1].conn)
    ne = size(conn, 2)
    emap = Int.(read_id_map(exo, ElementMap))
    enames = read_names(exo, ElementVariable)
    nnames = read_names(exo, NodalVariable)
    nsets = Dict(zip(read_names(exo, NodeSet), read_sets(exo, NodeSet)))
    k10, k20 = frame(times, 0.5), frame(times, 1.0)

    C = zeros(3, ne)
    for e in 1:ne, i in 1:4
        C[:, e] .+= X[:, conn[i, e]] ./ 4
    end
    rc = [hypot(C[1, e], C[2, e]) for e in 1:ne]
    inregion = [rc[e] <= 1.5A && C[3, e] >= H - 1.5A for e in 1:ne]

    # Interior faces of the region, by their sorted vertex triples.
    owner = Dict{NTuple{3, Int}, Int}()
    faces = Tuple{Int, Int}[]
    for e in 1:ne, f in FACES
        key = Tuple(sort([conn[i, e] for i in f]))
        o = pop!(owner, key, 0)
        if o == 0
            owner[key] = e
        elseif inregion[o] && inregion[e]
            push!(faces, (o, e))
        end
    end
    topfaces = Set{Int}(e for (key, e) in owner if all(abs(X[3, n] - H) < 1.0e-12 for n in key))
    @printf("%s: %d elements, %d frames; region: %d elements, %d interior faces\n",
            file, ne, length(times), count(inregion), length(faces))

    stats = Dict{Int, Any}()
    for (s, k) in ((10, k10), (20, k20))
        P, Q = element_values(exo, k, bname, enames)
        write_fields(out("fields-s$s.csv"), emap, C, P, Q)
        pm, qm = mean_of(P), mean_of(Q)
        stats[s] = (jump_stats(faces, inregion, pm), jump_stats(faces, inregion, qm))
        if s == 20
            onaxis = Set(Int.(nsets["axis"].nodes))
            axis_el = [e for e in 1:ne if any(conn[i, e] in onaxis for i in 1:4)]
            sort!(axis_el; by = e -> C[3, e])
            onsymy = Set(Int.(nsets["symm-y"].nodes))
            top_el = sort!([e for e in topfaces if any(conn[i, e] in onsymy for i in 1:4)]; by = e -> rc[e])
            open(out("profile.tsv"), "w") do io
                println(io, "line\tid\tr\tz\tp\teqps")
                for (line, els) in (("axis", axis_el), ("top", top_el)), e in els
                    @printf(io, "%s\t%d\t%.6e\t%.6e\t%.6e\t%.6e\n", line, emap[e], rc[e], C[3, e], pm[e], qm[e])
                end
            end
        end
    end
    open(out("jumps.tsv"), "w") do io
        println(io, "step\tquantity\tmax_abs_in_region\tmean\tp99\tmax\tfaces")
        for s in (10, 20), (name, st) in zip(("p", "eqps"), stats[s])
            @printf(io, "%d\t%s\t%.6e\t%.6e\t%.6e\t%.6e\t%d\n", s, name, st..., length(faces))
        end
    end

    # Axisymmetry of u_z on the top face at step 20.
    uz = read_values(exo, NodalVariable, k20, "displacement_z" in nnames ? "displacement_z" : "displ_z")
    top = [n for n in axes(X, 2) if abs(X[3, n] - H) < 1.0e-12]
    nb = 32                                                   # 4a / (a/8)
    bins = [Float64[] for _ in 1:nb]
    for n in top
        b = clamp(ceil(Int, hypot(X[1, n], X[2, n]) / (A / 8) - 1.0e-9), 1, nb)   # bins (r_lo, r_hi]
        push!(bins[b], uz[n])
    end
    open(out("axisymmetry.tsv"), "w") do io
        println(io, "r_lo\tr_hi\tcount\tmean_uz\tstd_uz")
        for b in 1:nb
            v = bins[b]
            @printf(io, "%.6e\t%.6e\t%d\t%.9e\t%.9e\n", (b - 1) * A / 8, b * A / 8, length(v),
                    isempty(v) ? NaN : mean(v), length(v) < 2 ? 0.0 : std(v; corrected = false))
        end
    end

    # Force.
    steps = [frame(times, 0.05s) for s in 0:20]
    if !isempty(forcefile)
        cp(forcefile, out("force.tsv"); force = true)
    elseif "punch_force" in read_names(exo, GlobalVariable)
        F = [read_values(exo, GlobalVariable, k, 1, 1)[findfirst(==("punch_force"), read_names(exo, GlobalVariable))]
             for k in steps]
        open(out("force.tsv"), "w") do io
            println(io, "# full punch force: 4 x the sum of the z reactions on the footprint nodes of the quarter")
            println(io, "step\tdepth\tforce")
            for (s, f) in zip(0:20, F)
                @printf(io, "%d\t%.6e\t%.9e\n", s, 2.0e-4 * 0.05s, -4f)
            end
        end
    end
    close(exo)
end

main(ARGS...)
