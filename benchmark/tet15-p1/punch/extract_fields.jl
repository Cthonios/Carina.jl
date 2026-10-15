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
#                    angle in [0, π/2] of u_z at step 20; a bin without
#                    nodes is left out
#   profile.tsv      at step 20, the elements with a vertex on the axis
#                    (r = 0), by the z of their centroid, and the elements
#                    with a face on the top face and a vertex on the symmetry
#                    plane y = 0 (node set symm-y), by the r of their
#                    centroid: line (axis | top), element id, r, z, p, eqps
#   nodal-s10.csv,   per vertex node: id, reference x, y, z, nodal p and eqps,
#   nodal-s20.csv    the common nodal projection of both codes (gist 2026-10-15
#                    07:33Z): the consistent L2 projection, in the measure of
#                    the reference configuration, onto the continuous P1 space
#                    of the element vertices; M u = b with
#                      M_ij = sum_e sum_q w_q phi_i(xi_q) phi_j(xi_q) detJ_q,
#                      b_i  = sum_e sum_q w_q phi_i(xi_q) f_q detJ_q,
#                    phi the barycentric (P1) functions, the 14-point degree-5
#                    rule of the element, detJ of the element's quadratic
#                    (10-node) geometry.  f_q is the quadrature-point value for
#                    Carina and the element value at every point for Sierra/SM
#                    (its stress is one value per element; its eqps the mean of
#                    its 14 integration-point values).  The other nodes take
#                    the P1 field at their position: an edge node the mean of
#                    its two vertices, a face node of its face's three, the
#                    center node of the element's four; they are not written,
#                    as they follow from the vertices.
#   nodal-all-s20.csv  the same at every node, for render.py (nodal=)

using Exodus
using LinearAlgebra
using Printf
using SparseArrays
using Statistics
import ReferenceFiniteElements as RFE

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

# The common nodal projection (see the header): the P1 mass matrix, its
# Cholesky factor and the integration weights w_q phi_i(xi_q) detJ_q per element.
function p1_projector(X, conn)
    ne = size(conn, 2)
    re = RFE.ReferenceFE(RFE.Tet{RFE.Lagrange, 2}(), RFE.GaussLegendre(5))
    nq = RFE.num_cell_quadrature_points(re)
    W = zeros(4, nq, ne)                                  # w_q phi_i(xi_q) detJ_q
    I, J, V = Int[], Int[], Float64[]
    for e in 1:ne
        Xe = X[:, conn[1:10, e]]
        for q in 1:nq
            xi = RFE.cell_quadrature_point(re, q)
            phi = (1 - xi[1] - xi[2] - xi[3], xi[1], xi[2], xi[3])
            dJ = det(Xe * Matrix(RFE.cell_shape_function_gradient(re, q)))
            dJ > 0 || error("element $e: detJ = $dJ at point $q")
            wq = RFE.cell_quadrature_weight(re, q) * dJ
            for i in 1:4
                W[i, q, e] = wq * phi[i]
                for j in 1:4
                    push!(I, conn[i, e]); push!(J, conn[j, e]); push!(V, wq * phi[i] * phi[j])
                end
            end
        end
    end
    n = size(X, 2)
    M = sparse(I, J, V, n, n)
    verts = sort!(unique(vec(conn[1:4, :])))
    F = cholesky(Symmetric(M[verts, verts]))
    # The other nodes: the vertices whose barycentric coordinate (in the
    # straight-sided tetrahedron of the vertices) is not near zero.
    others = Dict{Int, Vector{Int}}()
    for e in 1:ne
        v = conn[1:4, e]
        T = X[:, v[2:4]] .- X[:, v[1]]
        for a in 5:size(conn, 1)
            haskey(others, conn[a, e]) && continue
            l = T \ (X[:, conn[a, e]] .- X[:, v[1]])
            lam = (1 - sum(l), l[1], l[2], l[3])
            sup = [v[i] for i in 1:4 if lam[i] > 0.1]
            length(sup) in (2, 3, 4) || error("element $e, node $(conn[a, e]): barycentric $lam")
            others[conn[a, e]] = sup
        end
    end
    return (; W, F, verts, others, n)
end

# The projection of the quadrature-point values Q[q][e] onto the nodes.
function project(pr, Q)
    b = zeros(pr.n)
    nq = size(pr.W, 2)
    for e in axes(pr.W, 3), q in 1:nq, i in 1:4
        b[pr.conn[i, e]] += pr.W[i, q, e] * Q[length(Q) == 1 ? 1 : q][e]
    end
    u = fill(NaN, pr.n)
    u[pr.verts] = pr.F \ b[pr.verts]
    for (a, sup) in pr.others
        u[a] = sum(u[sup]) / length(sup)
    end
    return u
end

function write_nodal(path, X, nmap, verts, p, q)
    open(path, "w") do io
        println(io, "id,x,y,z,p,eqps")
        for a in verts
            @printf(io, "%d,%.9e,%.9e,%.9e,%.9e,%.9e\n", nmap[a], X[1, a], X[2, a], X[3, a], p[a], q[a])
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

    pr = merge(p1_projector(X, conn), (; conn))
    nmap = Int.(read_id_map(exo, NodeMap))
    @printf("P1 projection: %d vertices of %d nodes, volume %.9e\n", length(pr.verts), size(X, 2),
            sum(pr.W[1:4, :, :]))
    stats = Dict{Int, Any}()
    for (s, k) in ((10, k10), (20, k20))
        P, Q = element_values(exo, k, bname, enames)
        write_fields(out("fields-s$s.csv"), emap, C, P, Q)
        up, uq = project(pr, P), project(pr, Q)
        write_nodal(out("nodal-s$s.csv"), X, nmap, pr.verts, up, uq)
        s == 20 && write_nodal(out("nodal-all-s20.csv"), X, nmap, axes(X, 2), up, uq)
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
            isempty(v) && continue
            @printf(io, "%.6e\t%.6e\t%d\t%.9e\t%.9e\n", (b - 1) * A / 8, b * A / 8, length(v),
                    mean(v), length(v) < 2 ? 0.0 : std(v; corrected = false))
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
