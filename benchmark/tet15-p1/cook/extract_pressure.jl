# Pressure and displacement of a Cook's membrane run, at full load, in a
# compact form for the figures (the Exodus files of the h = 2 runs with
# --stress are 100 MB and more).
#
#   julia --project=. benchmark/tet15-p1/cook/extract_pressure.jl runs/<case>-<element>-h<h> ...
#
# For each run directory, reads the last frame of the output (cook.e, or the
# per-rank files cook.e.<np>.<k> of a parallel Albany run) and writes
#   <dir>/pressure.csv       element id, mean pressure p = -tr(σ)/3 over the
#                            quadrature points of the element
#   <dir>/displacement.csv   node id, u_x, u_y, u_z
#   <dir>/pressure_nodes.csv  element id, global ids n1-n4 of its four
#                            vertices, and the values p1-p10 of the fitted
#                            pressure (below) at the ten nodes of a quadratic
#                            tetrahedron: vertices 1-4, then the midpoints of
#                            the edges 1-2, 2-3, 3-1, 1-4, 2-4, 3-4
# with the global element and node ids of the output's id maps, which equal
# those of the input mesh in both codes.
#
# The fitted pressure is the L2 projection of the quadrature-point values
# onto polynomials of the element, evaluated with its quadrature rule: it
# minimizes Σ_q w_q (p_q − p(ξ_q))².  With the 14-point rule of Keast
# (TETRA15, exact to degree 5) the polynomials are the quadratics, whose
# products the rule integrates exactly; the linear projection of TETRA15,
# linear in the element, and the constant projection are recovered to
# rounding error, and the pointwise pressure of TETRA15, of higher degree,
# is approximated (largest residual 0.11 of max |p| at h = 2, plastic).  With
# the 4-point rule (TETRA10) the polynomials are the linear functions, which
# interpolate the four values.  With one value per element (the composite
# tetrahedron of Albany, whose volume-averaged J makes the pressure equal at
# its five points) the pressure is that constant.  Carina and Albany
# use the same 4-point rule in the same point order (checked against each
# other on the TETRA10 results); the points and weights are those of
# ReferenceFiniteElements.

using Exodus
using LinearAlgebra
using Printf
import ReferenceFiniteElements as RFE

const RULES = Dict(n => RFE.cell_quadrature_points_and_weights(RFE.Tet{RFE.Lagrange, 1}(),
                                                               RFE.GaussLegendre(d))
                   for (n, d) in ((1, 1), (4, 2), (14, 5)))

const P2_NODES = ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                  [0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.0],
                  [0.0, 0.0, 0.5], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5])

monomials(ξ, degree) = degree == 1 ? [1.0, ξ[1], ξ[2], ξ[3]] :
    [1.0, ξ[1], ξ[2], ξ[3], ξ[1]^2, ξ[2]^2, ξ[3]^2, ξ[1] * ξ[2], ξ[1] * ξ[3], ξ[2] * ξ[3]]

# Values at the ten nodes of the fitted polynomial, for the quadrature values
# P[q][e] of every element e of a block.
function node_values(P)
    nq = length(P)
    nq == 1 && return [ntuple(_ -> v, 10) for v in P[1]]
    ξ, w = RULES[nq]
    degree = nq == 14 ? 2 : 1
    A = reduce(vcat, (monomials(ξ[:, q], degree)' for q in 1:nq))
    E = reduce(vcat, (monomials(x, degree)' for x in P2_NODES))
    L = E * ((A' * Diagonal(w) * A) \ (A' * Diagonal(w)))
    return [Tuple(L * [P[q][e] for q in 1:nq]) for e in eachindex(P[1])]
end

function output_files(dir)
    fs = filter(f -> f == "cook.e" || startswith(f, "cook.e."), readdir(dir))
    parallel = filter(f -> f != "cook.e", fs)
    files = isempty(parallel) ? fs : parallel
    isempty(files) && error("no output file in $dir")
    return joinpath.(dir, sort(files))
end

# Pressure −tr(σ)/3 at each quadrature point, P[q][e], for the element
# variables of one block.
function qp_pressure(exo, step, block, names)
    if any(startswith("sigma_xx_"), names)                 # Carina
        nq = count(startswith("sigma_xx_"), names)
        return [-sum(read_values(exo, ElementVariable, step, block, "sigma_$(c)$(c)_$q")
                     for c in ("x", "y", "z")) ./ 3 for q in 1:nq]
    elseif any(startswith("Cauchy_Stress_"), names)        # Albany
        cs = filter(startswith("Cauchy_Stress_"), names)
        first_index = maximum(parse(Int, split(n, "_")[3]) for n in cs)
        if first_index == 3
            # TETRA10: Cauchy_Stress_<row>_<(q-1)*3+col>, zero-padded
            nq = count(startswith("Cauchy_Stress_1_"), names) ÷ 3
            w = length(split(first(filter(startswith("Cauchy_Stress_1_"), names)), "_")[end])
            return [-sum(read_values(exo, ElementVariable, step, block,
                                     "Cauchy_Stress_$(i)_" * lpad(3 * (q - 1) + i, w, '0'))
                         for i in 1:3) ./ 3 for q in 1:nq]
        elseif first_index == 9
            # Composite tetrahedron: Cauchy_Stress_<component 1-9, row-major>_<point>.
            # With the volume-averaged J the pressure is the same at every point;
            # it is returned as one value per element.
            np = count(startswith("Cauchy_Stress_1_"), names)
            P = [-sum(read_values(exo, ElementVariable, step, block, "Cauchy_Stress_$(c)_$q")
                      for c in (1, 5, 9)) ./ 3 for q in 1:np]
            spread = maximum(maximum(abs.(P[q] .- P[1])) for q in 1:np)
            spread <= 1e-8 * maximum(abs.(P[1])) ||
                error("composite tetrahedron: pressure differs between points by $spread")
            return [P[1]]
        end
        error("unrecognized layout of the Albany stress output")
    end
    error("no stress output; rerun with --stress")
end

function extract(dir)
    p_of = Dict{Int, Float64}()
    v_of = Dict{Int, Tuple{NTuple{4, Int}, NTuple{10, Float64}}}()
    u_of = Dict{Int, NTuple{3, Float64}}()
    for f in output_files(dir)
        exo = ExodusDatabase(f, "r")
        step = read_number_of_time_steps(exo)
        enames = read_names(exo, ElementVariable)
        emap = Int.(read_id_map(exo, ElementMap))
        nmap = Int.(read_id_map(exo, NodeMap))
        offset = 0
        for b in read_sets(exo, Block)
            name = read_names(exo, Block)[findfirst(==(b.id), read_ids(exo, Block))]
            P = qp_pressure(exo, step, name, enames)
            vv = node_values(P)
            for k in eachindex(P[1])
                id = emap[offset + k]
                p_of[id] = sum(P[q][k] for q in eachindex(P)) / length(P)
                v_of[id] = (Tuple(nmap[b.conn[1:4, k]]), vv[k])
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
    open(joinpath(dir, "pressure_nodes.csv"), "w") do io
        println(io, "element,n1,n2,n3,n4,", join(("p$k" for k in 1:10), ","))
        for id in sort(collect(keys(v_of)))
            n, v = v_of[id]
            println(io, join(string.((id, n...)), ","), ",", join((@sprintf("%.9e", x) for x in v), ","))
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
