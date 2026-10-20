# Rank of the consistent mass of the fifteen-node tetrahedron under the
# fourteen-point degree-five rule, on one element and assembled on a mesh.
# The scalar mass is sum_q w_q N(x_q) N(x_q)^T, a sum of 14 rank-one terms,
# so on one element its rank is at most 14 for 15 functions.
include("common.jl")
include("element.jl")
using Printf

function scalar_mass(coords, conn, pts, wts)
    nelem = size(conn, 2); nn = size(coords, 2)
    faces = Dict{NTuple{3,Int},Int}()
    fid = zeros(Int, 4, nelem)
    for e in 1:nelem, f in 1:4
        key = Tuple(sort([conn[m, e] for m in _TET_FACES[f]]))
        fid[f, e] = get!(faces, key, length(faces) + 1)
    end
    nf = length(faces); n = nn + nelem + nf
    M = zeros(n, n)
    el = ref_element(2)
    for e in 1:nelem
        X = coords[:, conn[1:10, e]]
        dofs = vcat(conn[1:10, e], nn + e, [nn + nelem + fid[f, e] for f in 1:4])
        for q in eachindex(wts)
            xi = pts[:, q]
            Jm = X * RFE.shape_function_gradient(el, xi)
            N, _ = enriched_shape(xi)
            M[dofs, dofs] .+= (wts[q] * det(Jm)) .* (N * N')
        end
    end
    return Symmetric(M)
end

function report(name, M)
    ev = eigvals(M)
    @printf("  %-34s n = %-5d  lam_max = %.3e  lam_min = %.3e  #(lam < 1e-12 lam_max) = %d\n",
            name, size(M, 1), ev[end], ev[1], count(<(1e-12 * ev[end]), ev))
end

function main()
    p14, w14 = keast14()
    p7, w7 = tet_rule(9)               # exact for the degree-8 mass integrand
    println("Scalar consistent mass of the enriched space (rank is basis independent).")
    c1 = [0.0 1 0 0; 0 0 1 0; 0 0 0 1]
    co, cn = promote_to_p2(c1, reshape([1, 2, 3, 4], 4, 1))
    report("one element, 14-point rule", scalar_mass(co, cn, p14, w14))
    report("one element, exact", scalar_mass(co, cn, p7, w7))
    for N in (1, 2, 3)
        co, cn = mesh_of(N, 2)
        report("Freudenthal N = $N, 14-point", scalar_mass(co, cn, p14, w14))
        report("Freudenthal N = $N, exact", scalar_mass(co, cn, p7, w7))
    end
end
main()

# Condition numbers of the reference-element consistent mass under exact
# integration, in the hierarchical and the nodal basis.
let (p, w) = tet_rule(9), A = nodal_transform()
    Mh = zeros(15, 15); Mn = zeros(15, 15)
    for q in eachindex(w)
        N, _ = enriched_shape(p[:, q]); Nn = A * N
        Mh .+= w[q] .* (N * N'); Mn .+= w[q] .* (Nn * Nn')
    end
    @printf("Reference element, exact integration: cond M hierarchical = %.1f, nodal = %.1f\n", cond(Mh), cond(Mn))
end
