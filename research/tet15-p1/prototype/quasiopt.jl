# Referee round 2, M15: quasi-optimality of TET15-P1 against kappa.
# The exact displacement of kappa_sweep.jl depends on kappa through its
# gradient part grad(phi)/(lambda + mu), so the error ratio e_u(10^6)/e_u(1)
# compares two different solutions.  Here each error is divided by the best
# approximation of the same solution in the same space in the H1 seminorm,
# |u - u_b|_1 = min over v_h |u - v_h|_1 (Galerkin projection with the vector
# Laplacian, the matrix Kh1).  A discretization that does not lock has a
# ratio Q = e_u / e_best bounded uniformly in kappa and h.
ENV["KS_NOMAIN"] = "1"
include("kappa_sweep.jl")

function best_error(N, bubble, kappa; mu = 1.0)
    lm = kappa - 2mu / 3 + mu
    coords, conn = mesh_of(N, 2)
    a = assemble_all(coords, conn, 2, 1; mu, bubble, q_degree = bubble === :none ? 4 : 6)
    _, gmap = free_dofs(coords); nelem = size(conn, 2)
    _, fmap = bubble in (:face, :full) ? face_table(coords, conn, _BC_ALL) : (0, zeros(Int, 4, nelem))
    el = ref_element(2); qp, qw = tet_rule(12)
    F = zeros(a.nu)
    for e in 1:nelem
        X = coords[:, conn[:, e]]
        rows = element_rows(conn, e, gmap, bubble, nelem, a.nu_nodal, fmap)
        for q in eachindex(qw)
            xi = qp[:, q]; Jm = X * RFE.shape_function_gradient(el, xi)
            x = X * collect(RFE.shape_function_value(el, xi))
            _, dNr = shapes(bubble, xi); dN = dNr / Jm
            Ge = graduex(x, lm); dV = qw[q] * det(Jm)
            for (i, r) in enumerate(rows)
                r == 0 && continue
                A, d = fldmod1(i, NSD)
                F[r] += dV * dot(Ge[d, :], dN[A, :])
            end
        end
    end
    U = cholesky(Symmetric(a.Kh1)) \ F
    eh = 0.0; nh = 0.0
    for e in 1:nelem
        X = coords[:, conn[:, e]]
        rows = element_rows(conn, e, gmap, bubble, nelem, a.nu_nodal, fmap)
        ue = [r == 0 ? 0.0 : U[r] for r in rows]
        for q in eachindex(qw)
            xi = qp[:, q]; Jm = X * RFE.shape_function_gradient(el, xi)
            x = X * collect(RFE.shape_function_value(el, xi))
            _, dNr = shapes(bubble, xi); dN = dNr / Jm
            Gh = zeros(3, 3)
            for A in axes(dN, 1), d in 1:NSD
                Gh[d, :] .+= ue[NSD * (A - 1) + d] .* dN[A, :]
            end
            Ge = graduex(x, lm); dV = qw[q] * det(Jm)
            eh += dV * sum(abs2, Gh - Ge); nh += dV * sum(abs2, Ge)
        end
    end
    return sqrt(eh / nh)
end

println("TET15-P1: relative H1 error e_u, best approximation e_best in the same space, Q = e_u/e_best")
for N in (2, 3, 4, 6, 8, 10, 12)
    line = @sprintf("  N = %-3d", N)
    for k in (1.0, 1e6)
        eu, _, _ = solve_case(N, :full, 4, k)
        eb = best_error(N, :full, k)
        line *= @sprintf("   k/mu = %-6.0e e_u %.4e e_best %.4e Q %.3f", k, eu, eb, eu / eb)
    end
    println(line); flush(stdout)
end
