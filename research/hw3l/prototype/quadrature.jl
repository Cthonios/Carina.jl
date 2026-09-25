# Which quadrature rule the enriched element needs.
#
# The bench integrates the enriched element with a rule exact to degree 6,
# because the quartic interior bubble has a cubic gradient and the linear
# stability measurements must not be softened by under-integration.  For the
# element itself the question is different: a rule is admissible when it
# leaves the condensed element stiffness with exactly six zero-energy modes
# and does not soften the remaining ones appreciably, and the rule that does
# so with the fewest points sets the number of constitutive evaluations per element.
#
# Two measures on one element, for each rule.  The number of zero modes of
# K_e = K_dev + kappa G' M^-1 G (must be 6) and of K_dev (10 for an exact
# rule: six rigid modes and the four conformal ones, dilatation and the three
# special conformal fields, which are quadratic and therefore in P2).  And the
# softening: the extreme generalized eigenvalues of K_e(rule) against
# K_e(exact) on the complement of the rigid modes, which are 1 for an exact
# rule; the minimum is the factor by which the softest under-integrated
# direction is under-stiffened.  Then the two assembled results that matter,
# beta_h and the spurious-mode count under the uniaxial plastic tangent, are
# repeated with the reduced rules.
#
#   julia --project=. research/hw3l/prototype/quadrature.jl

include("element.jl")

const RULES = ((:rfe2, "RFE deg 2"), (:rfe3, "RFE deg 3"), (3, "conical deg 3"),
               (:keast14, "Keast deg 5"), (5, "conical deg 5"), (7, "conical deg 7"),
               (9, "conical deg 9"))

"Extreme generalized eigenvalues of (Ka, Kb) on the range of Kb."
function softening(Ka, Kb)
    F = eigen(Symmetric(Kb))
    tol = 1e-10 * maximum(F.values)
    Q = F.vectors[:, F.values .> tol]
    ev = eigvals(Symmetric(Q' * Ka * Q), Symmetric(Q' * Kb * Q))
    return minimum(ev), maximum(ev)
end

function element_table(X, label)
    println(label)
    ref = element_ops(X, tet_rule(11))
    println("  ", rpad("rule", 16), rpad("points", 8), rpad("zero(K_e)", 11),
            rpad("zero(K_dev)", 13), rpad("soft min", 10), rpad("soft max", 10),
            rpad("dev min", 10), "dev max")
    for (r, name) in RULES
        qp = rule(r)
        e = element_ops(X, qp)
        smin, smax = softening(e.K, ref.K)
        dmin, dmax = softening(e.Kdev, ref.Kdev)
        println("  ", rpad(name, 16), rpad(string(length(qp[2])), 8),
                rpad(string(nzero(e.K)), 11), rpad(string(nzero(e.Kdev)), 13),
                rpad(string(round(smin, sigdigits = 4)), 10),
                rpad(string(round(smax, sigdigits = 4)), 10),
                rpad(string(round(dmin, sigdigits = 4)), 10),
                string(round(dmax, sigdigits = 4)))
    end
    println()
end

function assembled_checks()
    println("Assembled, Crouzeix-Raviart pair, whole boundary fixed: beta_h and the")
    println("spurious count under the uniaxial plastic tangent (r_v > 0.1 among the")
    println("twenty lowest, N = 4) with reduced rules.")
    include_beta = isdefined(Main, :beta_from)
    n_ax = let n = zeros(3, 3); n[1, 1] = n[2, 2] = -1 / sqrt(6); n[3, 3] = 2 / sqrt(6); n end
    println("  ", rpad("rule", 16), rpad("beta N=2", 10), rpad("N=3", 10), rpad("N=4", 10),
            rpad("N=5", 10), rpad("null", 6), "spurious/20 at N=4")
    for (deg, name) in ((3, "conical deg 3"), (:keast14, "Keast deg 5"), (5, "conical deg 5"), (7, "conical deg 7"))
        cells = String[]; nulls = Int[]
        for N in 2:5
            coords, conn = mesh_of(N, 2)
            a = assemble_all(coords, conn, 2, 4; bubble = :full, q_degree = deg, allow_reduced = true)
            F  = cholesky(Symmetric(a.Kh1)); Gt = Matrix(a.G')
            Z  = Matrix(F.L \ Gt[F.p, :]); W = Matrix(Z / cholesky(Symmetric(Matrix(a.M))).U)
            sv = sort(svdvals(W)); r = rank(Matrix(a.G); rtol = 1e-10)
            push!(cells, rpad(string(round(sv[length(sv) - r + 1], sigdigits = 4)), 10))
            push!(nulls, a.np - r)
        end
        coords, conn = mesh_of(4, 2)
        ae = assemble_all(coords, conn, 2, 4; bubble = :full, q_degree = deg, allow_reduced = true)
        ap = assemble_all(coords, conn, 2, 4; bubble = :full, q_degree = deg, allow_reduced = true,
                          flow = n_ax, beta = 1.0)
        rv = nothing
        Zk = nullspace(Matrix(ae.G); rtol = 1e-10)
        Fe = eigen(Symmetric(Zk' * Matrix(ap.Kdev ./ 2) * Zk), Symmetric(Zk' * Matrix(ae.Kh1) * Zk))
        V = Zk * Fe.vectors[:, 1:20]
        rv = [dot(V[:, i], ae.Kdiv * V[:, i]) / dot(V[:, i], ae.Kh1 * V[:, i]) for i in 1:20]
        println("  ", rpad(name, 16), join(cells), rpad(join(unique(nulls), ","), 6),
                count(>(0.1), rv), "  (r_v of the lowest: ", round(rv[1], sigdigits = 3), ")")
        flush(stdout)
    end
end

function main()
    println("Quadrature for the enriched element: zero-energy modes and softening")
    println("relative to a degree-11 rule (216 points), kappa/mu = 1e4 for K_e.")
    println("Keast 14-point rule: worst monomial error through degree 5 = ",
            monomial_error(keast14(), 5), ", at degree 6 = ", monomial_error(keast14(), 6), "\n")
    for (X, label) in ((tet10_coords(), "reference tetrahedron"),
                       (tet10_coords(distort = 0.06), "distorted midsides (0.06, det J from 0.55 to 1.19)"),
                       (kuhn_coords(), "Kuhn tetrahedron of the Freudenthal mesh"))
        element_table(X, label)
    end
    assembled_checks()
end

main()
