# Correctness checks on the pieces of common.jl that the enriched element
# depends on.  Each prints a number whose right value is known in advance, so
# the check is against a target and not against a previous run.
#
#   julia --project=<Carina> checks.jl

include("common.jl")

fact(n) = prod(1:big(n); init = big(1))
exact(a, b, c) = Float64(fact(a) * fact(b) * fact(c) // fact(a + b + c + 3))

println("1. Quadrature.  int_T x^a y^b z^c = a! b! c! / (a+b+c+3)!.  A rule built")
println("   for degree d must be exact through d, and the degree-6 rule must")
println("   NOT be exact at degree 8, or the test is not testing anything.\n")
for deg in (2, 3, 6, 7)
    pts, wts = tet_rule(deg)
    worst = 0.0
    for a in 0:deg, b in 0:deg, c in 0:deg
        a + b + c > deg && continue
        num = sum(wts[q] * pts[1, q]^a * pts[2, q]^b * pts[3, q]^c for q in axes(pts, 2))
        worst = max(worst, abs(num - exact(a, b, c)) / exact(a, b, c))
    end
    println("   degree ", deg, ": ", length(wts), " points, sum w = ", sum(wts),
            ", worst relative error ", round(worst, sigdigits = 2))
end
pts, wts = tet_rule(6)
println("   degree-6 rule on x^8: relative error ",
        round(abs(sum(wts[q] * pts[1, q]^8 for q in axes(pts, 2)) - exact(8, 0, 0)) / exact(8, 0, 0), sigdigits = 2),
        "  (must be O(1e-3), not zero)")

println("\n2. Interior bubble.  It vanishes on the element boundary, so int div b = 0")
println("   and its columns of G against a CONSTANT pressure must be exactly zero,")
println("   while against the linear P1disc modes they must not be.\n")
c4, n4 = tet4_mesh(2); nvert = size(c4, 2)
coords, conn = promote_to_p2(c4, n4)
a = assemble_all(coords, conn, 2, 1; bubble = :interior, nvert)
println("   P0 x interior-bubble block: max |entry| = ",
        round(maximum(abs, a.G[:, a.nu_nodal+1:end]), sigdigits = 2))
a = assemble_all(coords, conn, 2, 4; bubble = :interior, nvert)
G4 = Matrix(a.G[:, a.nu_nodal+1:end])
cr = 1:4:size(G4, 1)
println("   P1disc constant rows: max |entry| = ", round(maximum(abs, G4[cr, :]), sigdigits = 2))
println("   P1disc linear rows:   max |entry| = ", round(maximum(abs, G4[setdiff(1:size(G4, 1), cr), :]), sigdigits = 2))

println("\n3. Conformity.  For every free basis function, sum_m G[m,i] = int div N_i")
println("   = oint N_i . n = 0 when the space is H1-conforming and N_i has zero")
println("   trace on the fixed boundary.  A face bubble double counted, mismatched")
println("   between its two elements, or left free on a boundary face breaks this.\n")
for N in (2, 3), enr in (:none, :interior, :face, :full)
    local cc, nn = tet4_mesh(N)
    local xy, cn = promote_to_p2(cc, nn)
    local b = assemble_all(xy, cn, 2, 1; bubble = enr, nvert = size(cc, 2))
    println("   N=", N, " ", rpad(string(enr), 9), " n_u=", rpad(string(b.nu), 6),
            " max |int div N_i| = ", round(maximum(abs, vec(sum(Matrix(b.G), dims = 1))), sigdigits = 2))
end

println("\n4. Face bookkeeping on N=3: 162 tets, 648 element faces, (648+108)/2 = 378")
println("   distinct, 108 on the boundary; every interior face seen twice.\n")
c4, n4 = tet4_mesh(3); coords, conn = promote_to_p2(c4, n4)
nf_all, _ = face_table(coords, conn, _BC_ALL)
nf_zf, fmap = face_table(coords, conn, _BC_ZFACE)
mult = zeros(Int, nf_zf)
for e in axes(conn, 2), f in 1:4
    fmap[f, e] == 0 || (mult[fmap[f, e]] += 1)
end
println("   free faces, all-Dirichlet: ", nf_all, "  (expect 378 - 108 = 270)")
println("   free faces, z-face only:   ", nf_zf, "  (expect 378 - 18 = 360)")
println("   multiplicities present:    ", sort(unique(mult)), "  (expect [1, 2])")

println("\n5. Refusals.  Under-integration and P1 geometry must error, not proceed.\n")
for (kw, what) in (((; bubble = :interior, q_degree = 3), "degree-3 rule with a bubble"),
                   ((; bubble = :bogus,), "unknown enrichment"))
    try
        assemble_all(coords, conn, 2, 1; nvert, kw...)
        println("   FAIL: ", what, " was accepted")
    catch err
        println("   refused: ", what)
    end
end
