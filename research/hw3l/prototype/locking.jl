# Assembled soft-mode and locking sweep.
#
# softmode.jl counted spurious modes on ONE element.  Those are element-local
# statements, and an element-local mode only survives assembly if its trace is
# compatible with its neighbors across every shared face, so the global count
# could be far smaller -- possibly zero.  This measures it on assembled meshes.
#
# The same assembly answers the locking question, and again by counting rather
# than by eigenvalues.  With
#
#     K(kappa) = Kdev + kappa * G' * inv(M) * G,
#
# a displacement mode costs volumetric energy exactly when it is outside
# ker(G).  As kappa -> infinity the reachable deformations collapse onto that
# kernel, so
#
#     dim ker(G) = n_u - rank(G)
#
# is the dimension of the discretely isochoric subspace: the deformations the
# element can still represent when the material becomes incompressible.  If
# that number is a healthy fraction of n_u the element deforms freely; if it
# collapses toward zero the element locks, and no eigenvalue is needed to see
# it.
#
# Both failure modes are therefore rank statements about the SAME matrix G:
#
#     rank(G) too small  ->  volumetric directions unconstrained -> soft modes
#     rank(G) too large  ->  isochoric subspace destroyed        -> locking

include("common.jl")

"""
Assembled census for one pair, one mesh, one boundary condition.

`rank(G)` is the number of volumetric directions the pressure space constrains;
`n_u - rank(G)` is the dimension of the discretely isochoric subspace, the
deformations still available as kappa grows.  `lam_min` is normalized by the
deviatoric scale so it is comparable across pairs and meshes.
"""
function census(N, p, m; mu = 1.0, kappa = 1.0e4, bc = _BC_ALL)
    coords, conn = mesh_of(N, p)
    a = assemble_all(coords, conn, p, m; mu, bc)
    a.nu == 0 && return nothing
    Mf = cholesky(Symmetric(Matrix(a.M) + 1e-14I))
    Kvol = a.G' * (Mf \ Matrix(a.G))
    K = Matrix(a.Kdev) + kappa * Kvol
    ev = eigvals(Symmetric(K))
    devs = maximum(abs, diag(Matrix(a.Kdev)))
    rg = rank(Matrix(a.G); rtol = 1e-10)
    return (; a.nu, a.np, a.nelem, rank_G = rg,
              isochoric = a.nu - rg, iso_frac = (a.nu - rg) / a.nu,
              n_zero = count(e -> abs(e) < 1e-9 * devs, ev),
              lam_rel = minimum(ev) / devs)
end

function main()
    println("Assembled soft-mode and locking sweep, kappa/mu = 1e4.\n")
    println("  rank(G)   volumetric directions the pressure space constrains")
    println("  isochoric n_u - rank(G): deformations still available as kappa -> inf")
    println("            -> collapses toward zero = volumetric locking")
    println("  n_zero    assembled modes with no energy at all")
    println("  lam_min   smallest eigenvalue, relative to the deviatoric scale\n")
    println("Two boundary conditions.  `all` fixes the whole boundary, the standard")
    println("inf-sup setting and the most constraining a mesh can have.  `zface`")
    println("fixes only z = 0, which is far closer to the confined-but-not-clamped")
    println("conditions under which soft modes are reported.\n")

    hdr = (rpad("bc", 8), rpad("pair", 13), rpad("N", 4), rpad("n_u", 7),
           rpad("n_p", 7), rpad("rank G", 8), rpad("isochoric", 11),
           rpad("iso frac", 10), rpad("n_zero", 8), "lam_min/dev")
    println(hdr...)
    println("-"^96)
    for bc in (_BC_ALL, _BC_ZFACE)
        for (label, p, m) in (("P2/P0", 2, 1), ("P2/P1disc", 2, 4))
            for N in (3, 4)
                r = census(N, p, m; bc)
                r === nothing && continue
                println(rpad(string(bc), 8), rpad(label, 13), rpad(string(N), 4),
                        rpad(string(r.nu), 7), rpad(string(r.np), 7),
                        rpad(string(r.rank_G), 8), rpad(string(r.isochoric), 11),
                        rpad(string(round(r.iso_frac, sigdigits = 3)), 10),
                        rpad(string(r.n_zero), 8),
                        string(round(r.lam_rel, sigdigits = 3)))
            end
        end
    end

    println()
    println("Enriched pair, by the same count: one vector bubble per element adds")
    println("3*nelem displacement degrees of freedom without changing rank(G).")
    println(rpad("bc", 8), rpad("N", 4), rpad("n_u+3ne", 10), rpad("rank G", 8),
            rpad("isochoric", 11), "iso frac")
    for bc in (_BC_ALL, _BC_ZFACE), N in (3, 4)
        r = census(N, 2, 4; bc)
        r === nothing && continue
        nu2 = r.nu + 3 * r.nelem
        println(rpad(string(bc), 8), rpad(string(N), 4), rpad(string(nu2), 10),
                rpad(string(r.rank_G), 8), rpad(string(nu2 - r.rank_G), 11),
                string(round((nu2 - r.rank_G) / nu2, sigdigits = 3)))
    end
end

main()
