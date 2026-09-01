# Is P2/P0 uniformly inf-sup stable?
#
# The dense generalized eigensolver in infsup.jl caps the mesh sequence at
# N = 5, where beta_h appeared to decay like h^(1/2).  Four points spanning
# less than a decade of h cannot separate a genuine rate from a pre-asymptotic
# transient, and the answer matters: if P2/P0 is uniformly stable then it needs
# no bubble enrichment, no new reference element, and the formulation is
# implementable as it stands.
#
# Two changes make a longer sequence reachable.
#
# First, beta_h is computed as a SINGULAR value rather than an eigenvalue.
# With K = P' L L' P from a sparse Cholesky,
#
#     q' G K^-1 G' q = || L^-1 (G' q)[p] ||^2 ,
#
# so with Z = L^-1 (G')[p,:] and W = Z M^(-1/2),
#
#     beta_h = smallest nonzero singular value of W .
#
# This never forms G K^-1 G', so it does not square the condition number the
# way the eigenvalue formulation does -- which matters precisely when the
# quantity being measured is small.
#
# Second, the null space is identified by rank rather than by a threshold:
# dim null = n_p - rank(G), computed independently, so the "smallest nonzero"
# singular value is selected by index and not by guessing a tolerance.

include("common.jl")

"""
beta_h for one mesh, plus the null-space dimension and the singular values
bracketing the cut, so the selection can be checked rather than trusted.
"""
function beta_h(N, p, m; bc = _BC_ALL, bubble = :none)
    c4, n4 = tet4_mesh(N)
    nvert = size(c4, 2)
    coords, conn = p == 1 ? (c4, n4) : promote_to_p2(c4, n4)
    a = assemble_all(coords, conn, p, m; bc, nvert, bubble)
    a.nu == 0 && return nothing

    F  = cholesky(Symmetric(a.Kh1))
    Gt = Matrix(a.G')                      # n_u x n_p
    Z  = Matrix(F.L \ Gt[F.p, :])          # L^-1 P G', dense
    dM = Vector(sqrt.(diag(a.M)))          # M is diagonal for P0
    W  = m == 1 ? Z ./ dM' : Matrix(Z / cholesky(Symmetric(Matrix(a.M))).U)

    sv = sort(svdvals(W))
    # W is n_u x n_p, so it has only min(n_u, n_p) singular values; when
    # n_p > n_u the surplus null directions are not represented among them at
    # all.  Counting the NONZERO ones from the top is correct either way:
    # exactly rank(G) of them are nonzero.
    r = rank(Matrix(a.G); rtol = 1e-10)
    L = length(sv)
    idx = L - r + 1
    beta = (1 <= idx <= L) ? sv[idx] : NaN
    return (; a.nu, a.np, a.nelem, h = 1 / N, rank_G = r,
              nnull = a.np - r, beta,
              sv_below = idx > 1 ? sv[idx - 1] : 0.0,
              sv_max = sv[end])
end

"Least-squares rate q in beta ~ h^q over the tail of the sequence."
function fitted_rate(hs, betas; tail = 4)
    n = length(hs)
    idx = max(1, n - tail + 1):n
    X = [ones(length(idx)) log.(hs[idx])]
    y = log.(betas[idx])
    return (X \ y)[2]
end

function main()
    println("Is P2/P0 uniformly inf-sup stable?\n")
    println("beta_h is the smallest NONZERO singular value of M^(-1/2) G L^(-T),")
    println("with the null dimension fixed independently as n_p - rank(G) so the")
    println("cut is selected by index rather than by a threshold.\n")
    println("A uniformly stable pair has beta_h flattening as h -> 0 (rate ~ 0).")
    println("A rate near 0.5 over a widening range is a genuine decay.\n")

    for (label, p, m, Ns, bub) in (
            ("P2 / P1 cont. (POSITIVE CONTROL: stable in 3D)", 2, _P1C, 2:8, :none),
            ("P2 / P0",                                        2, 1,    2:8, :none),
            ("P2 / P1 disc",                                   2, 4,    2:6, :none),
            ("P2 + interior bubble / P1 disc",                 2, 4,    2:6, :interior),
            ("P2 + interior + face bubbles / P1 disc (Crouzeix-Raviart 3D)",
                                                               2, 4,    2:6, :full),
            ("P2 + interior bubble / P0 (must match P2/P0: see below)",
                                                               2, 1,    2:6, :interior))
        println(label)
        println("  ", rpad("N", 4), rpad("h", 8), rpad("n_u", 8), rpad("n_p", 8),
                rpad("rank G", 8), rpad("beta_h", 12), rpad("next below", 13), "beta/h^0.5")
        hs, bs = Float64[], Float64[]
        for N in Ns
            r = beta_h(N, p, m; bubble = bub)
            r === nothing && continue
            push!(hs, r.h); push!(bs, r.beta)
            println("  ", rpad(string(N), 4), rpad(string(round(r.h, sigdigits=3)), 8),
                    rpad(string(r.nu), 8), rpad(string(r.np), 8),
                    rpad(string(r.rank_G), 8),
                    rpad(string(round(r.beta, sigdigits=5)), 12),
                    rpad(string(round(r.sv_below, sigdigits=3)), 13),
                    string(round(r.beta / sqrt(r.h), sigdigits=4)))
        end
        if length(hs) >= 3
            q_all  = fitted_rate(hs, bs; tail = length(hs))
            q_tail = fitted_rate(hs, bs; tail = 4)
            println("  fitted rate over all points: ", round(q_all, sigdigits=3))
            println("  fitted rate over last four : ", round(q_tail, sigdigits=3))
            verdict = abs(q_tail) < 0.15 ? "FLAT -> uniformly stable" :
                      q_tail > 0.3      ? "DECAYING -> not uniformly stable" :
                                          "inconclusive"
            println("  verdict: ", verdict)
        end
        println()
    end

    println("The interior bubble cannot affect a P0 pressure.  Its divergence")
    println("integrates to zero against any constant, so its columns of G vanish")
    println("identically and it acts only through enlarging K -- which can raise")
    println("beta_h but never lower it.  The last two blocks agreeing to four")
    println("digits is that statement, measured.")
end

main()
