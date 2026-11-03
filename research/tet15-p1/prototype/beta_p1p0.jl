# The negative control of the inf-sup measurement: the pair P1/P0, continuous
# P1 displacements and one constant pressure per element, known unstable.
#
# With the whole boundary fixed the pair has more pressure than displacement
# unknowns, n_p - n_u = 6N^3 - 3(N-1)^3, so the null dimension n_p - rank(G)
# must be at least that.  This script records what the procedure of beta78.jl
# (singular values of L^-1 P G' M^-1/2, null dimension by rank) reports for
# the pair: the null dimension, and beta_h on the pressure modes the
# displacement does control.  The paper quotes the values at N = 2 to 5.

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


function main()
    println("P1/P0 (continuous P1 displacement, constant pressure per element), whole boundary fixed")
    println("  ", rpad("N", 4), rpad("h", 8), rpad("n_u", 8), rpad("n_p", 8),
            rpad("rank G", 8), rpad("null", 7), rpad("n_p - n_u", 11), rpad("beta_h", 12),
            rpad("next below", 13), "local rate")
    prev = nothing
    for N in 2:10
        r = beta_h(N, 1, 1)
        r === nothing && continue
        q = prev === nothing ? "" :
            string(round(log(prev[2] / r.beta) / log(N / prev[1]), sigdigits = 3))
        prev = (N, r.beta)
        println("  ", rpad(string(N), 4), rpad(string(round(r.h, sigdigits=3)), 8),
                rpad(string(r.nu), 8), rpad(string(r.np), 8),
                rpad(string(r.rank_G), 8), rpad(string(r.nnull), 7),
                rpad(string(r.np - r.nu), 11),
                rpad(string(round(r.beta, sigdigits=5)), 12),
                rpad(string(round(r.sv_below, sigdigits=3)), 13), q)
        flush(stdout)
    end
end

main()
