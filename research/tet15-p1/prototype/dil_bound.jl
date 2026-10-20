# Referee round 2, M13: a criterion for dilatational modes that does not depend
# on how many eigenpairs are computed.
#
# On ker G, with a(v,v) the energy of the continuum plastic tangent (beta = 1)
# and |v|^2 = int |grad v|^2, the Rayleigh quotient is lam(v) = a(v,v)/|v|^2
# and the dilatation measure is r_v(v) = int (div v)^2 / |v|^2.  The quantity
# asked for is
#
#     lam*(r0) = inf { lam(v) : v in ker G, r_v(v) >= r0 },   r0 = 0.1,
#
# the smallest energy of any member with at least that dilatation, eigenmode
# or not.  For s >= 0 let g(s) be the smallest eigenvalue of a - s (div,div)
# on ker G against |.|^2.  Every v with r_v(v) >= r0 satisfies
# lam(v) = [a(v,v) - s int (div v)^2]/|v|^2 + s r_v(v) >= g(s) + s r0, so
#
#     L(r0) = max over s of g(s) + s r0  <=  lam*(r0)          (weak duality).
#
# The minimizer of g(s), when its r_v >= r0, is a member with that
# dilatation, so its lam is an upper bound U(r0) >= lam*(r0).  g(s) is
# computed by the penalized subspace iteration of kernel_modes, with a shift
# sigma |.|^2 added until the penalized matrix is positive definite.
#
#   julia --project=. research/tet15-p1/prototype/dil_bound.jl [pair index]

ENV["CC_NOMAIN"] = "1"
include("cubit_cube.jl")            # cubit_mesh, quality, elem_div2, plastic_ext, kappa_sweep
using Random, Printf

function perturbed_mesh(N, delta, seed)
    coords, conn = mesh_of(N, 2)
    rng = MersenneTwister(seed)
    X = copy(coords); h = 1 / N
    for i in 1:(N + 1)^3
        x = coords[:, i]
        any(t -> t < 1e-12 || t > 1 - 1e-12, x) && continue
        X[:, i] = x .+ delta * h .* (2 .* rand(rng, 3) .- 1)
    end
    for e in axes(conn, 2), (k, (a, b)) in enumerate(_TET_EDGES)
        X[:, conn[4 + k, e]] = 0.5 .* (X[:, conn[a, e]] .+ X[:, conn[b, e]])
    end
    return X, conn
end

function lowest_weighted(a, K, s; b = 12, gamma = 1e9, tol = 1e-7, maxit = 5000)
    D = Diagonal(1 ./ Vector(diag(a.M)))
    P = gamma * (a.G' * D * a.G)
    Kh = a.Kh1
    sigma = 0.0; F = nothing; A = nothing
    for sg in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0) .* s
        A = sparse(Symmetric(K - s * a.Kdiv + sg * Kh + P))
        F = cholesky(Symmetric(A); check = false)
        issuccess(F) && (sigma = sg; break)
        F = nothing
    end
    F === nothing && error("no positive definite shift")
    X = randn(MersenneTwister(3), a.nu, b)
    lam = Inf
    for it in 1:maxit
        Y = F \ (Kh * X)
        E = eigen(Symmetric(Y' * (A * Y)), Symmetric(Y' * (Kh * Y)))
        X = Y * E.vectors
        X ./= sqrt.(sum(X .* (Kh * X), dims = 1))
        new = E.values[1] - sigma
        abs(new - lam) < tol * max(abs(new), 1e-3) && return new, X[:, 1], it, sigma
        lam = new
    end
    @warn "lowest_weighted: not converged"
    return lam, X[:, 1], maxit, sigma
end

const BPAIRS = (("P2+int+face/P1disc", 4, :full), ("P2/P0", 1, :none), ("P2+face/P0", 1, :face))

const R0 = 0.1

# g is concave in s with derivative -r_v(v_s), so r_v(v_s) increases with s and
# the best s is where it crosses r0; there L = U up to the bisection width.
function row(label, mesh, X, conn, m, bubble, flow; nbis = 14)
    ap = assemble_all(X, conn, 2, m; bubble, flow, beta = 1.0)
    K = ap.Kdev ./ 2
    qmin = minimum(quality(X, conn, e) for e in axes(conn, 2))
    L = -Inf; U = Inf; vU = nothing; sU = NaN; rU = NaN
    ntot = 0
    t = @elapsed begin
        function probe(s)
            g, v, it, _ = lowest_weighted(ap, K, s); ntot += it
            lam = dot(v, K * v) / dot(v, ap.Kh1 * v); rv = dilatation(ap, v)
            L = max(L, g + s * R0)
            if rv >= R0 && lam < U
                U = lam; vU = v; sU = s; rU = rv
            end
            return rv
        end
        lo, hi = 0.0, 0.25
        if probe(1e-9) >= R0
            hi = 0.0                             # the lowest mode is already dilatational
        else
            while probe(hi) < R0; lo = hi; hi *= 2; end
            for _ in 1:nbis
                mid = (lo + hi) / 2
                probe(mid) < R0 ? (lo = mid) : (hi = mid)
            end
        end
    end
    ed = elem_div2(X, conn, bubble, ap.nu_nodal, vU)
    p = sortperm(ed, rev = true)[1:3]
    @printf("%-20s %-14s %-7d qmin %-6.3f  L %-8.4g U %-8.4g s* %-8.4g r_v(U) %-6.3f top3 %-5.2f q3 %-6.3f (%d it, %d s)\n",
            label, mesh, ap.nu, qmin, L, U, sU, rU, sum(ed[p]) / sum(ed),
            minimum(quality(X, conn, e) for e in p), ntot, round(Int, t))
    flush(stdout)
end

function main(pairs = BPAIRS; flows = (("axial", N_AXIAL),))
    for (fname, flow) in flows
        println("\nflow $fname, r0 = $R0: L = max_s g(s) + s r0 is a lower bound and U the lam of a member ",
                "with r_v >= r0 (an upper bound) on the smallest lam among members of ker G with r_v >= r0; ",
                "top3 = share of int(div v)^2 of that member in its 3 largest elements, q3 = min quality of those")
        for (label, m, bubble) in pairs
            for N in (4, 6, 8)
                X, conn = mesh_of(N, 2); row(label, "Freud N=$N", X, conn, m, bubble, flow)
            end
            for delta in (0.1, 0.25), N in (6, 7)
                X, conn = perturbed_mesh(N, delta, 1); row(label, "d=$delta N=$N", X, conn, m, bubble, flow)
            end
            for n in (3, 4, 5, 6, 7, 8, 10)
                X, conn = cubit_mesh(n); row(label, "Cubit n=$n", X, conn, m, bubble, flow)
            end
        end
    end
end

if !isinteractive() && get(ENV, "DB_NOMAIN", "") != "1"
    main(isempty(ARGS) ? BPAIRS : BPAIRS[[parse(Int, ARGS[1])]])
end
