# Referee round 2, M14: the lowest eigenpairs of the assembled Carina tangent of
# the punch problem (TET15-P1, case B, h = a/4) at the last state whose tangent
# has a Cholesky factor and at the first state where the factorization fails.
# The tangents are written by Carina's direct solver with CARINA_DUMP_TANGENT
# set (free DOFs, N/m).  For each mode v (Euclidean norm on the free DOFs):
# the eigenvalue, the dilatation measure r_v = int (div v)^2 / int |grad v|^2
# in the reference configuration, the share of int (div v)^2 and of
# int |grad v|^2 in the three largest elements, and the position (r, z) of the
# node of largest displacement, in units of a.
#
#   julia --project=. research/tet15-p1/prototype/tangent_modes.jl <dump dir> <tetra15 mesh .g>
include("common.jl")
using Exodus, Random, Printf, SparseArrays

function read_dump(f)
    open(f) do io
        n, nz, nf, nu = (read(io, Int64) for _ in 1:4)
        colptr = Vector{Int64}(undef, n + 1); read!(io, colptr)
        rowval = Vector{Int64}(undef, nz); read!(io, rowval)
        nzval = Vector{Float64}(undef, nz); read!(io, nzval)
        free = Vector{Int64}(undef, nf); read!(io, free)
        U = Vector{Float64}(undef, nu); read!(io, U)
        return SparseMatrixCSC(n, n, colptr, rowval, nzval), free, U
    end
end

"Eigenpairs of K nearest sigma by shift-invert subspace iteration."
function nearest_modes(K, sigma; k = 6, b = 16, tol = 1e-8, maxit = 500)
    n = size(K, 1)
    F = lu(K - sigma * I)
    X = Matrix(qr(randn(MersenneTwister(4), n, b)).Q)
    lam = fill(Inf, k)
    for it in 1:maxit
        Y = F \ X
        Q = Matrix(qr(Y).Q)
        E = eigen(Symmetric(Q' * (K * Q)))
        o = sortperm(abs.(E.values .- sigma))
        X = Q * E.vectors[:, o]
        new = E.values[o[1:k]]
        if maximum(abs.(new .- lam)) < tol * maximum(abs.(new))
            return new, X[:, 1:k], it
        end
        lam = new
    end
    @warn "nearest_modes: not converged"
    return lam, X[:, 1:k], maxit
end

"Element nodes reordered to enriched_nodes: 10 TETRA10 nodes, centroid, face nodes opposite vertex 1..4."
function reorder(X, c)
    v = X[:, c[1:4]]
    cen = sum(v, dims = 2) ./ 4
    extra = c[11:15]
    ic = argmin([sum(abs2, X[:, m] .- cen) for m in extra])
    out = vcat(c[1:10], extra[ic])
    for f in 1:4
        fc = sum(v[:, collect(_TET_FACES[f])], dims = 2) ./ 3
        push!(out, extra[argmin([sum(abs2, X[:, m] .- fc) for m in extra])])
    end
    @assert length(unique(out)) == 15
    return out
end

function element_measures(X, conn, vfull)
    A = nodal_transform(); el = ref_element(2); qp, qw = tet_rule(6)
    ne = size(conn, 2); d2 = zeros(ne); g2 = zeros(ne)
    for e in 1:ne
        c = reorder(X, conn[:, e]); Xe = X[:, c[1:10]]
        ve = reshape(vfull[vec([3 * (m - 1) + d for d in 1:3, m in c])], 3, 15)
        for q in eachindex(qw)
            xi = qp[:, q]
            Jm = Xe * RFE.shape_function_gradient(el, xi)
            _, dNh = enriched_shape(xi)
            dN = (A * dNh) / Jm                 # 15 x 3, physical gradients of the nodal basis
            Gv = ve * dN
            w = qw[q] * det(Jm)
            d2[e] += w * tr(Gv)^2; g2[e] += w * sum(abs2, Gv)
        end
    end
    return d2, g2
end

function main(dir, meshfile)
    exo = ExodusDatabase(meshfile, "r")
    X = Matrix{Float64}(read_coordinates(exo)); conn = Matrix{Int}(read_sets(exo, Block)[1].conn)
    close(exo)
    a = 1.0e-3
    for name in ("last_pd.bin", "first_indef.bin")
        f = joinpath(dir, name); isfile(f) || continue
        K, free, U = read_dump(f)
        @assert length(U) == 3 * size(X, 2) || length(U) == length(free)
        Ks = (K + K') / 2
        @printf("\n%s: %d free DOFs, ||K - K'||/||K|| = %.2e\n", name, size(K, 1), norm(K - K') / norm(K))
        # scale: the largest diagonal entry; eigenvalues are reported over it
        dmax = maximum(abs, diag(Ks))
        lam, V, it = nearest_modes(Ks, 0.0)
        @printf("  modes nearest 0 (%d it); lam / max diag K:\n", it)
        for i in eachindex(lam)
            vfull = zeros(3 * size(X, 2)); vfull[free] = V[:, i]
            d2, g2 = element_measures(X, conn, vfull)
            p = sortperm(d2, rev = true)[1:3]; pg = sortperm(g2, rev = true)[1:3]
            amp = [norm(vfull[3m-2:3m]) for m in 1:size(X, 2)]; m = argmax(amp)
            @printf("  lam %+.3e  r_v %.3g  div share top3 %.2f  grad share top3 %.2f  max node at r = %.3f a, z = %.3f a\n",
                    lam[i] / dmax, sum(d2) / sum(g2), sum(d2[p]) / sum(d2), sum(g2[pg]) / sum(g2),
                    hypot(X[1, m], X[2, m]) / a, X[3, m] / a)
        end
    end
end

get(ENV, "TM_NOMAIN", "") == "1" || main(ARGS[1], ARGS[2])

# The most negative eigenvalue: bisection on the success of the Cholesky
# factorization of K + s dmax I gives lam_min / dmax in [-hi, -lo]; the mode is
# then the eigenpair nearest -hi dmax by shift-invert.
function negative_mode(dir, meshfile; name = "first_indef.bin")
    exo = ExodusDatabase(meshfile, "r")
    X = Matrix{Float64}(read_coordinates(exo)); conn = Matrix{Int}(read_sets(exo, Block)[1].conn)
    close(exo)
    K, free, U = read_dump(joinpath(dir, name)); Ks = sparse(Symmetric((K + K') / 2))
    dmax = maximum(abs, diag(Ks))
    ok(s) = issuccess(cholesky(Symmetric(Ks + s * dmax * I); check = false))
    lo, hi = 0.0, 1e-8
    while !ok(hi); lo = hi; hi *= 10; end
    for _ in 1:30; mid = sqrt(max(lo, hi * 1e-6) * hi); ok(mid) ? (hi = mid) : (lo = mid); end
    @printf("\n%s: lam_min / dmax in [-%.4e, -%.4e]\n", name, hi, lo)
    lam, V, it = nearest_modes(Ks, -hi * dmax; k = 3)
    a = 1.0e-3
    for i in eachindex(lam)
        vfull = zeros(3 * size(X, 2)); vfull[free] = V[:, i]
        d2, g2 = element_measures(X, conn, vfull)
        p = sortperm(d2, rev = true)[1:3]; pg = sortperm(g2, rev = true)[1:3]
        amp = [norm(vfull[3m-2:3m]) for m in 1:size(X, 2)]; m = argmax(amp)
        @printf("  lam %+.4e  r_v %.3g  div share top3 %.2f  grad share top3 %.2f  max node at r = %.3f a, z = %.3f a; elements %s\n",
                lam[i] / dmax, sum(d2) / sum(g2), sum(d2[p]) / sum(d2), sum(g2[pg]) / sum(g2),
                hypot(X[1, m], X[2, m]) / a, X[3, m] / a, string(pg))
    end
end
get(ENV, "TM_NEG", "") == "1" && negative_mode(ARGS[1], ARGS[2])
