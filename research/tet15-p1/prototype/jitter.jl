# The inf-sup constant on meshes whose interior vertices are moved at random
# (referee request: beta_h on meshes without the structure of the Freudenthal
# family).  Each interior vertex of the Freudenthal mesh is displaced by a
# vector with independent components uniform in [-delta h, delta h]; boundary
# vertices stay; edge nodes are reset to the midpoints, so every element is a
# straight-sided (affine) tetrahedron and the P1 pressure in reference
# coordinates is P1 in physical coordinates.  Three seeds per mesh.
#
#   julia --project=. research/tet15-p1/prototype/jitter.jl

include("common.jl")
using Random

# beta_from as in deformed.jl
function beta_from(a, m)
    F  = cholesky(Symmetric(a.Kh1))
    Gt = Matrix(a.G')
    Z  = Matrix(F.L \ Gt[F.p, :])
    W  = m == 1 ? Z ./ Vector(sqrt.(diag(a.M)))' :
                  Matrix(Z / cholesky(Symmetric(Matrix(a.M))).U)
    sv = sort(svdvals(W))
    r  = rank(Matrix(a.G); rtol = 1e-10)
    idx = length(sv) - r + 1
    return (1 <= idx <= length(sv)) ? sv[idx] : NaN, a.np - r
end

function jittered(N, delta, seed)
    coords, conn = mesh_of(N, 2)
    rng = MersenneTwister(seed)
    nv = (N + 1)^3
    h = 1 / N
    X = copy(coords)
    for i in 1:nv
        x = coords[:, i]
        any(t -> t < 1e-12 || t > 1 - 1e-12, x) && continue
        X[:, i] = x .+ delta * h .* (2 .* rand(rng, 3) .- 1)
    end
    for e in axes(conn, 2), (k, (a, b)) in enumerate(_TET_EDGES)
        X[:, conn[4 + k, e]] = 0.5 .* (X[:, conn[a, e]] .+ X[:, conn[b, e]])
    end
    # shape quality: minimum over elements of 6 sqrt(2) V / l_rms^3 (1 for regular)
    qmin = Inf
    for e in axes(conn, 2)
        P = X[:, conn[1:4, e]]
        V = det([P[:, 2] - P[:, 1]  P[:, 3] - P[:, 1]  P[:, 4] - P[:, 1]]) / 6
        V <= 0 && return X, conn, -1.0
        l2 = sum(sum(abs2, P[:, a] - P[:, b]) for (a, b) in _TET_EDGES) / 6
        qmin = min(qmin, 6 * sqrt(2) * V / l2^1.5)
    end
    return X, conn, qmin
end

const PAIRS = (("P2/P1c (TH)",        _P1C, :none, 2:6),
               ("P2/P0",              1,    :none, 2:6),
               ("P2+face/P0",         1,    :face, 2:6),
               ("P2+int+face/P1disc", 4,    :full, 2:6))

function main(; delta = 0.25, seeds = (1, 2, 3))
    println("beta_h on Freudenthal meshes with interior vertices perturbed by up to")
    println("delta h = $delta h per component; straight-sided elements; whole boundary")
    println("fixed.  Entries: beta_h for seeds $(seeds); q_min is the smallest")
    println("element quality 6 sqrt2 V / l_rms^3 over the three seeds (1 = regular,")
    println("Freudenthal elements 0.66); null = n_p - rank G.\n")
    for (label, m, bubble, Ns) in PAIRS
        println(label)
        for N in Ns
            bs = Float64[]; qs = Float64[]; nn = Int[]
            for s in seeds
                X, conn, q = jittered(N, delta, s)
                push!(qs, q)
                q <= 0 && (push!(bs, NaN); continue)
                nvert = m == _P1C ? (N + 1)^3 : 0
                coords0, _ = mesh_of(N, 2)
                a = assemble_all(X, conn, 2, m; bubble, nvert, bc_coords = coords0)
                b, n0 = beta_from(a, m)
                push!(bs, b); push!(nn, n0)
            end
            println("  N=", rpad(N, 3), " q_min ", rpad(round(minimum(qs), sigdigits = 3), 7),
                    " beta_h ", join((rpad(round(b, sigdigits = 4), 9) for b in bs)),
                    " null ", join(unique(nn), ","))
            flush(stdout)
        end
        println()
    end
end

main()
