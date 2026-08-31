# Numerical inf-sup test for the displacement-pressure pair.
#
# The soft-mode census in softmode.jl is a single-element rank count and cannot
# see stability, which is a property of a mesh SEQUENCE.  This is the
# Chapelle-Bathe numerical inf-sup test: assemble
#
#     K[i,j] = int grad(N_i) : grad(N_j)      (H1 seminorm on displacement)
#     G[m,i] = int phi_m div(N_i)             (volumetric coupling)
#     M[m,n] = int phi_m phi_n                (pressure mass)
#
# with displacement fixed on the whole boundary, and solve
#
#     G K^-1 G' q = lambda M q .
#
# The discrete inf-sup constant is beta_h = sqrt(lambda_min), taken over the
# nonzero spectrum.  Two distinct failures show up here:
#
#   * beta_h decaying as h -> 0 : the pair locks in the limit;
#   * extra zero eigenvalues    : spurious pressure modes (checkerboard).
#
# All-Dirichlet boundary data always leaves exactly ONE zero mode, the
# hydrostatic one, because int div(v) = 0 for v in H^1_0.  Any zero beyond that
# is spurious.
#
# The meshes are generated here rather than read, so the refinement sequence is
# exact and needs no mesher.

using LinearAlgebra, SparseArrays
import ReferenceFiniteElements as RFE

const NSD = 3

# --------------------------------------------------------------------------
# Structured tetrahedral mesh of the unit cube
# --------------------------------------------------------------------------
# Freudenthal (Kuhn) subdivision: each cube splits into the six tetrahedra
# {origin, e_a, e_a + e_b, (1,1,1)} over the six permutations (a,b,c).  This
# decomposition is conforming across shared cube faces, which a naive 5-tet
# split is not.
const _KUHN = ((1,2,3), (1,3,2), (2,1,3), (2,3,1), (3,1,2), (3,2,1))

function tet4_mesh(N::Int)
    nn = N + 1
    nid(i, j, k) = i + nn * (j + nn * k) + 1        # 0-based i,j,k
    coords = zeros(NSD, nn^3)
    for k in 0:N, j in 0:N, i in 0:N
        coords[:, nid(i, j, k)] = [i, j, k] ./ N
    end
    conns = Vector{NTuple{4, Int}}()
    for k in 0:N-1, j in 0:N-1, i in 0:N-1
        corner(b) = nid(i + b[1], j + b[2], k + b[3])
        for p in _KUHN
            e1 = ntuple(d -> d == p[1] ? 1 : 0, 3)
            e2 = ntuple(d -> (d == p[1] || d == p[2]) ? 1 : 0, 3)
            push!(conns, (corner((0,0,0)), corner(e1), corner(e2), corner((1,1,1))))
        end
    end
    conn = reduce(hcat, [collect(c) for c in conns])
    # Orient positively; the Kuhn ordering alternates handedness.
    for e in axes(conn, 2)
        X = coords[:, conn[:, e]]
        J = [X[:, 2] - X[:, 1]  X[:, 3] - X[:, 1]  X[:, 4] - X[:, 1]]
        if det(J) < 0
            conn[3, e], conn[4, e] = conn[4, e], conn[3, e]
        end
    end
    return coords, conn
end

"Add edge midpoints in RFE's TET10 order: (1,2),(2,3),(3,1),(1,4),(2,4),(3,4)."
const _TET_EDGES = ((1,2), (2,3), (3,1), (1,4), (2,4), (3,4))

function promote_to_p2(coords, conn)
    nv = size(coords, 2)
    edge_id = Dict{Tuple{Int,Int}, Int}()
    extra = Vector{Vector{Float64}}()
    conn10 = zeros(Int, 10, size(conn, 2))
    conn10[1:4, :] = conn
    for e in axes(conn, 2), (k, (a, b)) in enumerate(_TET_EDGES)
        na, nb = conn[a, e], conn[b, e]
        key = minmax(na, nb)
        id = get(edge_id, key, 0)
        if id == 0
            push!(extra, 0.5 * (coords[:, na] + coords[:, nb]))
            id = nv + length(extra)
            edge_id[key] = id
        end
        conn10[4 + k, e] = id
    end
    coords10 = hcat(coords, reduce(hcat, extra))
    return coords10, conn10
end

# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
pressure_basis(::Val{1}, xi) = (1.0,)
pressure_basis(::Val{4}, xi) = (1.0, xi[1], xi[2], xi[3])

"""
Assemble K (H1 seminorm), G (volumetric coupling) and M (pressure mass) for a
displacement space of degree `p` and an element-local pressure space of
dimension `m`.  Boundary displacement degrees of freedom are eliminated.
"""
function assemble(coords, conn, p::Int, m::Int; q_degree = 2)
    elem = p == 1 ? RFE.Tet{RFE.Lagrange, 1}() : RFE.Tet{RFE.Lagrange, 2}()
    ref  = RFE.ReferenceFE(elem, RFE.GaussLegendre(q_degree))
    nqp  = RFE.num_cell_quadrature_points(ref)
    nen  = size(conn, 1)
    nnode, nelem = size(coords, 2), size(conn, 2)
    ndof = NSD * nnode

    # Free displacement DOFs: everything not on the cube boundary.
    tol = 1e-10
    onbnd(x) = any(abs(x[d]) < tol || abs(x[d] - 1) < tol for d in 1:NSD)
    free = Int[]
    for n in 1:nnode, d in 1:NSD
        onbnd(coords[:, n]) || push!(free, NSD * (n - 1) + d)
    end
    gmap = zeros(Int, ndof)
    for (i, g) in enumerate(free); gmap[g] = i; end
    nu, np = length(free), m * nelem

    KI, KJ, KV = Int[], Int[], Float64[]
    GI, GJ, GV = Int[], Int[], Float64[]
    MI, MJ, MV = Int[], Int[], Float64[]

    for e in 1:nelem
        X = coords[:, conn[:, e]]
        for q in 1:nqp
            dN_ref = RFE.cell_shape_function_gradient(ref, q)
            w      = RFE.cell_quadrature_weight(ref, q)
            xi     = RFE.cell_quadrature_point(ref, q)
            J      = X * dN_ref
            detJ   = det(J)
            dN     = dN_ref / J
            dV     = w * detJ
            phi    = pressure_basis(Val(m), xi)

            for a in 1:nen, da in 1:NSD
                ga = NSD * (conn[a, e] - 1) + da
                ia = gmap[ga];  ia == 0 && continue
                # K: vector Laplacian, grad(N_a e_da) : grad(N_b e_db)
                for b in 1:nen
                    gb = NSD * (conn[b, e] - 1) + da
                    ib = gmap[gb];  ib == 0 && continue
                    v = dV * dot(dN[a, :], dN[b, :])
                    push!(KI, ia); push!(KJ, ib); push!(KV, v)
                end
                # G: int phi_m * div(N_a e_da) = int phi_m * dN_a/dx_da
                for mm in 1:m
                    push!(GI, m * (e - 1) + mm); push!(GJ, ia)
                    push!(GV, dV * phi[mm] * dN[a, da])
                end
            end
            for mm in 1:m, nn in 1:m
                push!(MI, m * (e - 1) + mm); push!(MJ, m * (e - 1) + nn)
                push!(MV, dV * phi[mm] * phi[nn])
            end
        end
    end
    K = sparse(KI, KJ, KV, nu, nu)
    G = sparse(GI, GJ, GV, np, nu)
    M = sparse(MI, MJ, MV, np, np)
    return K, G, M, nu, np
end

"""
Smallest nonzero eigenvalue of G K^-1 G' q = lambda M q, and the number of zero
modes.  Returns beta_h and the zero count.
"""
function infsup(N::Int, p::Int, m::Int)
    coords, conn = tet4_mesh(N)
    p == 2 && ((coords, conn) = promote_to_p2(coords, conn))
    K, G, M, nu, np = assemble(coords, conn, p, m)
    nu == 0 && return (; beta = NaN, nzero = -1, nu, np, h = 1 / N)

    F = cholesky(Symmetric(K))
    S = Matrix(G * (F \ Matrix(G')))          # np x np Schur complement
    S = Symmetric((S + S') / 2)
    Md = Symmetric(Matrix(M))
    ev = eigvals(S, Md)
    ev = sort(real.(ev))
    tol = 1e-9 * maximum(abs, ev)
    nzero = count(<(tol), ev)
    idx = findfirst(>=(tol), ev)
    beta = idx === nothing ? 0.0 : sqrt(max(ev[idx], 0.0))
    return (; beta, nzero, nu, np, h = 1 / N)
end

function main()
    println("Numerical inf-sup test, unit cube, displacement fixed on the boundary.")
    println("beta_h must not decay with h.  Exactly one zero mode is expected")
    println("(the hydrostatic mode); more than one means spurious pressure modes.\n")
    cases = (("P1 / P0   (control: known unstable)", 1, 1),
             ("P2 / P0",                             2, 1),
             ("P2 / P1 disc  (the candidate)",       2, 4))
    for (label, p, m) in cases
        println(label)
        println("  ", rpad("N", 4), rpad("h", 9), rpad("n_u", 8), rpad("n_p", 8),
                rpad("zero modes", 12), "beta_h")
        prev = nothing
        for N in 2:5
            r = infsup(N, p, m)
            trend = prev === nothing ? "" :
                    (r.beta < 0.7 * prev ? "   <-- decaying" : "")
            println("  ", rpad(string(N), 4), rpad(string(round(r.h, sigdigits=3)), 9),
                    rpad(string(r.nu), 8), rpad(string(r.np), 8),
                    rpad(string(r.nzero), 12),
                    string(round(r.beta, sigdigits=4)), trend)
            prev = r.beta
        end
        println()
    end
end

# --------------------------------------------------------------------------
# Dimension count
# --------------------------------------------------------------------------
# A necessary condition, independent of any eigenvalue: if the pressure space
# has more degrees of freedom than the displacement space, then
# rank(G K^-1 G') <= n_u < n_p and the Schur complement is singular by counting
# alone.  The pair cannot be inf-sup stable, whatever the constants do.
#
# This is cheap -- no assembly, no factorization -- so it can be pushed to mesh
# sizes the eigenvalue sweep cannot reach, which is where the asymptotic ratio
# becomes visible.

function dof_counts(N::Int, p::Int, m::Int; bubbles_per_elem::Int = 0)
    coords, conn = tet4_mesh(N)
    p == 2 && ((coords, conn) = promote_to_p2(coords, conn))
    nelem = size(conn, 2)
    tol = 1e-10
    onbnd(x) = any(abs(x[d]) < tol || abs(x[d] - 1) < tol for d in 1:NSD)
    nfree = count(n -> !onbnd(coords[:, n]), axes(coords, 2))
    nu = NSD * nfree + NSD * bubbles_per_elem * nelem
    np = m * nelem
    return (; nu, np, nelem, ratio = np / nu)
end

function dimension_report()
println()
    println("="^74)
    println("Dimension count: n_p / n_u must stay comfortably BELOW 1.")
    println("At or above 1 the Schur complement is singular by counting alone.")
    println("="^74)
    cases = (("P2 / P0",                        2, 1, 0),
             ("P2 / P1 disc",                   2, 4, 0),
             ("P2 + vector bubble / P1 disc",   2, 4, 1))
    for (label, p, m, nb) in cases
        println("\n", label)
        println("  ", rpad("N", 4), rpad("elems", 9), rpad("n_u", 9), rpad("n_p", 9), "n_p/n_u")
        for N in (2, 3, 4, 5, 6, 8, 10, 12)
            c = dof_counts(N, p, m; bubbles_per_elem = nb)
            flag = c.ratio >= 1.0 ? "   SINGULAR by counting" : ""
            println("  ", rpad(string(N), 4), rpad(string(c.nelem), 9),
                    rpad(string(c.nu), 9), rpad(string(c.np), 9),
                    rpad(string(round(c.ratio, sigdigits = 4)), 9), flag)
        end
    end
end

dimension_report()
println()
main()
