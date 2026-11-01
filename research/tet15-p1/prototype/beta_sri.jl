# Inf-sup constant of the Crouzeix-Raviart pair (P2 + face bubbles + interior
# bubble / P1 disc) when the pressure-displacement coupling G is integrated
# with the four-point rule of degree two, as in the selective reduced
# integration element TET15SR4 of Danielson (2014).  The H1 seminorm matrix
# and the pressure mass are integrated exactly (degree 6), as in beta.jl.
#
# beta_h is the smallest nonzero singular value of M^(-1/2) G L^(-T), with
# the null dimension fixed as n_p - rank(G) (beta.jl).  Both couplings are
# reported on each mesh: the exactly integrated one reproduces tab:beta of
# the manuscript, the four-point one is the new measurement.
#
#   julia -t 32 --project=. research/tet15-p1/prototype/beta_sri.jl [Nmax]

include("common.jl")

"The four-point rule of degree two on the reference tetrahedron: points
(b,a,a,a) in barycentric coordinates and permutations, weights 1/24."
function tet_rule4()
    a = (5 - sqrt(5)) / 20
    b = 1 - 3a
    pts = [b a a a; a b a a; a a b a; a a a b]        # rows: barycentric
    xi = Matrix(pts[:, 2:4]')                         # 3 x 4 reference coords
    return xi, fill(1 / 24, 4)
end

"""
Re-assemble the coupling matrix G of `assemble_all` with the quadrature rule
`(qpts, qwts)`, using the same degree-of-freedom numbering.  Only the enriched
space with a discontinuous pressure of dimension m is supported.
"""
function assemble_G(coords, conn, m::Int, qpts, qwts; bubble::Symbol = :full,
                    bc::Symbol = _BC_ALL, bc_coords = coords)
    el = ref_element(2)
    nqp = length(qwts)
    nen = size(conn, 1)
    nelem = size(conn, 2)
    free, gmap = free_dofs(bc_coords, bc)
    nu_nodal = length(free)
    has_int  = bubble in (:interior, :full)
    has_face = bubble in (:face, :full)
    n_int  = has_int ? NSD * nelem : 0
    nfaces, fmap = has_face ? face_table(bc_coords, conn, bc) : (0, zeros(Int, 4, nelem))
    n_face = NSD * nfaces
    nu = nu_nodal + n_int + n_face
    np = m * nelem
    nen_a = nen + (has_int ? 1 : 0) + (has_face ? 4 : 0)
    int_off  = nu_nodal
    face_off = nu_nodal + n_int
    GI, GJ, GV = Int[], Int[], Float64[]
    for e in 1:nelem
        X = coords[:, conn[:, e]]
        gdofs = [NSD * (conn[a, e] - 1) + d for a in 1:nen for d in 1:NSD]
        rows  = [gmap[g] for g in gdofs]
        has_int && append!(rows, [int_off + NSD * (e - 1) + d for d in 1:NSD])
        if has_face
            for f in 1:4
                fid = fmap[f, e]
                append!(rows, [fid == 0 ? 0 : face_off + NSD * (fid - 1) + d for d in 1:NSD])
            end
        end
        for q in 1:nqp
            xi = qpts[:, q]
            w  = qwts[q]
            dN_ref = RFE.shape_function_gradient(el, xi)
            J  = X * dN_ref
            dN = dN_ref / J
            if has_int
                _, gb = interior_bubble(xi)
                dN = vcat(dN, reshape(gb, 1, NSD) / J)
            end
            if has_face
                for f in 1:4
                    _, gf = face_bubble(f, xi)
                    dN = vcat(dN, reshape(gf, 1, NSD) / J)
                end
            end
            dV = w * det(J)
            B  = bmatrix(dN, nen_a)
            trB = B[1, :] + B[2, :] + B[3, :]
            phi = pressure_basis(Val(m), xi)
            for (i, ri) in enumerate(rows)
                ri == 0 && continue
                for mm in 1:m
                    push!(GI, m * (e - 1) + mm); push!(GJ, ri)
                    push!(GV, dV * phi[mm] * trB[i])
                end
            end
        end
    end
    return sparse(GI, GJ, GV, np, nu)
end

"""
beta_h from the H1 matrix, the pressure mass and a coupling matrix.  Also returns the
next singular value above beta_h and the overlap (cosine in the M inner product) of the
singular vector of beta_h with the elementwise constant pressure `cvec`; on a curved mesh
the four-point rule does not annihilate the constant pressure, which then appears as a
controlled mode with a small singular value.
"""
function beta_of(Kh1, M, G, cvec)
    F  = cholesky(Symmetric(Kh1))
    Gt = Matrix(G')
    Z  = Matrix(F.L \ Gt[F.p, :])
    UM = cholesky(Symmetric(Matrix(M))).U
    W  = Matrix(Z / UM)
    S  = svd(W)
    sv = S.S                               # descending
    r  = rank(Matrix(G); rtol = 1e-10)
    L  = length(sv)
    beta = r >= 1 ? sv[r] : NaN
    nxt  = r >= 2 ? sv[r - 1] : NaN
    c  = UM * cvec
    ov = r >= 1 ? abs(dot(S.V[:, r], c)) / norm(c) : NaN
    return (; beta, nnull = size(G, 1) - r, nxt, ov)
end

"Assemble on the given P2 mesh and report beta_h for the exact and the four-point coupling.
`bc_coords` are the coordinates that decide the boundary conditions (the reference cube
for a deformed mesh, as in deformed.jl)."
function run_mesh(coords, conn; bc = _BC_ALL, label = "", bc_coords = coords)
    a = assemble_all(coords, conn, 2, 4; bc, bubble = :full, q_degree = 6, bc_coords)
    qpts4, qwts4 = tet_rule4()
    G4 = assemble_G(coords, conn, 4, qpts4, qwts4; bubble = :full, bc, bc_coords)
    # consistency: the exact re-assembly must reproduce assemble_all's G
    qpts6, qwts6 = tet_rule(6)
    G6 = assemble_G(coords, conn, 4, qpts6, qwts6; bubble = :full, bc, bc_coords)
    dG = norm(G6 - a.G) / norm(a.G)
    cvec = [mod1(i, 4) == 1 ? 1.0 : 0.0 for i in 1:a.np]   # constant pressure, basis (1, xi1, xi2, xi3)
    rx = beta_of(a.Kh1, a.M, a.G, cvec)
    r4 = beta_of(a.Kh1, a.M, G4, cvec)
    dcoup = norm(G4 - a.G) / norm(a.G)
    println(rpad(label, 10), rpad(string(a.nelem), 8), rpad(string(a.nu), 8), rpad(string(a.np), 8),
            rpad(string(round(rx.beta, sigdigits = 7)), 10), rpad(string(rx.nnull), 6),
            rpad(string(round(r4.beta, sigdigits = 7)), 10), rpad(string(r4.nnull), 6),
            rpad(string(round(r4.nxt, sigdigits = 7)), 10), rpad(string(round(r4.ov, sigdigits = 3)), 8),
            rpad(string(round(r4.beta / rx.beta, sigdigits = 4)), 8),
            rpad(string(round(dcoup, sigdigits = 3)), 10),
            string(round(dG, sigdigits = 2)))
    return (; rx, r4)
end

# The five maps of deformed.jl: the isoparametric image of the P2 mesh is a
# curved mesh for the non-affine maps (twists, shear, inflation).
const CENTER = [0.5, 0.5, 0.5]
twist(alpha) = X -> begin
    a = alpha * X[3]; c, s = cos(a), sin(a)
    d = X .- CENTER
    CENTER .+ [c * d[1] - s * d[2], s * d[1] + c * d[2], d[3]]
end
shear(c) = X -> [X[1] + c * X[3]^2, X[2], X[3]]
inflate(k) = X -> (d = X .- CENTER; CENTER .+ d .* (1 + k * dot(d, d)))
compress(l) = X -> [X[1], X[2], l * X[3]]
const MAPS = (("reference", X -> X), ("twist 90", twist(pi / 2)), ("twist 180", twist(pi)),
              ("shear", shear(0.5)), ("inflate", inflate(0.3)), ("compress", compress(0.5)))

function deformed(N)
    println("Deformed configurations of deformed.jl at N = $N: exact and four-point coupling on the isoparametric image.")
    println(rpad("map", 10), rpad("elems", 8), rpad("n_u", 8), rpad("n_p", 8), rpad("exact", 10), rpad("null", 6),
            rpad("4pt", 10), rpad("null", 6), rpad("4pt next", 10), rpad("ov", 8), rpad("ratio", 8), rpad("|dG|/|G|", 10), "check")
    c4, n4 = tet4_mesh(N)
    coords, conn = promote_to_p2(c4, n4)
    for (mname, map) in MAPS
        def = reduce(hcat, (map(coords[:, i]) for i in axes(coords, 2)))
        run_mesh(def, conn; label = mname, bc_coords = coords)
        flush(stdout)
    end
end

function main()
    if length(ARGS) >= 1 && ARGS[1] == "deformed"
        deformed(length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5)
        return
    end
    Nmax = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 6
    println("Crouzeix-Raviart pair P2 + face + interior bubbles / P1 disc, Freudenthal mesh of the unit cube, whole boundary fixed.")
    println("beta_h with the exactly integrated coupling (exact) and with the coupling integrated by the four-point rule (4pt).")
    println("||G4 - G||/||G||: relative Frobenius difference of the two coupling matrices; check: ||G6 - G||/||G|| of the re-assembly.")
    println(rpad("N", 10), rpad("elems", 8), rpad("n_u", 8), rpad("n_p", 8), rpad("exact", 10), rpad("null", 6),
            rpad("4pt", 10), rpad("null", 6), rpad("4pt next", 10), rpad("ov", 8), rpad("ratio", 8), rpad("|dG|/|G|", 10), "check")
    for N in 2:Nmax
        c4, n4 = tet4_mesh(N)
        coords, conn = promote_to_p2(c4, n4)
        run_mesh(coords, conn; label = string(N))
        flush(stdout)
    end
end

main()
