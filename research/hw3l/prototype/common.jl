# Shared mesh generation and assembly for the HW3L prototypes.
#
# Meshes are generated rather than read, so a refinement sequence is exact and
# needs no mesher.  Everything here is small-strain and linear: the questions
# these scripts ask -- which modes exist, what the constraint ranks are, how the
# spectrum splits with kappa -- are all properties of the condensed linear
# operator and none of them need a material model or a solver.

using LinearAlgebra, SparseArrays
import ReferenceFiniteElements as RFE

const NSD = 3

# --------------------------------------------------------------------------
# Structured tetrahedral mesh of the unit cube
# --------------------------------------------------------------------------
# Freudenthal (Kuhn) subdivision: each cube splits into the six tetrahedra
# {0, e_a, e_a + e_b, (1,1,1)} over the six permutations (a,b,c).  Conforming
# across shared cube faces, which the more obvious five-tetrahedron split is
# not.
const _KUHN = ((1,2,3), (1,3,2), (2,1,3), (2,3,1), (3,1,2), (3,2,1))

function tet4_mesh(N::Int)
    nn = N + 1
    nid(i, j, k) = i + nn * (j + nn * k) + 1
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
    for e in axes(conn, 2)                       # orient positively
        X = coords[:, conn[:, e]]
        J = [X[:, 2] - X[:, 1]  X[:, 3] - X[:, 1]  X[:, 4] - X[:, 1]]
        det(J) < 0 && ((conn[3, e], conn[4, e]) = (conn[4, e], conn[3, e]))
    end
    return coords, conn
end

"Edge midpoints in RFE's TET10 order."
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
    return hcat(coords, reduce(hcat, extra)), conn10
end

mesh_of(N, p) = p == 1 ? tet4_mesh(N) : promote_to_p2(tet4_mesh(N)...)

# --------------------------------------------------------------------------
# Quadrature
# --------------------------------------------------------------------------
# ReferenceFiniteElements supplies tetrahedron rules only up to degree three.
# That is ample for P2 alone -- the shape function gradients are linear, so
# every integrand below is quadratic -- but not for the enriched space.  The
# quartic bubble has a cubic gradient, so the bubble block of Kdev and Kh1 is
# of degree six.
#
# Under-integration would not announce itself.  It would soften precisely the
# degrees of freedom whose effect on beta_h is the question being asked, and
# the Taylor-Hood control cannot catch it, having no bubble to under-integrate.
# A rule of arbitrary degree is therefore constructed here.
#
# The construction is the conical product: the cube [0,1]^3 is mapped onto the
# reference tetrahedron by
#
#     x = u,   y = v (1 - u),   z = w (1 - u)(1 - v),
#
# whose Jacobian is (1 - u)^2 (1 - v).  Absorbing that factor into Gauss-Jacobi
# weights rather than into the integrand keeps the rule exact: a monomial
# x^a y^b z^c of total degree d pulls back to
#
#     [u^a (1-u)^{b+c}] [v^b (1-v)^c] [w^c]
#
# against the weights (1-u)^2, (1-v) and 1, with degrees d, d and c
# respectively.  An n-point Gauss rule is exact to degree 2n-1, so
# n = ceil((d+1)/2) suffices in every direction.

"""
Gauss-Jacobi nodes and weights on [0,1] for the weight (1-t)^a, by
Golub-Welsch.  `a = 0` is Gauss-Legendre.
"""
function gauss_jacobi01(n::Int, a::Real)
    n >= 1 || error("need at least one quadrature point, got $n")
    # Recurrence coefficients for the weight (1-x)^a on [-1,1] (Gautschi), the
    # b = 0 case of the Jacobi family.
    mu0 = 2.0^(a + 1) / (a + 1)
    d = Vector{Float64}(undef, n)
    d[1] = -a / (a + 2)
    for k in 1:(n - 1)
        d[k + 1] = -a^2 / ((2k + a) * (2k + a + 2))
    end
    e = Vector{Float64}(undef, n - 1)
    for k in 1:(n - 1)
        b = k == 1 ? 4 * (a + 1) / ((a + 2)^2 * (a + 3)) :
                     4 * k^2 * (k + a)^2 /
                         ((2k + a)^2 * (2k + a + 1) * (2k + a - 1))
        e[k] = sqrt(b)
    end
    F = eigen(SymTridiagonal(d, e))
    x = F.values
    w = mu0 .* vec(F.vectors[1, :]) .^ 2
    # Map [-1,1] -> [0,1] by t = (1+x)/2, which carries (1-x)^a dx into
    # 2^(a+1) (1-t)^a dt.
    return (1 .+ x) ./ 2, w ./ 2.0^(a + 1)
end

"""
Conical-product rule on the reference tetrahedron, exact to total degree
`deg`.  Returns points as a 3 x nqp matrix and the matching weights, which sum
to the reference volume 1/6.
"""
function tet_rule(deg::Int)
    n = cld(deg + 1, 2)
    tu, wu = gauss_jacobi01(n, 2)
    tv, wv = gauss_jacobi01(n, 1)
    tw, ww = gauss_jacobi01(n, 0)
    pts = zeros(NSD, n^3)
    wts = zeros(n^3)
    q = 0
    for i in 1:n, j in 1:n, k in 1:n
        u, v, w = tu[i], tv[j], tw[k]
        q += 1
        pts[:, q] = [u, v * (1 - u), w * (1 - u) * (1 - v)]
        wts[q] = wu[i] * wv[j] * ww[k]
    end
    return pts, wts
end

# --------------------------------------------------------------------------
# Spaces and bookkeeping
# --------------------------------------------------------------------------
pressure_basis(::Val{1}, xi) = (1.0,)
pressure_basis(::Val{4}, xi) = (1.0, xi[1], xi[2], xi[3])

# Continuous P1 pressure on the tetrahedron vertices -- the Taylor-Hood pair.
# Carried here as a POSITIVE CONTROL: it is stable in three dimensions, so a
# test that reports it decaying is measuring something other than what it
# claims.  A negative control alone (P1/P0, known unstable) cannot catch a test
# that condemns everything.
const _P1C = -1
p1_shape(xi) = (1 - xi[1] - xi[2] - xi[3], xi[1], xi[2], xi[3])

ref_element(p) = p == 1 ? RFE.Tet{RFE.Lagrange, 1}() : RFE.Tet{RFE.Lagrange, 2}()

# Boundary condition selectors.  `:all` fixes the whole boundary, which is the
# standard setting for an inf-sup test but also the most constraining one a
# mesh can have.  `:zface` fixes only z = 0, leaving the block free to deform --
# much closer to the confined-but-not-clamped conditions under which soft modes
# are reported, and therefore the setting in which they have a chance to appear.
const _BC_ALL   = :all
const _BC_ZFACE = :zface

"Displacement DOFs left free by the chosen boundary condition."
function free_dofs(coords, bc::Symbol = _BC_ALL)
    tol = 1e-10
    fixed = if bc === _BC_ALL
        x -> any(abs(x[d]) < tol || abs(x[d] - 1) < tol for d in 1:NSD)
    elseif bc === _BC_ZFACE
        x -> abs(x[3]) < tol
    else
        error("unknown boundary condition $bc")
    end
    ndof = NSD * size(coords, 2)
    free = Int[]
    for n in axes(coords, 2), d in 1:NSD
        fixed(coords[:, n]) || push!(free, NSD * (n - 1) + d)
    end
    gmap = zeros(Int, ndof)
    for (i, g) in enumerate(free); gmap[g] = i; end
    return free, gmap
end

"Voigt B operator (6 x 3*nen), and dV, at one quadrature point."
function bmatrix(dN, nen)
    B = zeros(6, NSD * nen)
    for a in 1:nen
        c = NSD * (a - 1)
        B[1, c+1] = dN[a, 1]
        B[2, c+2] = dN[a, 2]
        B[3, c+3] = dN[a, 3]
        B[4, c+1] = dN[a, 2];  B[4, c+2] = dN[a, 1]
        B[5, c+2] = dN[a, 3];  B[5, c+3] = dN[a, 2]
        B[6, c+1] = dN[a, 3];  B[6, c+3] = dN[a, 1]
    end
    return B
end

const _VOIGT_W = Diagonal([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])  # energy metric
const _DEV = let I6 = Matrix{Float64}(I, 6, 6)
    for i in 1:3, j in 1:3; I6[i, j] -= 1/3; end
    I6
end

# --------------------------------------------------------------------------
# Bubble enrichment
# --------------------------------------------------------------------------
# The three-dimensional Crouzeix-Raviart displacement space over a
# discontinuous P1 pressure is P2 enriched by BOTH an interior bubble and one
# bubble per face.  The two are not interchangeable and they are not the same
# kind of object:
#
#   interior, b = 256 L1 L2 L3 L4 (quartic), vanishes on the whole element
#       boundary.  Its three degrees of freedom are element-local: never
#       shared, never reached by a boundary condition, condensable.  Its
#       divergence integrates to zero against any CONSTANT, so it contributes
#       nothing at all to a P0 pressure -- it acts only on the linear modes.
#
#   face, b_f = 27 La Lb Lc (cubic) over the three vertices of face f, vanishes
#       on the other three faces but NOT on its own.  Both elements sharing a
#       face see the same cubic on it, so the enrichment is conforming, and its
#       three degrees of freedom are therefore SHARED: they are global unknowns
#       and cannot be condensed element by element.  A face bubble on the
#       domain boundary is fixed by the displacement boundary condition.
#
# That distinction matters beyond bookkeeping.  The claim that the enrichment
# "adds no global degrees of freedom" holds for the interior bubble alone; the
# face bubbles are precisely the part that does add them.

bary(xi) = (1 - xi[1] - xi[2] - xi[3], xi[1], xi[2], xi[3])
const _BARY_GRAD = ([-1.0, -1.0, -1.0], [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],    [0.0, 0.0, 1.0])

"Face f is the one opposite vertex f, over the remaining three vertices."
const _TET_FACES = ((2,3,4), (1,3,4), (1,2,4), (1,2,3))

"Value and reference gradient of `scale * prod(L[m] for m in idx)`."
function _bubble_of(idx, scale, xi)
    L = bary(xi)
    v = scale * prod(L[m] for m in idx)
    g = zeros(NSD)
    for m in idx
        c = scale
        for n in idx
            n == m || (c *= L[n])
        end
        g .+= c .* _BARY_GRAD[m]
    end
    return v, g
end

"Quartic interior bubble, unit height at the centroid."
interior_bubble(xi) = _bubble_of((1, 2, 3, 4), 256.0, xi)

"Cubic bubble on face `f`, unit height at that face's centroid."
face_bubble(f, xi) = _bubble_of(_TET_FACES[f], 27.0, xi)

const _ENRICH = (:none, :interior, :full)

"""
Global face numbering for the tetrahedral mesh, with faces on the constrained
boundary removed.

Returns `(nfree, fmap)` where `fmap[f, e]` is the free-face index of local face
`f` of element `e`, or 0 if that face carries a displacement boundary
condition.  A face is on the boundary exactly when its centroid is, which on
this cube is an exact test: a centroid can only reach a bounding plane if all
three of its vertices already lie on it.
"""
function face_table(coords, conn, bc::Symbol)
    tol = 1e-10
    onbnd = if bc === _BC_ALL
        x -> any(abs(x[d]) < tol || abs(x[d] - 1) < tol for d in 1:NSD)
    elseif bc === _BC_ZFACE
        x -> abs(x[3]) < tol
    else
        error("unknown boundary condition $bc")
    end
    ids = Dict{NTuple{3,Int}, Int}()
    fmap = zeros(Int, 4, size(conn, 2))
    n = 0
    for e in axes(conn, 2), f in 1:4
        vs = ntuple(k -> conn[_TET_FACES[f][k], e], 3)
        key = Tuple(sort(collect(vs)))
        centroid = sum(coords[:, v] for v in vs) ./ 3
        onbnd(centroid) && continue
        id = get(ids, key, 0)
        if id == 0
            n += 1
            id = n
            ids[key] = id
        end
        fmap[f, e] = id
    end
    return n, fmap
end

"""
Assemble the operators the prototypes need, with boundary displacement
eliminated:

    Kdev[i,j] = int 2 mu dev(eps_i) : dev(eps_j)      deviatoric stiffness
    Kh1[i,j]  = int grad(N_i) : grad(N_j)             H1 seminorm
    G[m,i]    = int phi_m div(N_i)                    volumetric coupling
    M[m,n]    = int phi_m phi_n                       pressure mass

`bubble` selects the displacement enrichment: `:none`, `:interior` for the
element-local quartic bubble alone, or `:full` for the three-dimensional
Crouzeix-Raviart space, interior plus one bubble per face.  Only `:interior`
leaves the added degrees of freedom element-local; `:full` shares its face
bubbles between neighbors.

The returned `nu_nodal`, `n_int` and `n_face` give the three blocks of the
displacement numbering, in that order.
"""
function assemble_all(coords, conn, p::Int, m::Int; mu = 1.0,
                      bubble::Symbol = :none,
                      q_degree::Int = bubble === :none ? 2 : 6,
                      bc::Symbol = _BC_ALL, nvert::Int = 0)
    bubble in _ENRICH || error(
        "unknown enrichment $bubble; expected one of $(_ENRICH)")
    if bubble !== :none
        p == 2 || error("the bubble enrichment is defined on P2 only, got p = $p")
        q_degree >= 6 || error(
            "the enriched space needs a rule exact to degree 6 -- the quartic " *
            "interior bubble has a cubic gradient, so its block of Kdev and " *
            "Kh1 is of degree 6 -- but q_degree = $q_degree was given. " *
            "Under-integrating it would soften exactly the modes this study " *
            "measures.")
    end
    el   = ref_element(p)
    qpts, qwts = tet_rule(q_degree)
    nqp  = length(qwts)
    nen  = size(conn, 1)
    nelem = size(conn, 2)
    free, gmap = free_dofs(coords, bc)
    # Continuous P1 pressure is numbered by vertex; every discontinuous space is
    # numbered element by element.
    continuous = m == _P1C
    nu_nodal = length(free)
    n_int  = bubble === :none ? 0 : NSD * nelem
    nfaces, fmap = bubble === :full ? face_table(coords, conn, bc) : (0, zeros(Int, 4, nelem))
    n_face = NSD * nfaces
    nu = nu_nodal + n_int + n_face
    np = continuous ? nvert : m * nelem
    nen_a = nen + (bubble === :none ? 0 : 1) + (bubble === :full ? 4 : 0)
    int_off  = nu_nodal
    face_off = nu_nodal + n_int

    DI, DJ, DV = Int[], Int[], Float64[]
    HI, HJ, HV = Int[], Int[], Float64[]
    GI, GJ, GV = Int[], Int[], Float64[]
    MI, MJ, MV = Int[], Int[], Float64[]

    for e in 1:nelem
        X = coords[:, conn[:, e]]
        gdofs = [NSD * (conn[a, e] - 1) + d for a in 1:nen for d in 1:NSD]
        rows  = [gmap[g] for g in gdofs]
        # The interior bubble is always free: it vanishes on the element
        # boundary, so no boundary condition can reach it.  A face bubble is
        # free only when its face is not on the constrained boundary.
        if bubble !== :none
            append!(rows, [int_off + NSD * (e - 1) + d for d in 1:NSD])
        end
        if bubble === :full
            for f in 1:4
                fid = fmap[f, e]
                append!(rows, [fid == 0 ? 0 : face_off + NSD * (fid - 1) + d
                               for d in 1:NSD])
            end
        end
        for q in 1:nqp
            xi = qpts[:, q]
            w  = qwts[q]
            dN_ref = RFE.shape_function_gradient(el, xi)
            J  = X * dN_ref
            dN = dN_ref / J
            if bubble !== :none
                _, gb = interior_bubble(xi)
                dN = vcat(dN, reshape(gb, 1, NSD) / J)
            end
            if bubble === :full
                for f in 1:4
                    _, gf = face_bubble(f, xi)
                    dN = vcat(dN, reshape(gf, 1, NSD) / J)
                end
            end
            dV = w * det(J)
            B  = bmatrix(dN, nen_a)
            Bd = _DEV * B
            Ke = dV * 2mu * (Bd' * _VOIGT_W * Bd)
            phi = continuous ? p1_shape(xi) : pressure_basis(Val(m), xi)
            prow = continuous ? (a -> conn[a, e]) : (a -> m * (e - 1) + a)
            npl = continuous ? 4 : m
            trB = B[1, :] + B[2, :] + B[3, :]

            for (i, ri) in enumerate(rows)
                ri == 0 && continue
                for (j, rj) in enumerate(rows)
                    rj == 0 && continue
                    push!(DI, ri); push!(DJ, rj); push!(DV, Ke[i, j])
                end
                a, d = fldmod1(i, NSD)
                for (j, rj) in enumerate(rows)
                    rj == 0 && continue
                    b, d2 = fldmod1(j, NSD)
                    d == d2 || continue
                    push!(HI, ri); push!(HJ, rj)
                    push!(HV, dV * dot(dN[a, :], dN[b, :]))
                end
                for mm in 1:npl
                    push!(GI, prow(mm)); push!(GJ, ri)
                    push!(GV, dV * phi[mm] * trB[i])
                end
            end
            for mm in 1:npl, nn in 1:npl
                push!(MI, prow(mm)); push!(MJ, prow(nn))
                push!(MV, dV * phi[mm] * phi[nn])
            end
        end
    end
    return (; Kdev = sparse(DI, DJ, DV, nu, nu),
              Kh1 = sparse(HI, HJ, HV, nu, nu),
              G   = sparse(GI, GJ, GV, np, nu),
              M   = sparse(MI, MJ, MV, np, np),
              nu, np, nelem, nu_nodal, n_int, n_face)
end
