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
# Spaces and bookkeeping
# --------------------------------------------------------------------------
pressure_basis(::Val{1}, xi) = (1.0,)
pressure_basis(::Val{4}, xi) = (1.0, xi[1], xi[2], xi[3])

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

"""
Assemble the operators the prototypes need, with boundary displacement
eliminated:

    Kdev[i,j] = int 2 mu dev(eps_i) : dev(eps_j)      deviatoric stiffness
    Kh1[i,j]  = int grad(N_i) : grad(N_j)             H1 seminorm
    G[m,i]    = int phi_m div(N_i)                    volumetric coupling
    M[m,n]    = int phi_m phi_n                       pressure mass
"""
function assemble_all(coords, conn, p::Int, m::Int; mu = 1.0, q_degree = 2, bc::Symbol = _BC_ALL)
    ref  = RFE.ReferenceFE(ref_element(p), RFE.GaussLegendre(q_degree))
    nqp  = RFE.num_cell_quadrature_points(ref)
    nen  = size(conn, 1)
    nelem = size(conn, 2)
    free, gmap = free_dofs(coords, bc)
    nu, np = length(free), m * nelem

    DI, DJ, DV = Int[], Int[], Float64[]
    HI, HJ, HV = Int[], Int[], Float64[]
    GI, GJ, GV = Int[], Int[], Float64[]
    MI, MJ, MV = Int[], Int[], Float64[]

    for e in 1:nelem
        X = coords[:, conn[:, e]]
        gdofs = [NSD * (conn[a, e] - 1) + d for a in 1:nen for d in 1:NSD]
        rows  = [gmap[g] for g in gdofs]
        for q in 1:nqp
            dN_ref = RFE.cell_shape_function_gradient(ref, q)
            w  = RFE.cell_quadrature_weight(ref, q)
            xi = RFE.cell_quadrature_point(ref, q)
            J  = X * dN_ref
            dN = dN_ref / J
            dV = w * det(J)
            B  = bmatrix(dN, nen)
            Bd = _DEV * B
            Ke = dV * 2mu * (Bd' * _VOIGT_W * Bd)
            phi = pressure_basis(Val(m), xi)
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
                for mm in 1:m
                    push!(GI, m * (e - 1) + mm); push!(GJ, ri)
                    push!(GV, dV * phi[mm] * trB[i])
                end
            end
            for mm in 1:m, nn in 1:m
                push!(MI, m * (e - 1) + mm); push!(MJ, m * (e - 1) + nn)
                push!(MV, dV * phi[mm] * phi[nn])
            end
        end
    end
    return (; Kdev = sparse(DI, DJ, DV, nu, nu),
              Kh1 = sparse(HI, HJ, HV, nu, nu),
              G   = sparse(GI, GJ, GV, np, nu),
              M   = sparse(MI, MJ, MV, np, np),
              nu, np, nelem)
end
