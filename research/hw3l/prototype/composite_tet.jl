# Soft modes of the composite tetrahedron under a plastic tangent.
#
# The composite tetrahedron of Albany-LCM ("Use Composite Tet 10") has the
# ten nodes of the quadratic tetrahedron.  Its displacement is piecewise
# linear on twelve subtetrahedra: the four corner tetrahedra of the edge
# midpoints, and the eight tetrahedra joining the centroid c to the faces of
# the inner octahedron, with u(c) the mean of the six midpoint values
# (Intrepid2 Basis_HGRAD_TET_COMP12_FEM, OPERATOR_VALUE).  The gradient the
# element uses is not that piecewise-constant gradient: OPERATOR_GRAD returns
# a linear field, its L2 projection onto linear functions of the element
# (checked below).  Albany evaluates the constitutive response at the points
# of the degree-3 rule of Intrepid2 (five points, centroid weight −2/15), the
# rule of the Cook runs ("Cubature Degree: 3").
#
# Volumetric response, two variants:
#   (a) "Weighted Volume Average J: true" (Kinematics_Def.hpp): J at every
#       point is replaced by the element mean J̄ and F by (J̄/J)^(1/3) F.
#       Linearized, the deviatoric strain is that of the point and the
#       volumetric strain is the element mean of tr ε̃: the constraint that
#       survives κ → ∞ is  ∫_K tr ε̃ dV = 0  per element, a P0 pressure.
#   (b) without it, J at each of the five points: the constraints are
#       tr ε̃(ξ_q) = 0 at the five points, which for the linear tr ε̃ is
#       tr ε̃ ≡ 0 in the element, the same kernel as the four moments
#       ∫_K φ tr ε̃ dV = 0, φ ∈ P1.
#
# The measured quantities are those of plastic.jl and plastic_zone.jl: the
# lowest generalized eigenpairs of the deviatoric form (with the projected
# gradient, as the element computes it) on ker G_h against the H1 seminorm of
# the displacement field, and the dilatation r_v = ∫(div v)² / |∇v|²_1 that
# each mode carries.  The H1 seminorm and r_v use the gradient of the
# displacement field itself, piecewise constant on the subtetrahedra: r_v
# measures the volume change a continuum would charge with κ, which is the
# question.  r_v of the projected gradient is printed next to it.
#
#   julia --project=. research/hw3l/prototype/composite_tet.jl

include("common.jl")

# --------------------------------------------------------------------------
# The composite basis on the reference tetrahedron
# --------------------------------------------------------------------------
# Node order of Intrepid2 and of RFE's TET10 (the same): vertices 1-4, then
# edge midpoints 1-2, 2-3, 3-1, 1-4, 2-4, 3-4.
const _REF10 = ([0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [0.0, 0, 1],
                [0.5, 0, 0], [0.5, 0.5, 0], [0.0, 0.5, 0],
                [0.0, 0, 0.5], [0.5, 0, 0.5], [0.0, 0.5, 0.5])
# Twelve subtetrahedra, by node number (11 is the centroid): the corner
# tetrahedra, then the centroid joined to the eight faces of the octahedron,
# four cut off by the corners and four lying on the faces of the element.
const _SUBTETS = ((1, 5, 7, 8), (2, 5, 6, 9), (3, 6, 7, 10), (4, 8, 9, 10),
                  (11, 5, 7, 8), (11, 5, 6, 9), (11, 6, 7, 10), (11, 8, 9, 10),
                  (11, 5, 6, 7), (11, 5, 9, 8), (11, 7, 10, 8), (11, 6, 10, 9))

"Values of the ten composite functions at the eleven points (ten nodes, centroid)."
function _node_values()
    V = zeros(10, 11)
    for i in 1:10; V[i, i] = 1.0; end
    V[5:10, 11] .= 1 / 6
    return V
end

"""
Volumes and gradients (10 x 3, reference coordinates) of the composite
functions on each subtetrahedron.
"""
function subtet_gradients()
    P = hcat(_REF10..., [0.25, 0.25, 0.25])
    V = _node_values()
    vols = Float64[]; grads = Matrix{Float64}[]
    for s in _SUBTETS
        D = hcat(P[:, s[2]] - P[:, s[1]], P[:, s[3]] - P[:, s[1]], P[:, s[4]] - P[:, s[1]])
        dV = hcat(V[:, s[2]] - V[:, s[1]], V[:, s[3]] - V[:, s[1]], V[:, s[4]] - V[:, s[1]])
        push!(vols, abs(det(D)) / 6)
        push!(grads, dV / D)
    end
    abs(sum(vols) - 1 / 6) < 1e-14 || error("subtetrahedra do not tile the element")
    return vols, grads, [sum(P[:, k] for k in s) ./ 4 for s in _SUBTETS]
end

"Intrepid2 Basis_HGRAD_TET_COMP12_FEM, OPERATOR_GRAD, at reference point (r, s, t)."
function comp12_grad(x)
    r, s, t = x
    g = zeros(10, 3)
    g[1, :] .= (-17 + 20r + 20s + 20t) / 8
    g[2, 1] = -0.375 + 5r / 2
    g[3, 2] = -0.375 + 5s / 2
    g[4, 3] = -0.375 + 5t / 2
    g[5, :] = [-35 * (-1 + 2r + s + t), -4 - 35r + 5s + 10t, -4 - 35r + 10s + 5t] ./ 12
    g[6, :] = [-1 + 5r + 40s - 5t, -1 + 40r + 5s - 5t, -5 * (-1 + r + s + 2t)] ./ 12
    g[7, :] = [-4 + 5r - 35s + 10t, -35 * (-1 + r + 2s + t), -4 + 10r - 35s + 5t] ./ 12
    g[8, :] = [-4 + 5r + 10s - 35t, -4 + 10r + 5s - 35t, -35 * (-1 + r + s + 2t)] ./ 12
    g[9, :] = [-1 + 5r - 5s + 40t, -5 * (-1 + r + 2s + t), -1 + 40r - 5s + 5t] ./ 12
    g[10, :] = [-5 * (-1 + 2r + s + t), -1 - 5r + 5s + 40t, -1 - 5r + 40s + 5t] ./ 12
    return g
end

"The degree-3 rule of Intrepid2 (CubatureDirectTetDefault), five points."
const ALBANY_RULE = ([0.25 1/6 1/6 1/6 1/2; 0.25 1/6 1/6 1/2 1/6; 0.25 1/6 1/2 1/6 1/6],
                     [-2 / 15, 3 / 40, 3 / 40, 3 / 40, 3 / 40] ./ 6)

"""
Largest difference between comp12_grad and the L2 projection onto linear
functions of the subtetrahedron gradients, at the points of a degree-4 rule.
"""
function check_projection()
    vols, grads, cents = subtet_gradients()
    lin(x) = [1.0, x[1], x[2], x[3]]
    qp, qw = tet_rule(2)
    M = sum(qw[q] * lin(qp[:, q]) * lin(qp[:, q])' for q in eachindex(qw))
    # ∫_s ψ_a g = vol_s ψ_a(centroid_s) g_s, exact for linear ψ_a.
    b = [sum(vols[k] * lin(cents[k])[a] * grads[k] for k in eachindex(vols)) for a in 1:4]
    C = [sum((M \ Matrix(I, 4, 4))[a, c] * b[c] for c in 1:4) for a in 1:4]
    xs, _ = tet_rule(4)
    return maximum(norm(sum(lin(xs[:, q])[a] * C[a] for a in 1:4) - comp12_grad(xs[:, q]))
                   for q in axes(xs, 2))
end

# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
"""
Operators on the P2 mesh `coords, conn` (straight-edged, so the map of every
element is affine), boundary displacement eliminated:

    Kdev   ∫ 2μ dev ε̃ : dev ε̃ − 2μ β (n : dev ε̃)²   at the points of `rule`
    Kh1    ∫ ∇v : ∇w                 gradient of the field, per subtetrahedron
    Kdiv   ∫ div v div w             idem
    Kdivp  ∫ tr ε̃(v) tr ε̃(w)         projected gradient, exact
    G      ∫_K φ tr ε̃                φ ∈ P0 (`m = 1`) or P1 (`m = 4`), exact

with ε̃ the symmetric part of the projected gradient.  `element = :p2`
replaces the composite functions by the quadratic Lagrange functions, the
projected gradient by their gradient and the subtetrahedra by a degree-2
rule; it reproduces assemble_all(coords, conn, 2, m) and checks the harness.
"""
function assemble_ct(coords, conn, m::Int; mu = 1.0, bc::Symbol = _BC_ALL,
                     flow = nothing, beta::Real = 0.0, element::Symbol = :ct,
                     rule = element === :ct ? ALBANY_RULE : tet_rule(2))
    flow_fn = flow isa Function ? flow : flow === nothing ? nothing : (x -> (flow, beta))
    free, gmap = free_dofs(coords, bc)
    nu = length(free)
    nelem = size(conn, 2)
    np = m * nelem
    el2 = ref_element(2)
    vols, grads, _ = subtet_gradients()
    energy_grad(xi) = element === :ct ? comp12_grad(xi) :
                      Matrix(RFE.shape_function_gradient(el2, xi))
    # Points and weights, with per-point reference gradients, for the H1 and
    # dilatation forms of the field itself.
    field_pts = if element === :ct
        [(vols[k], grads[k]) for k in eachindex(vols)]
    else
        qp, qw = tet_rule(2)
        [(qw[q], Matrix(RFE.shape_function_gradient(el2, qp[:, q]))) for q in eachindex(qw)]
    end
    exact_pts, exact_wts = tet_rule(2)
    lin(x) = [1.0, x[1], x[2], x[3]]
    Ref2 = [-1.0 -1 -1; 1 0 0; 0 1 0; 0 0 1]
    DI, DJ, DV = Int[], Int[], Float64[]
    HI, HJ, HV = Int[], Int[], Float64[]
    VI, VJ, VV = Int[], Int[], Float64[]
    PI, PJ, PV = Int[], Int[], Float64[]
    GI, GJ, GV = Int[], Int[], Float64[]
    function add!(I_, J_, V_, rows, K)
        for (i, ri) in enumerate(rows), (j, rj) in enumerate(rows)
            (ri == 0 || rj == 0) && continue
            push!(I_, ri); push!(J_, rj); push!(V_, K[i, j])
        end
    end
    for e in 1:nelem
        X = coords[:, conn[:, e]]
        J = X[:, 1:4] * Ref2
        detJ = det(J)
        detJ > 0 || error("non-positive Jacobian in element $e")
        rows = [gmap[NSD * (conn[a, e] - 1) + d] for a in 1:10 for d in 1:NSD]
        # deviatoric form at the points of the element's rule
        Ke = zeros(30, 30)
        for q in eachindex(rule[2])
            xi = rule[1][:, q]
            B = bmatrix(energy_grad(xi) / J, 10)
            Bd = _DEV * B
            dV = rule[2][q] * detJ
            Ke .+= dV * 2mu * (Bd' * _VOIGT_W * Bd)
            if flow_fn !== nothing
                xq = X[:, 1:4] * [1 - sum(xi), xi[1], xi[2], xi[3]]
                n, bq = flow_fn(xq)
                if bq != 0
                    nB = _flow_voigt(n)' * Bd
                    Ke .-= dV * 2mu * bq * (nB' * nB)
                end
            end
        end
        add!(DI, DJ, DV, rows, Ke)
        # H1 seminorm and dilatation of the field
        He = zeros(30, 30); Ve = zeros(30, 30)
        for (w, g) in field_pts
            dN = g / J
            dV = w * detJ
            trB = vec(dN')             # (a, d) -> ∂N_a/∂x_d, ordered as the dofs
            Ve .+= dV * (trB * trB')
            for a in 1:10, b in 1:10, d in 1:NSD
                He[NSD * (a - 1) + d, NSD * (b - 1) + d] += dV * dot(dN[a, :], dN[b, :])
            end
        end
        add!(HI, HJ, HV, rows, He); add!(VI, VJ, VV, rows, Ve)
        # projected dilatation and the constraints, exact (integrands of degree 2)
        Pe = zeros(30, 30)
        Ge = zeros(m, 30)
        for q in eachindex(exact_wts)
            xi = exact_pts[:, q]
            trB = vec((energy_grad(xi) / J)')
            dV = exact_wts[q] * detJ
            Pe .+= dV * (trB * trB')
            phi = m == 1 ? [1.0] : lin(xi)
            Ge .+= dV * (phi * trB')
        end
        add!(PI, PJ, PV, rows, Pe)
        for mm in 1:m, (j, rj) in enumerate(rows)
            rj == 0 && continue
            push!(GI, m * (e - 1) + mm); push!(GJ, rj); push!(GV, Ge[mm, j])
        end
    end
    return (; Kdev = sparse(DI, DJ, DV, nu, nu), Kh1 = sparse(HI, HJ, HV, nu, nu),
              Kdiv = sparse(VI, VJ, VV, nu, nu), Kdivp = sparse(PI, PJ, PV, nu, nu),
              G = sparse(GI, GJ, GV, np, nu), nu, np)
end

# --------------------------------------------------------------------------
# Measurement, as in plastic.jl and plastic_zone.jl
# --------------------------------------------------------------------------
const N_AXIAL = let n = zeros(3, 3); n[1, 1] = n[2, 2] = -1 / sqrt(6); n[3, 3] = 2 / sqrt(6); n end
const N_SHEAR = let n = zeros(3, 3); n[1, 2] = n[2, 1] = 1 / sqrt(2); n end
const NMODES = 20
const RV_SPURIOUS = 0.1

function kernel_modes(a, K; k = NMODES)
    Z = nullspace(Matrix(a.G); rtol = 1e-10)
    size(Z, 2) == 0 && return Float64[], zeros(a.nu, 0), 0
    F = eigen(Symmetric(Z' * Matrix(K) * Z), Symmetric(Z' * Matrix(a.Kh1) * Z))
    kk = min(k, length(F.values))
    return F.values[1:kk], Z * F.vectors[:, 1:kk], size(Z, 2)
end
dilatation(a, v) = dot(v, a.Kdiv * v) / dot(v, a.Kh1 * v)
dilatation_proj(a, v) = dot(v, a.Kdivp * v) / dot(v, a.Kh1 * v)
spurious(rv) = count(>(RV_SPURIOUS), rv)

# Flow fields of plastic_zone.jl, copied.
const CENTER = [0.5, 0.5, 0.5]
punch(X) = [0.0, 0.0, -exp(-((X[1] - 0.5)^2 + (X[2] - 0.5)^2) / 0.25^2) * X[3]]
function twist_u(X)
    a = (pi / 2) * X[3]; c, s = cos(a), sin(a)
    d = X .- CENTER
    return [c * d[1] - s * d[2] - d[1], s * d[1] + c * d[2] - d[2], 0.0]
end
function dev_strain(u, x; h = 1e-6)
    H = zeros(3, 3)
    for j in 1:3
        e = zeros(3); e[j] = h
        H[:, j] = (u(x .+ e) .- u(x .- e)) ./ (2h)
    end
    eps = 0.5 * (H + H')
    return eps - tr(eps) / 3 * I
end
function flow_field(u, peak_x; frac = 0.35)
    peak = norm(dev_strain(u, peak_x))
    return x -> begin
        d = dev_strain(u, x)
        s = norm(d)
        s > frac * peak ? (d ./ s, 1.0) : (zeros(3, 3) .+ [1 0 0; 0 -1 0; 0 0 0] ./ sqrt(2), 0.0)
    end
end
const FIELDS = (("punch", flow_field(punch, [0.5, 0.5, 1.0])),
                ("twist 90", flow_field(twist_u, [0.9, 0.5, 0.5])))

const VARIANTS = (("CT, volume-averaged J (P0)", 1), ("CT, pointwise J (P1disc)", 4))

fmt(x) = string(round(x, sigdigits = 3))
cell(v, i) = i <= length(v) ? fmt(v[i]) : "--"

function main()
    println("Check 1: comp12_grad equals the L2 projection of the subtetrahedron gradients")
    println("         onto linear functions; largest difference ", fmt(check_projection()), "\n")

    println("Check 2: with element = :p2 the harness reproduces assemble_all (P2/P0, N = 3)")
    coords, conn = mesh_of(3, 2)
    for (label, flow) in (("elastic", nothing), ("axial, beta = 1", N_AXIAL))
        a = assemble_ct(coords, conn, 1; element = :p2, flow, beta = 1.0)
        b = assemble_all(coords, conn, 2, 1; flow, beta = flow === nothing ? 0.0 : 1.0)
        d = maximum(maximum(abs.(x - y)) for (x, y) in ((a.Kdev, b.Kdev), (a.Kh1, b.Kh1),
                                                       (a.Kdiv, b.Kdiv), (a.G, b.G)))
        println("         $label: largest entry difference of Kdev, Kh1, Kdiv, G = ", fmt(d))
    end
    ap = assemble_ct(coords, conn, 1; element = :p2, flow = N_AXIAL, beta = 1.0)
    _, vp, _ = kernel_modes(ap, ap.Kdev ./ 2)
    println("         P2/P0, N = 3, axial, beta = 1: r_v of the three lowest plastic modes ",
            join(fmt.([dilatation(ap, vp[:, i]) for i in 1:3]), ", "),
            " (plastic_out.txt: 0.208, 0.195, 0.218)\n")

    println("Uniform flow, beta = 1, all-Dirichlet Freudenthal cube, mu = 1.  lam_i: i-th")
    println("generalized eigenvalue of the deviatoric form on ker G_h against the H1")
    println("seminorm of the field; r_v: dilatation of the field; r_v~: of the projected")
    println("gradient.  n_spur/20 counts the twenty lowest modes with r_v > $(RV_SPURIOUS).\n")
    for (fname, flow) in (("axial  n = dev(e3 e3)/|.|", N_AXIAL), ("shear  n = sym(e1 e2)/sqrt2", N_SHEAR))
        println("flow: $fname")
        println(rpad("pair", 29), rpad("N", 3), rpad("dim ker", 8), rpad("lam1 el", 9),
                rpad("lam1 pl", 9), rpad("lam2 pl", 9), rpad("lam3 pl", 9),
                rpad("r_v pl1", 9), rpad("r_v pl2", 9), rpad("r_v pl3", 9),
                rpad("r_v~ pl1", 9), "n_spur/20")
        println("-"^122)
        for (label, m) in VARIANTS, N in 2:5
            coords, conn = mesh_of(N, 2)
            ae = assemble_ct(coords, conn, m)
            ap = assemble_ct(coords, conn, m; flow, beta = 1.0)
            le, _, kd = kernel_modes(ae, ae.Kdev ./ 2; k = 1)
            lp, vp, _ = kernel_modes(ap, ap.Kdev ./ 2)
            rv = [dilatation(ae, vp[:, i]) for i in 1:length(lp)]
            rvp = [dilatation_proj(ae, vp[:, i]) for i in 1:length(lp)]
            println(rpad(label, 29), rpad(string(N), 3), rpad(string(kd), 8), rpad(cell(le, 1), 9),
                    rpad(cell(lp, 1), 9), rpad(cell(lp, 2), 9), rpad(cell(lp, 3), 9),
                    rpad(cell(rv, 1), 9), rpad(cell(rv, 2), 9), rpad(cell(rv, 3), 9),
                    rpad(cell(rvp, 1), 9), string(spurious(rv)))
            flush(stdout)
        end
        println()
    end

    println("Confined plastic zone with varying flow direction (plastic_zone.jl fields),")
    println("whole boundary fixed.  n_spur/20 as above.\n")
    for (fname, flow) in FIELDS
        println("field: $fname")
        println(rpad("pair", 29), rpad("N", 3), rpad("lam1 pl", 9), rpad("r_v pl1", 9),
                rpad("r_v pl2", 9), rpad("r_v pl3", 9), "n_spur/20")
        println("-"^80)
        for (label, m) in VARIANTS, N in 3:5
            coords, conn = mesh_of(N, 2)
            ae = assemble_ct(coords, conn, m)
            ap = assemble_ct(coords, conn, m; flow)
            lp, vp, _ = kernel_modes(ap, ap.Kdev ./ 2)
            rv = [dilatation(ae, vp[:, i]) for i in 1:length(lp)]
            println(rpad(label, 29), rpad(string(N), 3), rpad(cell(lp, 1), 9),
                    rpad(cell(rv, 1), 9), rpad(cell(rv, 2), 9), rpad(cell(rv, 3), 9),
                    string(spurious(rv)))
            flush(stdout)
        end
        println()
    end
end

main()
