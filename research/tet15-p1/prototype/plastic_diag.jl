# Diagnosis of dilatational kernel modes on perturbed meshes (referee round 1,
# follow-up of plastic_jitter.jl).  For each case: the twenty lowest modes of
# the deviatoric form on ker G under the uniaxial perfectly plastic tangent,
# the number with r_v > 0.1, the eigenvalue and r_v of the lowest such mode,
# the share of its int (div v)^2 carried by its three largest elements, and
# the smallest shape quality among those three elements and over the mesh.
#
#   julia --project=. research/tet15-p1/prototype/plastic_diag.jl
ENV["PE_NOMAIN"] = "1"; ENV["KS_NOMAIN"] = "1"
include("plastic_ext.jl")
include("kappa_sweep.jl")
using Random

function perturbed(N, delta, seed)
    coords, conn = mesh_of(N, 2)
    rng = MersenneTwister(seed)
    X = copy(coords); h = 1 / N
    if delta > 0
        for i in 1:(N + 1)^3
            x = coords[:, i]
            any(t -> t < 1e-12 || t > 1 - 1e-12, x) && continue
            X[:, i] = x .+ delta * h .* (2 .* rand(rng, 3) .- 1)
        end
        for e in axes(conn, 2), (k, (a, b)) in enumerate(_TET_EDGES)
            X[:, conn[4 + k, e]] = 0.5 .* (X[:, conn[a, e]] .+ X[:, conn[b, e]])
        end
    end
    return coords, X, conn
end

function quality(X, conn, e)
    P = X[:, conn[1:4, e]]
    V = det([P[:, 2] - P[:, 1]  P[:, 3] - P[:, 1]  P[:, 4] - P[:, 1]]) / 6
    l2 = sum(sum(abs2, P[:, a] - P[:, b]) for (a, b) in _TET_EDGES) / 6
    return 6 * sqrt(2) * V / l2^1.5
end

"int (div v)^2 per element."
function elem_div2(X, conn, c0, bubble, nu_nodal, v)
    _, gmap = free_dofs(c0)
    nelem = size(conn, 2)
    _, fmap = bubble in (:face, :full) ? face_table(c0, conn, _BC_ALL) : (0, zeros(Int, 4, nelem))
    el = ref_element(2); qp, qw = tet_rule(6)
    out = zeros(nelem)
    for e in 1:nelem
        Xe = X[:, conn[:, e]]
        rows = element_rows(conn, e, gmap, bubble, nelem, nu_nodal, fmap)
        ue = [r == 0 ? 0.0 : v[r] for r in rows]
        for q in eachindex(qw)
            xi = qp[:, q]
            Jm = Xe * RFE.shape_function_gradient(el, xi)
            _, dNr = shapes(bubble, xi); dN = dNr / Jm
            d = sum(ue[NSD * (A - 1) + k] * dN[A, k] for A in axes(dN, 1), k in 1:NSD)
            out[e] += qw[q] * det(Jm) * d^2
        end
    end
    return out
end

const DCASES = (("P2+int+face/P1disc", 4, :full), ("P2/P0", 1, :none), ("P2+face/P0", 1, :face))

println("uniaxial flow, beta = 1; twenty lowest modes on ker G")
println(rpad("pair", 20), rpad("delta", 6), rpad("seed", 5), rpad("N", 3), rpad("q_min", 8),
        rpad("lam1", 8), rpad("lam20", 8), rpad("n_dil", 6), rpad("lam_dil", 9), rpad("rv_dil", 8),
        rpad("top3", 7), "q_top3")
for (label, m, bubble) in DCASES, delta in (0.0, 0.1, 0.25), seed in (delta == 0 ? (0,) : (1, 2)), N in (4, 5, 6)
    c0, X, conn = perturbed(N, delta, seed)
    qmin = minimum(quality(X, conn, e) for e in axes(conn, 2))
    ap = assemble_all(X, conn, 2, m; bubble, flow = N_AXIAL, beta = 1.0, bc_coords = c0)
    lp, vp, nit = kernel_modes(ap, ap.Kdev ./ 2; k = 20, b = 40)
    rv = [dilatation(ap, vp[:, i]) for i in 1:length(lp)]
    dil = findall(>(0.1), rv)
    if isempty(dil)
        s = rpad("-", 9) * rpad("-", 8) * rpad("-", 7) * "-"
    else
        i = dil[1]
        ed = elem_div2(X, conn, c0, bubble, ap.nu_nodal, vp[:, i])
        p = sortperm(ed, rev = true)[1:3]
        s = rpad(string(round(lp[i], sigdigits = 3)), 9) * rpad(string(round(rv[i], sigdigits = 3)), 8) *
            rpad(string(round(sum(ed[p]) / sum(ed), sigdigits = 2)), 7) *
            string(round(minimum(quality(X, conn, e) for e in p), sigdigits = 3))
    end
    println(rpad(label, 20), rpad(string(delta), 6), rpad(string(seed), 5), rpad(string(N), 3),
            rpad(string(round(qmin, sigdigits = 3)), 8), rpad(string(round(lp[1], sigdigits = 3)), 8),
            rpad(string(round(lp[end], sigdigits = 3)), 8), rpad(string(length(dil)), 6), s)
    flush(stdout)
end
