# Hierarchical bubbles or a nodal basis: conditioning, lumped mass, and the
# explicit critical time step.
#
# The enriched space can be carried by the hierarchical basis of the bench
# (ten Lagrange functions plus five bubbles) or by a nodal basis with unit
# value at fifteen nodes, the TETRA15 arrangement: vertices, edge midpoints,
# face centroids, centroid.  The space, and with it every stiffness and
# stability result, is the same; what differs is conditioning, how the mass
# lumps, and what a rigid translation looks like.  Section 4 of the note
# argued that HRZ lumping of the hierarchical basis charges the bubbles for
# mass that a translation never carries; that argument is measured here, as
# is the critical time step against TETRA4 and TETRA10 on the same element.
#
# HRZ (diagonal scaling) lumping: m_i = M_ii * (rho V) / sum_j M_jj per
# component.  Variants: over all fifteen functions; over the ten Lagrange
# functions only, the bubbles keeping their consistent diagonal (the remedy
# the note proposed); and the nodal basis.  Row-sum lumping is reported for
# TETRA10 to show why it cannot be used.
#
# Critical time step from the element bound: omega_max^2 = max eig(M_L^-1
# K), dt = 2 / omega_max, reported as the Courant number c_p dt / h with h
# the element's edge length and c_p = sqrt((kappa + 4mu/3) / rho).
#
#   julia --project=. research/hw3l/prototype/basis.jl

include("element.jl")

const MU, KAPPA, RHO = 1.0, 2.0, 1.0
const CP = sqrt((KAPPA + 4MU / 3) / RHO)

"HRZ lumping of the consistent mass over the functions in `idx` (all three components)."
function hrz(Mu, nfun, idx, total)
    # `total` is rho V per component, the element mass, which is neither the
    # trace of M (sum rho int N_a^2) nor, for the hierarchical basis, the sum
    # of its entries (the bubbles are not part of a partition of unity).
    m = diag(Mu)
    out = copy(m)
    for d in 1:NSD
        dofs = [NSD * (a - 1) + d for a in idx]
        s = sum(m[dofs])
        out[dofs] .= m[dofs] .* (total / s)
    end
    return out
end

"Row-sum lumping."
rowsum(Mu) = vec(sum(Mu, dims = 2))

"Momentum of a unit translation in x under the lumped mass, over the exact rho V."
function momentum(mL, nfun, coeff)
    t = zeros(NSD * nfun)
    for a in 1:nfun; t[NSD * (a - 1) + 1] = coeff[a]; end
    return dot(mL, t)
end

function courant(K, mL, h)
    w2 = maximum(eigvals(Symmetric(Diagonal(1 ./ sqrt.(mL)) * K * Diagonal(1 ./ sqrt.(mL)))))
    return CP * (2 / sqrt(w2)) / h
end

cond_nonrigid(K) = (ev = eigvals(Symmetric(K)); tol = 1e-10 * ev[end]; ev[end] / minimum(ev[ev .> tol]))

function report(X, label, h)
    println(label)
    qp = tet_rule(7)
    Ah = Matrix{Float64}(I, 15, 15); An = nodal_transform()
    eh = element_ops(X, qp; A = Ah, mu = MU, kappa = KAPPA, rho = RHO)
    en = element_ops(X, qp; A = An, mu = MU, kappa = KAPPA, rho = RHO)
    V = eh.vol
    println("  volume ", round(V, sigdigits = 5), ",  Courant number c_p dt / h from the element bound")
    println("  ", rpad("basis / lumping", 44), rpad("cond K", 10), rpad("cond M", 10),
            rpad("min m", 10), rpad("momentum", 10), "Courant")
    rows = []
    # Hierarchical: coefficients of a translation are 1 on Lagrange, 0 on bubbles.
    ch = vcat(ones(10), zeros(5)); cn = ones(15)
    push!(rows, ("hierarchical, consistent M", eh, nothing, ch, cond_nonrigid(eh.K), cond(eh.Mu)))
    push!(rows, ("hierarchical, HRZ over all 15", eh, hrz(eh.Mu, 15, 1:15, RHO * V), ch, nothing, nothing))
    push!(rows, ("hierarchical, HRZ over Lagrange 10", eh, hrz(eh.Mu, 15, 1:10, RHO * V), ch, nothing, nothing))
    push!(rows, ("nodal (TETRA15), consistent M", en, nothing, cn, cond_nonrigid(en.K), cond(en.Mu)))
    push!(rows, ("nodal (TETRA15), HRZ", en, hrz(en.Mu, 15, 1:15, RHO * V), cn, nothing, nothing))
    push!(rows, ("nodal (TETRA15), row sum", en, rowsum(en.Mu), cn, nothing, nothing))
    # Displacement elements for comparison: TETRA10 (first ten functions) and TETRA4.
    e10 = element_ops(X, qp; nfun = 10, mu = MU, kappa = KAPPA, rho = RHO)
    push!(rows, ("TETRA10 displacement, HRZ", e10, hrz(e10.Mu, 10, 1:10, RHO * V), ones(10), nothing, nothing))
    push!(rows, ("TETRA10 displacement, row sum", e10, rowsum(e10.Mu), ones(10), nothing, nothing))
    for (name, e, mL, c, cK, cM) in rows
        nf = length(c)
        K = nf == 15 ? e.K : e.Kdisp
        if mL === nothing
            w2 = maximum(eigvals(Symmetric(K), Symmetric(e.Mu)))
            co = CP * (2 / sqrt(w2)) / h
            println("  ", rpad(name, 44), rpad(string(round(cK, sigdigits = 3)), 10),
                    rpad(string(round(cM, sigdigits = 3)), 10), rpad("--", 10), rpad("--", 10),
                    round(co, sigdigits = 4))
        else
            mom = momentum(mL, nf, c) / (RHO * V)
            co = minimum(mL) > 0 ? courant(K, mL, h) : NaN
            println("  ", rpad(name, 44), rpad("--", 10), rpad("--", 10),
                    rpad(string(round(minimum(mL) / (RHO * V), sigdigits = 3)), 10),
                    rpad(string(round(mom, sigdigits = 4)), 10),
                    minimum(mL) > 0 ? string(round(co, sigdigits = 4)) : "undefined")
        end
    end
    # TETRA4 on the vertices.
    el1 = ref_element(1); qp1 = tet_rule(3)
    X4 = X[:, 1:4]
    K4 = zeros(12, 12); M4 = zeros(12, 12); D = voigt_tangent(KAPPA, MU)
    for q in axes(qp1[1], 2)
        xi = qp1[1][:, q]
        N = collect(RFE.shape_function_value(el1, xi)); dNr = Matrix(RFE.shape_function_gradient(el1, xi))
        J = X4 * dNr; dV = qp1[2][q] * det(J); dN = dNr / J
        B = bmatrix(dN, 4); K4 .+= dV * (B' * D * B)
        for a in 1:4, b in 1:4, d in 1:3; M4[3(a-1)+d, 3(b-1)+d] += dV * RHO * N[a] * N[b]; end
    end
    println("  ", rpad("TETRA4 displacement, row sum", 44), rpad("--", 10), rpad("--", 10),
            rpad(string(round(minimum(rowsum(M4)) / (RHO * V), sigdigits = 3)), 10), rpad("1.0", 10),
            round(courant(K4, rowsum(M4), h), sigdigits = 4))
    println()
end

function main()
    println("Basis choice for the enriched element: kappa/mu = 2 (nu = 0.286), mu = rho = 1.")
    println("`min m` is the smallest lumped mass over rho V; `momentum` is the momentum of a")
    println("unit translation under the lumped mass over its exact value rho V.\n")
    report(tet10_coords(), "reference tetrahedron, h = 1", 1.0)
    report(tet10_coords(distort = 0.06), "distorted midsides (0.06, det J from 0.55 to 1.19), h = 1", 1.0)
    report(kuhn_coords(), "Kuhn tetrahedron of the Freudenthal mesh, N = 1 (longest edge sqrt 3)", sqrt(3))
end

main()
