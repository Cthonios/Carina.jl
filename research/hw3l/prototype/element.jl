# One enriched element, in either basis, under any quadrature rule.
# Shared by quadrature.jl and basis.jl.

include("common.jl")

"Straight reference TETRA10 coordinates (3 x 10), optionally with midsides displaced."
function tet10_coords(; distort = 0.0)
    v = [0.0 1 0 0; 0 0 1 0; 0 0 0 1]
    X = zeros(NSD, 10)
    X[:, 1:4] = v
    for (k, (a, b)) in enumerate(_TET_EDGES)
        X[:, 4 + k] = 0.5 .* (v[:, a] .+ v[:, b])
    end
    for k in 5:10
        X[1, k] += distort * sin(2.3k); X[2, k] += distort * cos(1.7k); X[3, k] += distort * sin(1.1k)
    end
    return X
end

"A Kuhn tetrahedron of the Freudenthal mesh, as TETRA10 (3 x 10)."
function kuhn_coords()
    c, n = mesh_of(1, 2)
    return c[:, n[:, 1]]
end

"""
The 14-point degree-5 symmetric rule with positive weights
(Keast 1986), verified in quadrature.jl to integrate every monomial
through degree 5 to 4e-16 and to fail at degree 6.
"""
function keast14()
    a1, w1 = 0.0927352503108912, 0.01224884051939366
    a2, w2 = 0.3108859192633006, 0.01878132095300264
    b,  w3 = 0.4544962958743503, 0.007091003462846911
    pts = Vector{Vector{Float64}}(); wts = Float64[]
    for (a, w) in ((a1, w1), (a2, w2))
        c = 1 - 3a
        for L in ([c, a, a, a], [a, c, a, a], [a, a, c, a], [a, a, a, c])
            push!(pts, L[2:4]); push!(wts, w)
        end
    end
    c = 0.5 - b
    for L in ([b, b, c, c], [b, c, b, c], [b, c, c, b], [c, b, b, c], [c, b, c, b], [c, c, b, b])
        push!(pts, L[2:4]); push!(wts, w3)
    end
    return reduce(hcat, pts), wts
end

"Worst relative error of a rule on the monomials through `deg`."
function monomial_error(rule, deg)
    P, W = rule
    worst = 0.0
    for d in 0:deg, a in 0:d, b in 0:(d - a)
        c = d - a - b
        exact = factorial(a) * factorial(b) * factorial(c) / factorial(d + 3)
        num = sum(W[q] * P[1, q]^a * P[2, q]^b * P[3, q]^c for q in eachindex(W))
        worst = max(worst, abs(num - exact) / exact)
    end
    return worst
end

# Quadrature rules by name.
function rule(name)
    name isa Int && return tet_rule(name)
    name == :keast14 && return keast14()
    el = ref_element(2)
    if name == :rfe2 || name == :rfe3
        ref = RFE.ReferenceFE(el, RFE.GaussLegendre(name == :rfe2 ? 2 : 3))
        nq = RFE.num_cell_quadrature_points(ref)
        pts = reduce(hcat, (collect(RFE.cell_quadrature_point(ref, q)) for q in 1:nq))
        wts = [RFE.cell_quadrature_weight(ref, q) for q in 1:nq]
        return pts, wts
    end
    error("unknown rule $name")
end

"Voigt isotropic tangent with engineering shear: lambda m m' + 2 mu W."
function voigt_tangent(kappa, mu)
    lam = kappa - 2mu / 3
    m = [1.0, 1, 1, 0, 0, 0]
    return lam * (m * m') + 2mu * Matrix(Diagonal([1.0, 1, 1, 0.5, 0.5, 0.5]))
end

"""
Element operators for the enriched element on geometry X (3 x 10) with the
given rule, in the hierarchical basis (A = I) or the nodal one (A =
nodal_transform()).  Returns the deviatoric stiffness, the condensed
three-field volumetric stiffness kappa G' M^-1 G with a P1disc pressure, the
consistent mass rho int N N', and the displacement-element stiffness for the
same material for reference.
"""
function element_ops(X, (qpts, qwts); A = Matrix{Float64}(I, 15, 15),
                     mu = 1.0, kappa = 2.0, rho = 1.0, nfun = 15)
    el = ref_element(2)
    nd = NSD * nfun
    Kdev = zeros(nd, nd); Kdisp = zeros(nd, nd); Mu = zeros(nd, nd)
    G = zeros(4, nd); Mp = zeros(4, 4)
    D = voigt_tangent(kappa, mu)
    vol = 0.0
    for q in axes(qpts, 2)
        xi = qpts[:, q]
        N, dNr = enriched_shape(xi)
        N = (A * N)[1:nfun]; dNr = (A * dNr)[1:nfun, :]
        J = X * Matrix(RFE.shape_function_gradient(el, xi))
        dV = qwts[q] * det(J)
        vol += dV
        dN = dNr / J
        B = bmatrix(dN, nfun)
        Bd = _DEV * B
        Kdev .+= dV * 2mu * (Bd' * _VOIGT_W * Bd)
        Kdisp .+= dV * (B' * D * B)
        phi = collect(pressure_basis(Val(4), xi))
        trB = B[1, :] + B[2, :] + B[3, :]
        G .+= dV * (phi * trB'); Mp .+= dV * (phi * phi')
        for a in 1:nfun, b in 1:nfun, d in 1:NSD
            Mu[NSD * (a - 1) + d, NSD * (b - 1) + d] += dV * rho * N[a] * N[b]
        end
    end
    Kvol = kappa * (G' * (Mp \ G))
    return (; Kdev, Kvol, K = Kdev + Kvol, Kdisp, Mu, G, Mp, vol)
end

"Eigenvalues below `rtol` times the largest, counted."
nzero(K; rtol = 1e-10) = count(<(rtol * maximum(abs, eigvals(Symmetric(K)))), eigvals(Symmetric(K)))
