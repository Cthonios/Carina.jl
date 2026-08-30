# Soft-mode spectrum of the three-field Hu-Washizu tetrahedron.
#
# For linear response the three fields condense exactly.  With Q_e the
# element-local pressure space of dimension m,
#
#     K_e = K_dev + kappa * G' * inv(M) * G,
#
#     M[m,n] = int mu_m mu_n,     G[m,i] = int mu_m (tr B)_i,
#
# so the volumetric stiffness has rank exactly m.  A P2 displacement produces a
# volumetric strain tr(eps(u)) that is a P1 field -- 4-dimensional.  A P0
# pressure therefore constrains one of those four directions and leaves three
# unconstrained: those three are near-isochoric, cost only deviatoric energy,
# and are the soft modes reported for constant-pressure formulations.
#
# The prediction this script tests: 3 soft modes per element for P0, none for
# P1-discontinuous.  A mode is "soft" when its energy fails to grow with kappa.

using LinearAlgebra
import ReferenceFiniteElements as RFE

const NSD = 3

"Reference P2 tetrahedron, mildly distorted so the test is not special-cased."
function tet10_coords(; distort = 0.0)
    v = [0.0 0.0 0.0; 1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    X = zeros(10, NSD)
    X[1:4, :] = v
    edges = ((1,2),(2,3),(3,1),(1,4),(2,4),(3,4))
    for (k, (a, b)) in enumerate(edges)
        X[4+k, :] = 0.5 * (v[a, :] + v[b, :])
    end
    if distort != 0.0
        # Push midside nodes off the straight edges; a real mesh is never ideal.
        for k in 5:10
            X[k, 1] += distort * sin(2.3k)
            X[k, 2] += distort * cos(1.7k)
            X[k, 3] += distort * sin(1.1k)
        end
    end
    return X
end

"Small-strain B operator (6 x 30, Voigt) and dV at each quadrature point."
function element_kinematics(X; q_degree = 2)
    ref = RFE.ReferenceFE(RFE.Tet{RFE.Lagrange, 2}(), RFE.GaussLegendre(q_degree))
    nqp = RFE.num_cell_quadrature_points(ref)
    Bs, dVs, xis = Matrix{Float64}[], Float64[], Vector{Float64}[]
    for q in 1:nqp
        dN_ref = RFE.cell_shape_function_gradient(ref, q)   # 10 x 3
        w      = RFE.cell_quadrature_weight(ref, q)
        xi     = RFE.cell_quadrature_point(ref, q)
        J      = X' * dN_ref                                # 3 x 3
        detJ   = det(J)
        detJ > 0 || error("non-positive Jacobian: $detJ")
        dN     = dN_ref / J                                 # 10 x 3, d/dx
        B = zeros(6, 10 * NSD)
        for a in 1:10
            c = NSD * (a - 1)
            B[1, c+1] = dN[a, 1]
            B[2, c+2] = dN[a, 2]
            B[3, c+3] = dN[a, 3]
            B[4, c+1] = dN[a, 2];  B[4, c+2] = dN[a, 1]
            B[5, c+2] = dN[a, 3];  B[5, c+3] = dN[a, 2]
            B[6, c+1] = dN[a, 3];  B[6, c+3] = dN[a, 1]
        end
        push!(Bs, B); push!(dVs, w * detJ); push!(xis, collect(xi))
    end
    return Bs, dVs, xis
end

"Monomial basis of the element-local pressure space, evaluated at xi."
pressure_basis(::Val{1}, xi) = [1.0]                                  # P0
pressure_basis(::Val{4}, xi) = [1.0, xi[1], xi[2], xi[3]]             # P1 disc
pressure_basis(::Val{10}, xi) = [1.0, xi[1], xi[2], xi[3],            # P2 disc
                                 xi[1]^2, xi[2]^2, xi[3]^2,
                                 xi[1]*xi[2], xi[2]*xi[3], xi[3]*xi[1]]

"""
Condensed element stiffness for the three-field element with pressure-space
dimension `m`.  `m = 1` is the constant-pressure formulation; `m = 4` is
P1-discontinuous.
"""
function element_stiffness(X, mu, kappa, m::Int; q_degree = 2)
    Bs, dVs, xis = element_kinematics(X; q_degree)
    ndof = 10 * NSD

    # Deviatoric block, evaluated pointwise.  In Voigt form the deviatoric
    # projector must weight the shear rows by 2 to keep the energy correct.
    Idev = Matrix{Float64}(I, 6, 6)
    for i in 1:3, j in 1:3
        Idev[i, j] -= 1/3
    end
    W = Diagonal([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])   # Voigt energy metric

    K_dev = zeros(ndof, ndof)
    M = zeros(m, m)
    G = zeros(m, ndof)
    for (B, dV, xi) in zip(Bs, dVs, xis)
        Bd = Idev * B
        K_dev .+= dV * 2mu * (Bd' * W * Bd)
        phi = pressure_basis(Val(m), xi)
        M .+= dV * (phi * phi')
        trB = B[1, :] + B[2, :] + B[3, :]
        G .+= dV * (phi * trB')
    end
    K_vol = kappa * (G' * (M \ G))
    return K_dev + K_vol, K_dev, K_vol
end

"""
Census of the element's mode structure.

The mechanism is a rank count.  `K_dev` annihilates every mode with zero
deviatoric strain at quadrature: the six rigid-body modes plus the volumetric
directions the displacement space can produce.  `K_vol = kappa G' inv(M) G` has
rank equal to the number of those volumetric directions the pressure space can
see, which is at most m.  Whatever is left over is a mode with no deviatoric
energy and no volumetric energy -- an exactly zero-energy spurious mode.

    spurious = dim null(K_dev) - 6 - rank(K_vol)

That is the whole story of the constant-pressure soft mode, and it is a
statement about ranks, not about materials or magnitudes.
"""
function census(X, mu, kappa, m; q_degree = 2)
    Bs, dVs, xis = element_kinematics(X; q_degree)
    ndof = 10 * NSD

    T = zeros(length(Bs), ndof)
    for (q, B) in enumerate(Bs)
        T[q, :] = B[1, :] + B[2, :] + B[3, :]
    end

    K, K_dev, K_vol = element_stiffness(X, mu, kappa, m; q_degree)
    rk_dev  = rank(K_dev;  rtol = 1e-10)
    rk_vol  = rank(K_vol;  rtol = 1e-10)
    null_dev = ndof - rk_dev
    predicted_spurious = null_dev - 6 - rk_vol

    ev = eigvals(Symmetric(K))
    # Zero-energy modes measured against the deviatoric scale, not the
    # kappa-inflated maximum: a mode costing O(mu) must not be counted as zero.
    dev_scale = maximum(abs, diag(K_dev))
    observed_zero = count(e -> abs(e) < 1e-9 * dev_scale, ev)
    observed_spurious = observed_zero - 6

    return (; rank_T = rank(T; rtol = 1e-10), rk_dev, null_dev, rk_vol,
              predicted_spurious, observed_spurious,
              first_nonzero = ev[findfirst(e -> abs(e) >= 1e-9 * dev_scale, ev)])
end

function main()
    mu, kappa = 1.0, 1.0e4
    println("Three-field HW tetrahedron: spurious-mode census")
    println("P2 displacement, 30 dof, mu = $mu, kappa = $kappa\n")
    println("  null(K_dev) = 6 rigid + (volumetric directions at quadrature)")
    println("  rank(K_vol) = volumetric directions the pressure space constrains")
    println("  spurious    = null(K_dev) - 6 - rank(K_vol)\n")
    hdr = (rpad("distort", 9), rpad("pressure", 20), rpad("m", 5),
           rpad("null Kdev", 11), rpad("rank Kvol", 11),
           rpad("predicted", 11), rpad("observed", 10), "1st nonzero")
    println(hdr...)
    println("-"^92)
    ok = true
    for distort in (0.0, 0.05, 0.12)
        for (label, m) in (("P0 (constant)", 1), ("P1 (linear disc)", 4),
                           ("P2 (quadratic disc)", 10))
            X = tet10_coords(; distort)
            c = census(X, mu, kappa, m)
            c.predicted_spurious == c.observed_spurious || (ok = false)
            println(rpad(string(distort), 9), rpad(label, 20), rpad(string(m), 5),
                    rpad(string(c.null_dev), 11), rpad(string(c.rk_vol), 11),
                    rpad(string(c.predicted_spurious), 11),
                    rpad(string(c.observed_spurious), 10),
                    string(round(c.first_nonzero, sigdigits = 3)))
        end
    end
    println()
    if ok
        println("PASS: the rank prediction matches the measured spectrum in every case.")
    else
        println("FAIL: predicted and observed spurious-mode counts disagree.")
    end
    println()
    println("Reading: the constant-pressure element carries three spurious modes")
    println("that are EXACTLY zero-energy, not merely soft -- no deviatoric energy")
    println("and no volumetric energy.  That is why a penalty is unavoidable there.")
    println("A P1-discontinuous pressure constrains all four volumetric directions")
    println("and leaves the six rigid-body modes alone.")
end

main()
