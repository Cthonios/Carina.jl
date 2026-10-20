# Locking measured directly: error against a known solution as kappa/mu grows.
#
# Linear isotropic elasticity on the unit cube with the whole boundary fixed,
# Freudenthal meshes, and a manufactured solution whose displacement and mean
# stress stay bounded as kappa -> infinity:
#
#     u = curl(B a) + grad(phi) / (lambda + mu),    phi = s B,
#     B = (x(1-x) y(1-y) z(1-z))^2,                 a = (1, 2, 3),
#
# so that div u = Lap(phi)/(lambda + mu), the mean stress is
# p = kappa div u -> Lap(phi), and the body force
#
#     f = -(mu Lap u + (lambda + mu) grad div u)
#       = -(mu Lap curl(B a) + mu grad Lap phi /(lambda + mu) + grad Lap phi)
#
# is bounded uniformly in kappa.  B and grad B vanish on the boundary, so u
# does.  A discretization that does not lock has errors bounded uniformly in
# kappa/mu; a locking one has errors that grow with kappa/mu at fixed h.
#
# Reported: relative H1-seminorm error of u and relative L2 error of the mean
# stress p_h (kappa times the projected dilatation for the mixed pairs, kappa
# times the pointwise dilatation for the pointwise elements).

include("common.jl")
using ForwardDiff, Printf

const AVEC = [1.0, 2.0, 3.0]
const SPHI = 3.0

Bfun(x) = (x[1] * (1 - x[1]) * x[2] * (1 - x[2]) * x[3] * (1 - x[3]))^2
gradB(x) = ForwardDiff.gradient(Bfun, x)
wfield(x) = cross(gradB(x), AVEC)                    # curl(B a) = grad B x a
uexact(x, lm) = wfield(x) .+ (SPHI / lm) .* gradB(x) # lm = lambda + mu
graduex(x, lm) = ForwardDiff.jacobian(y -> uexact(y, lm), x)
divu(x, lm) = tr(graduex(x, lm))
lapvec(fun, x) = [tr(ForwardDiff.hessian(y -> fun(y)[i], x)) for i in 1:3]
function body_force(x, mu, lm)
    lu = lapvec(y -> uexact(y, lm), x)
    gd = ForwardDiff.gradient(y -> divu(y, lm), x)
    return -(mu .* lu .+ lm .* gd)
end

"Inverse of a matrix that is block diagonal with m x m blocks in element order."
function blockinv(M, m)
    n = size(M, 1); I = Int[]; J = Int[]; V = Float64[]
    for e in 1:(n ÷ m)
        r = (m * (e - 1) + 1):(m * e)
        B = inv(Matrix(M[r, r]))
        for j in 1:m, i in 1:m
            push!(I, r[i]); push!(J, r[j]); push!(V, B[i, j])
        end
    end
    return sparse(I, J, V, n, n)
end

"Element rows in assemble_all's numbering (0 = constrained)."
function element_rows(conn, e, gmap, bubble, nelem, nu_nodal, fmap)
    nen = size(conn, 1)
    rows = [gmap[NSD * (conn[a, e] - 1) + d] for a in 1:nen for d in 1:NSD]
    if bubble in (:interior, :full)
        append!(rows, [nu_nodal + NSD * (e - 1) + d for d in 1:NSD])
    end
    if bubble in (:face, :full)
        off = nu_nodal + (bubble === :full ? NSD * nelem : 0)
        for f in 1:4
            fid = fmap[f, e]
            append!(rows, [fid == 0 ? 0 : off + NSD * (fid - 1) + d for d in 1:NSD])
        end
    end
    return rows
end

"Shape values and reference gradients in assemble_all's order."
function shapes(bubble, xi)
    el = ref_element(2)
    N  = collect(RFE.shape_function_value(el, xi))
    dN = Matrix(RFE.shape_function_gradient(el, xi))
    if bubble in (:interior, :full)
        v, g = interior_bubble(xi); push!(N, v); dN = vcat(dN, reshape(g, 1, NSD))
    end
    if bubble in (:face, :full)
        for f in 1:4
            v, g = face_bubble(f, xi); push!(N, v); dN = vcat(dN, reshape(g, 1, NSD))
        end
    end
    return N, dN
end

"""
Solve one pair on one mesh at one kappa.  `m` = 1 or 4 for P0 or P1disc,
`:pointwise` for no projection.
"""
function solve_case(N, bubble, m, kappa; mu = 1.0)
    lam = kappa - 2mu / 3
    lm  = lam + mu
    coords, conn = mesh_of(N, 2)
    mm = m === :pointwise ? 1 : m
    a = assemble_all(coords, conn, 2, mm; mu, bubble,
                     q_degree = bubble === :none ? 4 : 6)
    K = if m === :pointwise
        a.Kdev + kappa * a.Kdiv
    else
        Minv = blockinv(a.M, mm)                   # block diagonal by element
        a.Kdev + kappa * (a.G' * Minv * a.G)
    end
    _, gmap = free_dofs(coords)
    nelem = size(conn, 2)
    _, fmap = bubble in (:face, :full) ? face_table(coords, conn, _BC_ALL) :
                                         (0, zeros(Int, 4, nelem))
    el = ref_element(2)
    qp, qw = tet_rule(12)
    F = zeros(a.nu)
    for e in 1:nelem
        X = coords[:, conn[:, e]]
        rows = element_rows(conn, e, gmap, bubble, nelem, a.nu_nodal, fmap)
        for q in eachindex(qw)
            xi = qp[:, q]
            Jm = X * RFE.shape_function_gradient(el, xi)
            x  = X * collect(RFE.shape_function_value(el, xi))
            Nv, _ = shapes(bubble, xi)
            f = body_force(x, mu, lm)
            dV = qw[q] * det(Jm)
            for (i, r) in enumerate(rows)
                r == 0 && continue
                A, d = fldmod1(i, NSD)
                F[r] += dV * Nv[A] * f[d]
            end
        end
    end
    U = cholesky(Symmetric(K)) \ F
    pbar = m === :pointwise ? nothing : blockinv(a.M, mm) * (a.G * U)
    eh1 = 0.0; nh1 = 0.0; ep = 0.0; np2 = 0.0
    for e in 1:nelem
        X = coords[:, conn[:, e]]
        rows = element_rows(conn, e, gmap, bubble, nelem, a.nu_nodal, fmap)
        ue = [r == 0 ? 0.0 : U[r] for r in rows]
        for q in eachindex(qw)
            xi = qp[:, q]
            Jm = X * RFE.shape_function_gradient(el, xi)
            x  = X * collect(RFE.shape_function_value(el, xi))
            _, dNr = shapes(bubble, xi)
            dN = dNr / Jm
            Gh = zeros(3, 3)
            for A in axes(dN, 1), d in 1:NSD
                Gh[d, :] .+= ue[NSD * (A - 1) + d] .* dN[A, :]
            end
            Ge = graduex(x, lm)
            ph = if m === :pointwise
                kappa * tr(Gh)
            else
                phi = pressure_basis(Val(mm), xi)
                kappa * sum(phi[k] * pbar[mm * (e - 1) + k] for k in 1:mm)
            end
            pe = kappa * tr(Ge)
            dV = qw[q] * det(Jm)
            eh1 += dV * sum(abs2, Gh - Ge); nh1 += dV * sum(abs2, Ge)
            ep  += dV * (ph - pe)^2;        np2 += dV * pe^2
        end
    end
    return sqrt(eh1 / nh1), sqrt(ep / np2), a.nu
end

const PAIRS = (("TET10 pointwise",      :none, :pointwise),
               ("P2/P0",                :none, 1),
               ("TET15 pointwise",      :full, :pointwise),
               ("TET15-P0",             :full, 1),
               ("TET15-P1",             :full, 4))

function main(; Ns = (2, 3, 4, 6), kappas = (1.0, 1e2, 1e4, 1e6), pairs = PAIRS)
    println("Manufactured solution, unit cube, whole boundary fixed, mu = 1.")
    println("e_u: relative H1-seminorm error of u; e_p: relative L2 error of the mean stress.")
    for (name, bubble, m) in pairs
        println("\n", name)
        @printf("  %-4s %-8s", "N", "n_u")
        for k in kappas; @printf("  k/mu=%-8.0e e_u      e_p     ", k); end
        println()
        for N in Ns
            first = true
            line = ""
            nu = 0
            for k in kappas
                eu, epr, nu = solve_case(N, bubble, m, k)
                line *= @sprintf("  %-13s %-8.3e %-8.3e", "", eu, epr)
            end
            @printf("  %-4d %-8d%s\n", N, nu, line)
            flush(stdout)
        end
    end
end

if !isinteractive() && get(ENV, "KS_NOMAIN", "") != "1"
    if get(ENV, "KS_EXT", "") == "1"         # referee round 2: finer meshes
        main(; Ns = (8, 10, 12), pairs = PAIRS[[5, 4, 1]])
    else
        main()
    end
end
