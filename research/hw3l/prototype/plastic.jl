# Soft modes under a plastic tangent: which pairs have physical ones.
#
# Brezzi's second hypothesis asks that the deviatoric form be coercive on
# ker G_h, the displacements the discrete pressure cannot see.  Under plastic
# loading the consistent J2 tangent
#
#     C_ep = 2 mu [ I_dev - beta n (x) n ],     beta = 1 for perfect plasticity,
#
# removes the shear stiffness along the flow direction n, and the first
# version of this script measured the coercivity constant
#
#     lambda_min = min_{v in ker G_h}  (1/2mu) a_ep(v, v) / |v|_1^2
#
# expecting it to separate a stable pair from a constant-pressure one.  It
# cannot, for a reason that is exact.  For every v in H^1_0,
#
#     int |dev eps|^2 = (1/2) |grad v|^2 + (1/6) int (div v)^2  >=  (1/2) |grad v|^2,
#
# so (1/2mu) a_ep(v, v) >= (1 - beta)/2 |v|_1^2 for EVERY conforming pair and
# every kernel, isochoric or not: for beta < 1 the constant is bounded below
# by (1 - beta)/2 uniformly, and for beta = 1 the continuum operator itself
# loses strong ellipticity (a shear band with normal e2 and jump e1 costs
# nothing), so the constant tends to zero for every pair, Taylor-Hood
# included.  The hypothesis, read as a constant, does not distinguish pairs.
#
# What distinguishes them is what the soft directions ARE.  A mode in ker G_h
# that is nearly isochoric is a shear band the continuum also has, and its
# low energy is physical.  A mode with pointwise nonzero dilatation of zero
# element mean is one the continuum would charge kappa int (div v)^2 for and
# the element charges nothing: when the tangent removes its shear energy it
# is free in the element and stiff in the physics, which is the soft mode
# reported for constant-pressure elements.  The measurement is therefore the
# dilatation carried by the lowest plastic eigenmodes,
#
#     r_v(v) = int (div v)^2 / |grad v|^2 ,
#
# on the same kernel eigenproblem as before.  Prediction: r_v of the soft
# modes is O(h^2) for the stable pair with a linear discontinuous pressure and
# O(1) for a constant pressure.
#
#   julia --project=. research/hw3l/prototype/plastic.jl

include("common.jl")

const N_SHEAR = let n = zeros(3, 3); n[1, 2] = n[2, 1] = 1 / sqrt(2); n end
const N_AXIAL = let n = zeros(3, 3); n[1, 1] = n[2, 2] = -1 / sqrt(6); n[3, 3] = 2 / sqrt(6); n end

const PAIRS = (("P2/P0",              1,     :none, 5),
               ("P2+face/P0",         1,     :face, 5),
               ("P2/P1disc",          4,     :none, 4),
               ("P2+int+face/P1disc", 4,     :full, 5),
               ("P2/P1c (TH)",        _P1C,  :none, 5))
const BETAS = (1.0, 0.9)
const NMODES = 3

"Lowest generalized eigenpairs of K on ker G against the H1 seminorm."
function kernel_modes(a, K; k = NMODES)
    Z = nullspace(Matrix(a.G); rtol = 1e-10)
    size(Z, 2) == 0 && return Float64[], zeros(a.nu, 0)
    A = Symmetric(Z' * Matrix(K) * Z)
    B = Symmetric(Z' * Matrix(a.Kh1) * Z)
    F = eigen(A, B)
    kk = min(k, length(F.values))
    return F.values[1:kk], Z * F.vectors[:, 1:kk]
end

dilatation(a, v) = dot(v, a.Kdiv * v) / dot(v, a.Kh1 * v)

function run_case(m, bubble, N, flow, beta)
    coords, conn = mesh_of(N, 2)
    nvert = m == _P1C ? size(coords, 2) : 0
    ae = assemble_all(coords, conn, 2, m; bubble, nvert)
    ap = assemble_all(coords, conn, 2, m; bubble, nvert, flow, beta)
    ae.nu == 0 && return nothing
    le, ve = kernel_modes(ae, ae.Kdev ./ 2)
    lp, vp = kernel_modes(ap, ap.Kdev ./ 2)
    isempty(lp) && return nothing
    return (; nu = ae.nu, kdim = size(nullspace(Matrix(ae.G); rtol = 1e-10), 2),
              le, lp,
              rve = [dilatation(ae, ve[:, i]) for i in 1:length(le)],
              rvp = [dilatation(ae, vp[:, i]) for i in 1:length(lp)])
end

fmt(x) = string(round(x, sigdigits = 3))
cell(v, i) = i <= length(v) ? fmt(v[i]) : "--"

function main()
    println("Lowest modes of the deviatoric form on ker G_h against the H1 seminorm,")
    println("all-Dirichlet Freudenthal cube, mu = 1, elastic (beta = 0) and plastic")
    println("along a uniform flow direction.  lam_i is the i-th generalized")
    println("eigenvalue; r_v,i = int (div v_i)^2 / |grad v_i|^2 is the dilatation")
    println("the mode carries, zero for an isochoric field.  For beta < 1 every")
    println("lam_i >= (1 - beta)/2 exactly, for every pair; the discriminating column")
    println("is r_v of the plastic modes.\n")
    for beta in BETAS, (fname, flow) in (("shear  n = sym(e1 e2)/sqrt2", N_SHEAR),
                                          ("axial  n = dev(e3 e3)/|.|", N_AXIAL))
        println("beta = $beta   floor (1-beta)/2 = $(fmt((1 - beta) / 2))   flow: $fname")
        println(rpad("pair", 21), rpad("N", 3), rpad("dim ker", 8),
                rpad("lam1 el", 9), rpad("r_v el1", 9),
                rpad("lam1 pl", 9), rpad("lam2 pl", 9), rpad("lam3 pl", 9),
                rpad("r_v pl1", 9), rpad("r_v pl2", 9), "r_v pl3")
        println("-"^104)
        for (label, m, bubble, Nmax) in PAIRS
            for N in 2:Nmax
                r = run_case(m, bubble, N, flow, beta)
                r === nothing && continue
                println(rpad(label, 21), rpad(string(N), 3), rpad(string(r.kdim), 8),
                        rpad(cell(r.le, 1), 9), rpad(cell(r.rve, 1), 9),
                        rpad(cell(r.lp, 1), 9), rpad(cell(r.lp, 2), 9), rpad(cell(r.lp, 3), 9),
                        rpad(cell(r.rvp, 1), 9), rpad(cell(r.rvp, 2), 9), cell(r.rvp, 3))
                flush(stdout)
            end
        end
        println()
    end
end

main()
