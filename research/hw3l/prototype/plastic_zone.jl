# Soft modes under a plastic tangent with a spatially varying flow direction.
#
# plastic.jl uses a uniform flow direction, which is the cleanest test of the
# claim and not what a plastic zone looks like.  Here the flow direction and
# the plastic zone come from a prescribed displacement field u(X): at each
# quadrature point n = dev eps(u) / |dev eps(u)| and beta = 1 where |dev eps|
# exceeds a threshold, beta = 0 elsewhere, so the tangent is the consistent
# perfectly plastic tangent inside a confined zone and elastic outside it.
# Two fields:
#   punch   u3 = -exp(-r^2 / w^2) X3, r the distance from the vertical axis:
#           a zone under an indenter on the top face, elastic surroundings
#   twist   u = twist_90(X) - X: torsion, plastic except near the axis
# The measured quantities are those of plastic.jl: the lowest generalized
# eigenvalues of the deviatoric form on ker G_h against the H1 seminorm and
# the dilatation r_v each mode carries.  The prediction is unchanged.
#
#   julia --project=. research/hw3l/prototype/plastic_zone.jl

include("common.jl")

const CENTER = [0.5, 0.5, 0.5]

punch(X) = [0.0, 0.0, -exp(-((X[1] - 0.5)^2 + (X[2] - 0.5)^2) / 0.25^2) * X[3]]
function twist_u(X)
    a = (pi / 2) * X[3]; c, s = cos(a), sin(a)
    d = X .- CENTER
    return [c * d[1] - s * d[2] - d[1], s * d[1] + c * d[2] - d[2], 0.0]
end

"Deviatoric small strain of the field `u` at `x`, by central differences."
function dev_strain(u, x; h = 1e-6)
    H = zeros(3, 3)
    for j in 1:3
        e = zeros(3); e[j] = h
        H[:, j] = (u(x .+ e) .- u(x .- e)) ./ (2h)
    end
    eps = 0.5 * (H + H')
    return eps - tr(eps) / 3 * I
end

"Flow field: (n, beta) at x, plastic where |dev eps| exceeds `frac` of `peak`."
function flow_field(u, peak_x; frac = 0.35)
    peak = norm(dev_strain(u, peak_x))
    return x -> begin
        d = dev_strain(u, x)
        s = norm(d)
        s > frac * peak ? (d ./ s, 1.0) : (zeros(3, 3) .+ [1 0 0; 0 -1 0; 0 0 0] ./ sqrt(2), 0.0)
    end
end

const FIELDS = (("punch (zone under an indenter)", flow_field(punch, [0.5, 0.5, 1.0])),
                ("twist 90 (torsion)",             flow_field(twist_u, [0.9, 0.5, 0.5])))

const PAIRS = (("P2/P0",              1,     :none, 5),
               ("P2+face/P0",         1,     :face, 5),
               ("P2+int+face/P1disc", 4,     :full, 5),
               ("P2/P1c (TH)",        _P1C,  :none, 5))
# Eigenpairs kept per case.  The three lowest are printed; the count of
# spurious modes is taken over all twenty, because in a plastic zone with a
# varying flow direction physical shear bands can lie below the spurious
# dilatational modes, which are then invisible to the lowest three alone.
const NMODES = 20
# A mode is counted as spurious when its dilatation exceeds this: the stable
# pair's soft modes sit below 0.02 from N = 4 on, the constant-pressure
# pairs' spurious ones at 0.2-0.4.
const RV_SPURIOUS = 0.1

function kernel_modes(a, K; k = NMODES)
    Z = nullspace(Matrix(a.G); rtol = 1e-10)
    size(Z, 2) == 0 && return Float64[], zeros(a.nu, 0)
    F = eigen(Symmetric(Z' * Matrix(K) * Z), Symmetric(Z' * Matrix(a.Kh1) * Z))
    kk = min(k, length(F.values))
    return F.values[1:kk], Z * F.vectors[:, 1:kk]
end
dilatation(a, v) = dot(v, a.Kdiv * v) / dot(v, a.Kh1 * v)

"Number of spurious modes among the kept ones, and the index of the first."
function spurious(rv)
    idx = findall(>(RV_SPURIOUS), rv)
    return length(idx), isempty(idx) ? 0 : first(idx)
end

"Fraction of the volume in the plastic zone, from the quadrature points."
function plastic_fraction(flow, coords, conn)
    el = ref_element(2); qpts, qwts = tet_rule(2)
    vol = 0.0; pl = 0.0
    for e in axes(conn, 2), q in axes(qpts, 2)
        X = coords[:, conn[:, e]]
        w = qwts[q] * det(X * RFE.shape_function_gradient(el, qpts[:, q]))
        vol += w
        pl += w * flow(X * collect(RFE.shape_function_value(el, qpts[:, q])))[2]
    end
    return pl / vol
end

fmt(x) = string(round(x, sigdigits = 3))
cell(v, i) = i <= length(v) ? fmt(v[i]) : "--"

function main()
    println("Lowest modes of the deviatoric form on ker G_h against the H1 seminorm")
    println("under a perfectly plastic tangent inside a confined zone (beta = 1 where")
    println("|dev eps(u)| > 0.35 of its peak, elastic elsewhere), flow direction")
    println("n = dev eps(u)/|dev eps(u)| from the prescribed field u.  r_v is the")
    println("dilatation of the mode; `plastic` is the volume fraction of the zone.")
    println("n_spur/20 counts the modes among the twenty lowest with r_v > $(RV_SPURIOUS),")
    println("and `first` is the index of the first such mode (0 if none).\n")
    for (fname, flow) in FIELDS, bc in (_BC_ALL, _BC_ZFACE)
        println("field: $fname    bc: $bc")
        println(rpad("pair", 21), rpad("N", 3), rpad("plastic", 9), rpad("dim ker", 8),
                rpad("lam1 el", 9), rpad("lam1 pl", 9), rpad("lam2 pl", 9), rpad("lam3 pl", 9),
                rpad("r_v pl1", 9), rpad("r_v pl2", 9), rpad("r_v pl3", 9),
                rpad("n_spur/20", 10), "first")
        println("-"^124)
        for (label, m, bubble, Nmax) in PAIRS
            for N in 2:Nmax
                coords, conn = mesh_of(N, 2)
                nvert = m == _P1C ? (N + 1)^3 : 0
                ae = assemble_all(coords, conn, 2, m; bubble, bc, nvert)
                ap = assemble_all(coords, conn, 2, m; bubble, bc, nvert, flow)
                ae.nu == 0 && continue
                le, _ = kernel_modes(ae, ae.Kdev ./ 2; k = 1)
                lp, vp = kernel_modes(ap, ap.Kdev ./ 2)
                isempty(lp) && continue
                rv = [dilatation(ae, vp[:, i]) for i in 1:length(lp)]
                println(rpad(label, 21), rpad(string(N), 3),
                        rpad(fmt(plastic_fraction(flow, coords, conn)), 9),
                        rpad(string(size(vp, 1) == 0 ? 0 : ae.nu - rank(Matrix(ae.G); rtol = 1e-10)), 8),
                        rpad(cell(le, 1), 9), rpad(cell(lp, 1), 9), rpad(cell(lp, 2), 9),
                        rpad(cell(lp, 3), 9), rpad(cell(rv, 1), 9), rpad(cell(rv, 2), 9), rpad(cell(rv, 3), 9),
                        rpad(string(spurious(rv)[1]), 10), string(spurious(rv)[2]))
                flush(stdout)
            end
        end
        println()
    end
end

main()
