# What the materials implemented in Norma satisfy (note, sec:matreq, tab:split
# and rem:thetachoice).  This script needs Norma, not Carina:
#
#     julia --project=/path/to/Norma.jl materials.jl
#
# Three measurements.
#
#   1. The split test.  If W(F) = W_vol(J) + W_iso(F_bar) with F_bar = J^-1/3 F,
#      then W(F) - W(F_bar) = W_vol(J) - W_vol(1) depends on J alone.  Holding J
#      fixed while varying the isochoric part must leave that difference
#      unmoved; the spread across isochoric parts, relative to the magnitude,
#      is machine zero exactly when the split is exact.  This decides the split
#      without knowing W_vol.
#
#   2. The quadratic fit.  W_vol(J) = W(J^{1/3} I) - W(I) is extracted along the
#      pure dilatation line and fitted to c (J-1)^2 and to c (log J)^2.  The
#      relative residual says whether W_vol is quadratic in that variable, and
#      c / kappa says whether the coefficient is the bulk modulus.
#
#   3. The choice of volumetric variable.  On one element under a non-affine
#      displacement, the three-field energy with theta = log J projected and
#      with theta = J - 1 projected, for the SAME Hencky material, so that the
#      only difference is which variable is projected.  The relative
#      difference is O(strain): both are consistent discretizations of the same
#      continuum problem and they are different elements.

using LinearAlgebra, StaticArrays, Random
using Norma

const E0, ν0 = 1.0e9, 0.3
base(model) = Dict{String,Any}("model" => model, "elastic modulus" => E0,
                               "Poisson's ratio" => ν0)
const MATERIALS = (
    ("Linear elastic",          base("linear elastic")),
    ("Saint-Venant Kirchhoff",  base("Saint-Venant Kirchhoff")),
    ("Neo-Hookean",             base("neohookean")),
    ("Reciprocal neo-Hookean",  base("r-neohookean")),
    ("Seth-Hill (m = n = 1)",   merge(base("seth-hill"), Dict("m" => 1, "n" => 1))),
    ("Hencky",                  base("hencky")),
    # Yield set far above anything reached here, so the response is elastic and
    # W is the stored energy of the Simo-Hughes split.
    ("J2, Simo-Hughes",         merge(base("j2 plasticity"), Dict("yield stress" => 1.0e30))),
)

energy(m, F) = Norma.strain_energy(m, F)
energy(m::Norma.J2Plasticity, F) =
    Norma.constitutive(m, F, Norma.initial_state(m); need_tangent = false)[1]

"Random isochoric F_bar with principal stretches within about 1 +- amp."
function random_isochoric(rng, amp)
    A = randn(rng, 3, 3)
    Fb = Matrix(I, 3, 3) + amp * A / opnorm(A)
    Fb = Fb / cbrt(det(Fb))
    return SMatrix{3,3,Float64,9}(Fb)
end

function split_spread(m; Js = (0.85, 0.95, 1.05, 1.2), nsamp = 20, amp = 0.15)
    rng = MersenneTwister(1)
    worst = 0.0
    for J in Js
        D = Float64[]
        for _ in 1:nsamp
            Fb = random_isochoric(rng, amp)
            push!(D, energy(m, cbrt(J) * Fb) - energy(m, Fb))
        end
        spread = (maximum(D) - minimum(D)) / max(abs(sum(D) / nsamp), eps())
        worst = max(worst, spread)
    end
    return worst
end

"Relative residual of the fit W_vol ~ c x^2 over J in [0.8, 1.25], and c/kappa."
function quadratic_fit(m, x::Function; Js = range(0.8, 1.25; length = 25))
    I3 = SMatrix{3,3,Float64,9}(I)
    W0 = energy(m, I3)
    Wv = [energy(m, cbrt(J) * I3) - W0 for J in Js]
    xs = [x(J)^2 for J in Js]
    c = dot(xs, Wv) / dot(xs, xs)
    return norm(Wv - c * xs) / norm(Wv), 2c / m.κ
end

fmt(x) = x == 0 ? "0" : string(round(x, sigdigits = 2))

println("1-2. Split test and quadratic fit (E = $E0, nu = $ν0).\n")
println(rpad("material", 26), rpad("split spread", 14), rpad("fit (J-1)^2", 14),
        rpad("c/kappa", 10), rpad("fit (log J)^2", 15), "c/kappa")
println("-"^90)
for (name, params) in MATERIALS
    m = Norma.create_material(params)
    sp = split_spread(m)
    r1, c1 = quadratic_fit(m, J -> J - 1)
    r2, c2 = quadratic_fit(m, J -> log(J))
    println(rpad(name, 26), rpad(fmt(sp), 14), rpad(fmt(r1), 14),
            rpad(string(round(c1, digits = 4)), 10),
            rpad(fmt(r2), 15), string(round(c2, digits = 4)))
end

# --------------------------------------------------------------------------
# 3. Which variable is projected changes the element.
# --------------------------------------------------------------------------
# One reference tetrahedron, a displacement quadratic in X so that F varies
# within the element, and a rule exact to degree 7 so that quadrature is not
# what is being measured.  The projection is onto P0 (the element mean).

"Conical-product rule on the reference tetrahedron; see common.jl."
function tet_rule(n)
    gj(n, a) = begin
        d = [k == 0 ? -a / (a + 2) : -a^2 / ((2k + a) * (2k + a + 2)) for k in 0:n-1]
        e = [k == 1 ? sqrt(4(a + 1) / ((a + 2)^2 * (a + 3))) :
                      sqrt(4k^2 * (k + a)^2 / ((2k + a)^2 * (2k + a + 1) * (2k + a - 1)))
             for k in 1:n-1]
        Fe = eigen(SymTridiagonal(d, e))
        w = (2.0^(a + 1) / (a + 1)) .* vec(Fe.vectors[1, :]) .^ 2
        (1 .+ Fe.values) ./ 2, w ./ 2.0^(a + 1)
    end
    tu, wu = gj(n, 2); tv, wv = gj(n, 1); tw, ww = gj(n, 0)
    pts = Vector{SVector{3,Float64}}(); wts = Float64[]
    for i in 1:n, j in 1:n, k in 1:n
        u, v, w = tu[i], tv[j], tw[k]
        push!(pts, SVector(u, v * (1 - u), w * (1 - u) * (1 - v)))
        push!(wts, wu[i] * wv[j] * ww[k])
    end
    return pts, wts
end

"F = I + grad u for u quadratic in X, scaled to a nominal strain `amp`."
function defgrad(X, amp)
    x, y, z = X
    G = amp * SMatrix{3,3,Float64,9}(
        1.0 + 2x,   0.5y,     0.3z,
        0.4x,       0.8 + z,  0.2y,
        0.1y,       0.6x,     0.5 + 2z)
    return SMatrix{3,3,Float64,9}(I) + G
end

function three_field_energy(m, θ::Function, pts, wts, amp)
    κ = m.κ
    Ws = Float64[]; θs = Float64[]
    for X in pts
        F = defgrad(X, amp)
        push!(Ws, energy(m, F)); push!(θs, θ(det(F)))
    end
    θbar = dot(wts, θs) / sum(wts)                    # P0 projection
    return sum(wts[q] * (Ws[q] - 0.5κ * θs[q]^2 + 0.5κ * θbar^2) for q in eachindex(wts))
end

println("\n3. Hencky material, one tetrahedron, P0 projection: energy with theta = J - 1")
println("   projected relative to energy with theta = log J projected.\n")
m = Norma.create_material(base("hencky"))
pts, wts = tet_rule(4)
println(rpad("strain", 10), "relative difference")
for amp in (0.005, 0.05, 0.2, 0.5)
    e1 = three_field_energy(m, J -> log(J), pts, wts, amp)
    e2 = three_field_energy(m, J -> J - 1, pts, wts, amp)
    println(rpad(string(amp), 10), fmt(abs(e2 - e1) / abs(e1)))
end
