# Where the dilatational kernel modes of TET15-P1 on the Cubit cube meshes
# sit: for each mode among the twenty lowest with r_v > 0.1 (uniaxial flow,
# beta = 1), the element carrying the largest int (div v)^2, its distance to
# the boundary of the cube, its number of vertices on the boundary, its
# quality, and the share of the mode's dilatation in that element.
ENV["CC_NOMAIN"] = "1"
include("cubit_cube.jl")
for n in (3, 4, 5, 6)
    X, conn = cubit_mesh(n)
    ap = assemble_all(X, conn, 2, 4; bubble = :full, flow = N_AXIAL, beta = 1.0)
    lp, vp, _ = kernel_modes(ap, ap.Kdev ./ 2; k = 20, b = 40)
    println("n = $n")
    for i in eachindex(lp)
        rv = dilatation(ap, vp[:, i]); rv > 0.1 || continue
        ed = elem_div2(X, conn, :full, ap.nu_nodal, vp[:, i])
        e = argmax(ed)
        c = sum(X[:, conn[a, e]] for a in 1:4) ./ 4
        nb = count(a -> any(t -> t < 1e-9 || t > 1 - 1e-9, X[:, conn[a, e]]), 1:4)
        println("  mode $i lam $(round(lp[i], sigdigits=3)) r_v $(round(rv, sigdigits=3)) elem $e share $(round(ed[e]/sum(ed), sigdigits=2)) ",
                "dist $(round(minimum(min.(c, 1 .- c)), sigdigits=2)) h=$(round(1/n, sigdigits=2)) boundary vertices $nb q $(round(quality(X, conn, e), sigdigits=3))")
    end
    flush(stdout)
end
