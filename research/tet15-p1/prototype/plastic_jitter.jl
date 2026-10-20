# r_v of the softest kernel modes at beta = 1 on the perturbed meshes of
# jitter.jl (interior vertices moved by up to 0.25 h per component, seed 1),
# uniaxial and shear flow.  Same solver as plastic_ext.jl.
#
#   julia --project=. research/tet15-p1/prototype/plastic_jitter.jl
ENV["PE_NOMAIN"] = "1"
include("plastic_ext.jl")
using Random

function jittered(N, delta, seed)
    coords, conn = mesh_of(N, 2)
    rng = MersenneTwister(seed)
    X = copy(coords); h = 1 / N
    for i in 1:(N + 1)^3
        x = coords[:, i]
        any(t -> t < 1e-12 || t > 1 - 1e-12, x) && continue
        X[:, i] = x .+ delta * h .* (2 .* rand(rng, 3) .- 1)
    end
    for e in axes(conn, 2), (k, (a, b)) in enumerate(_TET_EDGES)
        X[:, conn[4 + k, e]] = 0.5 .* (X[:, conn[a, e]] .+ X[:, conn[b, e]])
    end
    return coords, X, conn
end

const JCASES = (("P2+int+face/P1disc", 4, :full, (4, 5, 6, 7)),
                ("P2/P0",              1, :none, (4, 5, 6, 7)),
                ("P2+face/P0",         1, :face, (4, 5, 6)),
                ("P2/P1c (TH)",     _P1C, :none, (4, 5, 6)))

for (fname, flow) in (("axial", N_AXIAL), ("shear", N_SHEAR))
    println("perturbed meshes (delta 0.25, seed 1), beta = 1, flow: $fname;  lam_1..5 | r_v,1..5")
    for (label, m, bubble, Ns) in JCASES, N in Ns
        c0, X, conn = jittered(N, 0.25, 1)
        nvert = m == _P1C ? (N + 1)^3 : 0
        ap = assemble_all(X, conn, 2, m; bubble, nvert, flow, beta = 1.0, bc_coords = c0)
        lp, vp, nit = kernel_modes(ap, ap.Kdev ./ 2)
        rv = [dilatation(ap, vp[:, i]) for i in 1:length(lp)]
        println(rpad(label, 21), rpad(string(N), 3), rpad(string(ap.nu), 8),
                join(fmt.(lp)), "| ", join(fmt.(rv)), "  (", nit, " it)")
        flush(stdout)
    end
    println()
end
