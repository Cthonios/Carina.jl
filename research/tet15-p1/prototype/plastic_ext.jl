# Extension of plastic.jl to finer meshes at beta = 1 (referee request: the
# decay of r_v for TET15-P1 was fitted from N = 2-5 only).  Same operators and
# measures as plastic.jl; reports lam and r_v of the five lowest modes.
#
#   julia --project=. research/tet15-p1/prototype/plastic_ext.jl

include("common.jl")
using Random
const N_SHEAR = let n = zeros(3, 3); n[1, 2] = n[2, 1] = 1 / sqrt(2); n end
const N_AXIAL = let n = zeros(3, 3); n[1, 1] = n[2, 2] = -1 / sqrt(6); n[3, 3] = 2 / sqrt(6); n end

# Lowest eigenpairs of K on ker G against the H1 seminorm, for meshes too
# large for a dense null-space basis.  The constraint is imposed by a penalty,
# A = K + gamma G' D^-1 G with D the diagonal of the pressure mass matrix,
# whose eigenpairs converge to the constrained ones as gamma grows (error of
# order 1/gamma).  Subspace iteration with Rayleigh--Ritz on the inverse
# operator A^-1 Kh1, Cholesky factorization of A.
function kernel_modes(a, K; k = 5, b = 16, gamma = 1e9, tol = 1e-7, maxit = 3000)
    D = Diagonal(1 ./ Vector(diag(a.M)))
    A = sparse(Symmetric(K + gamma * (a.G' * D * a.G)))
    F = cholesky(Symmetric(A))
    Kh = a.Kh1
    X = randn(MersenneTwister(1), a.nu, b)
    lam = fill(Inf, k)
    for it in 1:maxit
        Y = F \ (Kh * X)
        AY = A * Y; BY = Kh * Y
        E = eigen(Symmetric(Y' * AY), Symmetric(Y' * BY))
        X = Y * E.vectors
        X ./= sqrt.(sum(X .* (Kh * X), dims = 1))
        new = E.values[1:k]
        if maximum(abs.(new .- lam) ./ abs.(new)) < tol
            return new, X[:, 1:k], it
        end
        lam = new
    end
    @warn "kernel_modes: not converged"
    return lam, X[:, 1:k], maxit
end
dilatation(a, v) = dot(v, a.Kdiv * v) / dot(v, a.Kh1 * v)
fmt(x) = rpad(string(round(x, sigdigits = 3)), 9)

const _CASES = (("P2+int+face/P1disc", 4, :full, (6, 7, 8)),
               ("P2/P0",              1, :none, (6, 7, 8)),
               ("P2+face/P0",         1, :face, (6, 7)),
               ("P2/P1c (TH)",     _P1C, :none, (6, 7, 8)))

const CASES = haskey(ENV, "PE_CHECK") ?
    ((c[1], c[2], c[3], (4,)) for c in _CASES) : _CASES

get(ENV, "PE_NOMAIN", "") == "1" || for (fname, flow) in (("axial", N_AXIAL), ("shear", N_SHEAR))
    println("beta = 1, flow: $fname;  lam_1..5 | r_v,1..5")
    for (label, m, bubble, Ns) in CASES, N in Ns
        t = @elapsed begin
            coords, conn = mesh_of(N, 2)
            nvert = m == _P1C ? (N + 1)^3 : 0
            ap = assemble_all(coords, conn, 2, m; bubble, nvert, flow, beta = 1.0)
            lp, vp, nit = kernel_modes(ap, ap.Kdev ./ 2)
            rv = [dilatation(ap, vp[:, i]) for i in 1:length(lp)]
        end
        println(rpad(label, 21), rpad(string(N), 3), rpad(string(ap.nu), 8),
                join(fmt.(lp)), "| ", join(fmt.(rv)), "  (", nit, " it, ", round(Int, t), " s)")
        flush(stdout)
    end
    println()
end
