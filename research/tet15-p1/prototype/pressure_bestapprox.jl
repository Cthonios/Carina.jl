# Referee round 2, M15: best-approximation errors behind the kappa sweep.
# For the manufactured solution of kappa_sweep.jl at kappa -> infinity the
# mean stress is p = Lap(phi), phi = 3B, a polynomial of degree 10.  Brezzi's
# estimate for a stable pair bounds the error uniformly in kappa by the best
# approximation of u in the displacement space and of p in the pressure
# space; the second enters only for large kappa.  Reported: the relative L2
# error of the element-wise L2 projection of p onto P1 (discontinuous) and
# onto P0, on the Freudenthal meshes, with local rates.
ENV["KS_NOMAIN"] = "1"
include("kappa_sweep.jl")
lapB(x) = tr(ForwardDiff.hessian(Bfun, x))
pex(x) = SPHI * lapB(x)

function proj_err(N, m)
    coords, conn = mesh_of(N, 1)
    qp, qw = tet_rule(14)
    e2 = 0.0; n2 = 0.0
    for e in axes(conn, 2)
        X = coords[:, conn[1:4, e]]
        Jm = [X[:, 2] - X[:, 1]  X[:, 3] - X[:, 1]  X[:, 4] - X[:, 1]]; dJ = abs(det(Jm))
        Mloc = zeros(m, m); b = zeros(m); ps = Float64[]; phis = Vector{Float64}[]
        for q in eachindex(qw)
            xi = qp[:, q]; x = X[:, 1] + Jm * xi
            phi = collect(pressure_basis(Val(m), xi)); p = pex(x)
            Mloc .+= qw[q] * dJ .* (phi * phi'); b .+= qw[q] * dJ * p .* phi
            push!(ps, p); push!(phis, phi)
        end
        c = Mloc \ b
        for q in eachindex(qw)
            e2 += qw[q] * dJ * (ps[q] - dot(c, phis[q]))^2; n2 += qw[q] * dJ * ps[q]^2
        end
    end
    return sqrt(e2 / n2)
end

println("Relative L2 error of the element-wise projection of p = Lap(3B); local rates in N")
for (m, name) in ((4, "P1 discontinuous"), (1, "P0"))
    prev = nothing
    for N in (2, 3, 4, 6, 8, 10, 12, 16, 24, 32)
        err = proj_err(N, m)
        r = prev === nothing ? "" : @sprintf("  rate %.2f", log(prev[2] / err) / log(N / prev[1]))
        @printf("  %-17s N = %-3d  %.4e%s\n", name, N, err, r)
        prev = (N, err)
    end
end
