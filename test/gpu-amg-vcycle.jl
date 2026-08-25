# Device-AMG V-cycle machinery, validated on the CPU backend.
#
# DeviceAMGHierarchy and _amg_vcycle! are backend-generic KernelAbstractions
# code; KA.CPU() runs the same kernels the GPU does, so their numerics are
# testable in CPU CI.  The GPU end-to-end path (CG+AMG on ROCm against the
# CPU direct answer) is covered in mechanics-gpu-device.jl when a device is
# present.
#
# The operator here is a 3D 7-point Laplacian: SPD, large enough (n = 20^3)
# that smoothed aggregation produces a real multilevel hierarchy, and with a
# known near-nullspace (constants), so the V-cycle's effect on CG iteration
# counts is unambiguous.

@testset "GPU AMG V-cycle (CPU backend)" begin
    import AlgebraicMultigrid as AMG
    using SparseArrays, LinearAlgebra
    import Krylov
    import LinearOperators: LinearOperator

    N = 20
    n = N^3
    # 3D Laplacian, Dirichlet: standard 7-point stencil.
    idx(i, j, k) = i + N * (j - 1) + N * N * (k - 1)
    I_ = Int[]; J_ = Int[]; V_ = Float64[]
    for k in 1:N, j in 1:N, i in 1:N
        r = idx(i, j, k)
        push!(I_, r); push!(J_, r); push!(V_, 6.0)
        for (di, dj, dk) in ((1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1))
            ii, jj, kk = i + di, j + dj, k + dk
            (1 <= ii <= N && 1 <= jj <= N && 1 <= kk <= N) || continue
            push!(I_, r); push!(J_, idx(ii, jj, kk)); push!(V_, -1.0)
        end
    end
    A = sparse(I_, J_, V_, n, n)

    backend = Carina.KA.CPU()
    ml = AMG.smoothed_aggregation(A)
    @test length(ml.levels) >= 1

    dinv_h = 1.0 ./ Vector(diag(A))
    inv_d1 = copy(dinv_h)                      # "device" vector on CPU backend
    lmax1  = Carina._host_lambda_max(A, dinv_h)
    h = Carina.DeviceAMGHierarchy(backend, ml, inv_d1, lmax1)

    # The fine level is matrix-free by contract; emulate it with a closure.
    fine_mv!(y, x) = mul!(y, A, x)

    b = ones(n)
    x_direct = A \ b

    # --- V-cycle actually solves: as a preconditioner it must cut CG
    #     iterations by a lot relative to Jacobi.
    M_amg = LinearOperator(Float64, n, n, true, true,
        (y, v) -> (Carina._amg_vcycle!(y, v, h, fine_mv!, backend); y))
    M_jac = LinearOperator(Float64, n, n, true, true,
        (y, v) -> (@. y = dinv_h * v; y))

    x_amg, st_amg = Krylov.cg(A, b; M=M_amg, rtol=1e-10, history=true)
    x_jac, st_jac = Krylov.cg(A, b; M=M_jac, rtol=1e-10, history=true)

    @test st_amg.solved
    @test x_amg ≈ x_direct rtol = 1e-8
    @test x_jac ≈ x_direct rtol = 1e-8
    # Jacobi needs ~O(N) iterations on the Laplacian; the V-cycle must be
    # h-independent-ish.  Enforce both an absolute and a relative bound.
    @test st_amg.niter < 20
    @test st_amg.niter * 3 < st_jac.niter

    # --- Preconditioner must be symmetric positive definite for CG:
    #     symmetry check via random vectors, <u, Mv> == <Mu, v>.
    u = randn(n); v = randn(n)
    Mu = zeros(n); Mv = zeros(n)
    Carina._amg_vcycle!(Mu, u, h, fine_mv!, backend)
    Carina._amg_vcycle!(Mv, v, h, fine_mv!, backend)
    @test dot(u, Mv) ≈ dot(Mu, v) rtol = 1e-6
    @test dot(u, Mu) > 0.0
end

# --------------------------------------------------------------------------- #
# _fit_candidates replaces AMG.fit_candidates in the hierarchy build (the
# stock version inserts into a live sparse matrix per entry and dropzeros!-es
# per aggregate — 41% of the build at 528k DOF, 12x slower than this one).
# Same per-aggregate LAPACK QR, so the outputs must agree to roundoff, up to
# entries the stock version drops below its 1e-10 tolerance (ours drops the
# same way).  The 3D Laplacian with a 6-column fake-rigid-body nullspace
# exercises rank-deficient aggregates too (aggregates near corners can have
# fewer rows than columns).
# --------------------------------------------------------------------------- #
@testset "_fit_candidates matches AMG.fit_candidates" begin
    import AlgebraicMultigrid as AMGx
    n1 = 10
    A = Carina.AMG.poisson((n1, n1, n1))     # 1000-DOF 3D Laplacian
    n = size(A, 1)
    # Six smooth candidate vectors (constant + coordinates + products).
    xs = repeat(1.0:n1, outer = n1 * n1); ys = repeat(1.0:n1, inner = n1, outer = n1)
    zs = repeat(1.0:n1, inner = n1 * n1)
    B = hcat(ones(n), xs, ys, zs, xs .* ys, ys .* zs)
    S, _  = Carina.AMG.SymmetricStrength()(A, false)
    AggOp = Carina.AMG.StandardAggregation()(S)

    T1, Bc1 = AMGx.fit_candidates(AggOp, B)
    T2, Bc2 = Carina._fit_candidates(AggOp, B)

    @test size(T1) == size(T2)
    d = T1 - T2
    @test (SparseArrays.nnz(d) == 0 ? 0.0 : maximum(abs, d.nzval)) < 1e-9
    @test Bc1 ≈ Bc2 atol = 1e-12
    # Defining property: the tentative prolongator reproduces the candidates.
    @test maximum(abs, T2 * Bc2 - B) < 1e-10
    # Orthonormal blocks: TᵀT is the identity on the range T spans.  A
    # rank-deficient aggregate (fewer rows than candidates) leaves its
    # trailing columns empty — same as the stock version — so the diagonal
    # is 0 there and 1 elsewhere, with zero off-diagonals everywhere.
    G = Matrix(T2' * T2)
    dg = LinearAlgebra.diag(G)
    @test all(d -> abs(d) < 1e-10 || abs(d - 1.0) < 1e-10, dg)
    @test maximum(abs, G - LinearAlgebra.Diagonal(dg)) < 1e-10
end
