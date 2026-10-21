# Carina: Copyright 2026 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software. This software
# is released under the BSD license detailed in the file LICENSE.md in the
# top-level Carina.jl directory.

@testset "Linear solver tolerances" begin
    # The iterative linear solver takes a relative and an absolute tolerance,
    # as Norma's do: `relative tolerance` (alias `tolerance`) and `absolute
    # tolerance`, the latter on the Euclidean norm of the linear residual and
    # by default one tenth of the Newton absolute residual tolerance.  A solve
    # whose right-hand side is already below the absolute tolerance takes no
    # iteration, and a loose absolute tolerance shortens a solve.
    template = zeros(12)
    newton(atol) = Dict{String, Any}("termination" => Dict{String, Any}(
        "converge when any" => Any[Dict{String, Any}("absolute residual" => atol),
                                   Dict{String, Any}("relative residual" => 1e-10)],
        "fail when any" => Any[Dict{String, Any}("maximum iterations" => 20)]))
    parse(ls; natol = 0.0) = Carina._parse_linear_solver(
        Dict{String, Any}(merge(Dict{String, Any}("type" => "iterative"), ls)), template,
        Carina.KA.CPU(), () -> Carina.NoPreconditioner(); newton_atol = natol)

    @testset "keys and defaults" begin
        ls = parse(Dict{String, Any}())
        @test ls.rtol == 1e-8 && ls.atol == 0.0
        @test parse(Dict{String, Any}("tolerance" => 1e-6)).rtol == 1e-6
        @test parse(Dict{String, Any}("relative tolerance" => 1e-7)).rtol == 1e-7
        @test parse(Dict{String, Any}("absolute tolerance" => 2e-5)).atol == 2e-5
        @test parse(Dict{String, Any}(); natol = 1e-6).atol ≈ 1e-7
        @test parse(Dict{String, Any}("absolute tolerance" => 0.0); natol = 1e-6).atol == 0.0
        @test_throws ErrorException parse(Dict{String, Any}("tolerance" => 1e-6,
                                                            "relative tolerance" => 1e-7))
        @test_throws ErrorException parse(Dict{String, Any}("relative tolerance" => 2.0))
        @test_throws ErrorException parse(Dict{String, Any}("absolute tolerance" => -1.0))
        @test Carina._newton_abs_residual(newton(3e-6)) == 3e-6
        @test Carina._newton_abs_residual(Dict{String, Any}("termination" => Dict{String, Any}(
            "converge when any" => Any[Dict{String, Any}("relative residual" => 1e-10)],
            "fail when any" => Any[Dict{String, Any}("maximum iterations" => 20)]))) == 0.0
    end

    @testset "the absolute test stops conjugate gradients" begin
        # Condition number about 3: CG converges in a few dozen iterations,
        # far from the exact termination at n.
        n = 1000
        A = Carina.SparseArrays.spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(4.0, n),
                                        1 => fill(-1.0, n - 1))
        b = ones(n)
        x = zeros(n)
        it_rel, _, ok_rel = Carina._device_pcg!(x, A, b, nothing, 1e-12, 10_000)
        @test ok_rel
        it_abs, _, ok_abs = Carina._device_pcg!(x, A, b, nothing, 1e-12, 10_000; atol = 1e-2)
        @test ok_abs && it_abs < it_rel
        @test Carina.LinearAlgebra.norm(A * x - b) <= 1e-2
        it0, _, ok0 = Carina._device_pcg!(x, A, 1e-9 .* b, nothing, 1e-12, 10_000; atol = 1e-6)
        @test ok0 && it0 == 0 && all(iszero, x)
    end

    @testset "a preconditioner that is not positive falls back to the diagonal" begin
        # M⁻¹ = −I gives rᵀz < 0 for every r ≠ 0.  The solve reports it once
        # and converges with the diagonal preconditioner, where it returned
        # ΔU = 0 before 2026-10-18.
        n = 1000
        A = Carina.SparseArrays.spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(4.0, n),
                                        1 => fill(-1.0, n - 1))
        b = ones(n)
        M = Carina.LinearOperators.LinearOperator(Float64, n, n, true, true,
                                                  (y, v) -> (y .= .-v))
        for fallback in (nothing, fill(0.25, n))
            x = zeros(n)
            reports = Float64[]
            it, _, ok = Carina._device_pcg!(x, A, b, M, 1e-10, 10_000;
                fallback = fallback, report = (r, z, rz, rr) -> push!(reports, rz))
            @test ok && it > 0
            @test length(reports) == 1 && reports[1] < 0
            @test Carina.LinearAlgebra.norm(A * x - b) <= 1e-8 * Carina.LinearAlgebra.norm(b)
        end
        # A preconditioner that is positive for the right-hand side but not for
        # a later residual: rᵀz > 0 at the start, < 0 at a check.  Before
        # 2026-10-18 the check read ρ < 0 as convergence.
        w = zeros(n); w[1:10] .= 1.0
        M2 = Carina.LinearOperators.LinearOperator(Float64, n, n, true, true,
            (y, v) -> (y .= v .- 1e3 .* w .* v))
        x = zeros(n)
        reports = Float64[]
        it, _, ok = Carina._device_pcg!(x, A, b .* (1.0 .- w), M2, 1e-10, 10_000;
            fallback = fill(0.25, n), report = (r, z, rz, rr) -> push!(reports, rz))
        @test ok
        @test Carina.LinearAlgebra.norm(A * x - b .* (1.0 .- w)) <= 1e-8 * Carina.LinearAlgebra.norm(b)
        @test length(reports) == 1 && reports[1] < 0
    end
end
