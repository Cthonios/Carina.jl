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
end
