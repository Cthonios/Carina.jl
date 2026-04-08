@testset "Termination criteria parsing" begin

    # ----- New syntax: converge when any + fail when any ---------------------
    @testset "new syntax — converge when any / fail when any" begin
        sol = Dict(
            "termination" => Dict(
                "converge when any" => [
                    Dict("absolute residual" => 1e-6),
                    Dict("relative residual" => 1e-10),
                ],
                "fail when any" => [
                    Dict("maximum iterations" => 16),
                ],
            ),
        )
        root = Carina._parse_termination(sol)

        # Root is a ComboOrTest (converge-group OR fail-group OR FiniteValueTest)
        @test root isa Carina.ComboOrTest
        @test length(root.tests) == 3  # converge, fail, FiniteValueTest

        conv = root.tests[1]
        @test conv isa Carina.ComboOrTest
        @test length(conv.tests) == 2
        @test conv.tests[1] isa Carina.AbsResidualTest
        @test conv.tests[1].tol == 1e-6
        @test conv.tests[2] isa Carina.RelResidualTest
        @test conv.tests[2].tol == 1e-10

        fail_grp = root.tests[2]
        @test fail_grp isa Carina.ComboOrTest
        @test length(fail_grp.tests) == 1
        @test fail_grp.tests[1] isa Carina.MaxIterationsTest
        @test fail_grp.tests[1].max_iters == 16

        @test root.tests[3] isa Carina.FiniteValueTest
    end

    # ----- New syntax: converge when all + nested any -----------------------
    @testset "new syntax — converge when all with nested any" begin
        sol = Dict(
            "termination" => Dict(
                "converge when all" => [
                    Dict("minimum iterations" => 0),
                    Dict("any" => [
                        Dict("absolute residual" => 1e-8),
                        Dict("relative residual" => 1e-12),
                    ]),
                ],
                "fail when any" => [
                    Dict("maximum iterations" => 16),
                ],
            ),
        )
        root = Carina._parse_termination(sol)
        @test root isa Carina.ComboOrTest

        conv = root.tests[1]
        @test conv isa Carina.ComboAndTest
        @test length(conv.tests) == 2
        @test conv.tests[1] isa Carina.MinIterationsTest
        @test conv.tests[1].min_iters == 0

        nested_or = conv.tests[2]
        @test nested_or isa Carina.ComboOrTest
        @test length(nested_or.tests) == 2
        @test nested_or.tests[1] isa Carina.AbsResidualTest
        @test nested_or.tests[2] isa Carina.RelResidualTest
    end

    # ----- New syntax: nested all inside any --------------------------------
    @testset "new syntax — nested all inside converge when any" begin
        sol = Dict(
            "termination" => Dict(
                "converge when any" => [
                    Dict("all" => [
                        Dict("absolute residual" => 1e-6),
                        Dict("absolute update"   => 1e-8),
                    ]),
                    Dict("relative residual" => 1e-14),
                ],
                "fail when any" => [
                    Dict("maximum iterations" => 20),
                ],
            ),
        )
        root = Carina._parse_termination(sol)
        conv = root.tests[1]
        @test conv isa Carina.ComboOrTest
        @test conv.tests[1] isa Carina.ComboAndTest
        @test conv.tests[1].tests[1] isa Carina.AbsResidualTest
        @test conv.tests[1].tests[2] isa Carina.AbsUpdateTest
        @test conv.tests[2] isa Carina.RelResidualTest
    end

    # ----- Evaluation: converge when any fires on first match ---------------
    @testset "evaluation — converge when any" begin
        sol = Dict(
            "termination" => Dict(
                "converge when any" => [
                    Dict("absolute residual" => 1e-6),
                ],
                "fail when any" => [
                    Dict("maximum iterations" => 10),
                ],
            ),
        )
        root = Carina._parse_termination(sol)
        # Residual below tolerance → converged
        info = Carina.SolverInfo(1, 1e-7, 1.0, 1.0, 0.0, 1.0)
        @test Carina.check(root, info) == Carina.Converged
        # Residual above tolerance, not at max iters → unconverged
        info2 = Carina.SolverInfo(1, 1e-3, 1.0, 1.0, 0.0, 1.0)
        @test Carina.check(root, info2) == Carina.Unconverged
        # At max iters → failed
        info3 = Carina.SolverInfo(10, 1e-3, 1.0, 1.0, 0.0, 1.0)
        @test Carina.check(root, info3) == Carina.Failed
    end

    # ----- Evaluation: converge when all requires all children ---------------
    @testset "evaluation — converge when all" begin
        sol = Dict(
            "termination" => Dict(
                "converge when all" => [
                    Dict("minimum iterations" => 3),
                    Dict("absolute residual"  => 1e-6),
                ],
                "fail when any" => [
                    Dict("maximum iterations" => 20),
                ],
            ),
        )
        root = Carina._parse_termination(sol)
        # Residual OK but below min iters → unconverged
        info = Carina.SolverInfo(1, 1e-7, 1.0, 1.0, 0.0, 1.0)
        @test Carina.check(root, info) == Carina.Unconverged
        # Both satisfied → converged
        info2 = Carina.SolverInfo(3, 1e-7, 1.0, 1.0, 0.0, 1.0)
        @test Carina.check(root, info2) == Carina.Converged
    end

    # ----- Legacy syntax (list of typed entries) still works -----------------
    @testset "legacy syntax — list of typed entries" begin
        sol = Dict(
            "termination" => [
                Dict(
                    "type" => "combo",
                    "combo" => "or",
                    "tests" => [
                        Dict("type" => "absolute residual", "tolerance" => 1e-6),
                        Dict("type" => "relative residual", "tolerance" => 1e-10),
                    ],
                ),
                Dict("type" => "maximum iterations", "value" => 16),
            ],
        )
        root = Carina._parse_termination(sol)
        @test root isa Carina.ComboOrTest
        info = Carina.SolverInfo(1, 1e-7, 1.0, 1.0, 0.0, 1.0)
        @test Carina.check(root, info) == Carina.Converged
    end

    # ----- Flat-key legacy (oldest format) ----------------------------------
    @testset "oldest legacy — flat tolerance keys" begin
        sol = Dict(
            "absolute tolerance" => 1e-8,
            "relative tolerance" => 1e-12,
        )
        root = Carina._parse_termination(sol)
        @test root isa Carina.ComboOrTest
        @test any(t -> t isa Carina.AbsResidualTest, root.tests)
        @test any(t -> t isa Carina.FiniteValueTest, root.tests)
    end

    # ----- Error on unknown test name ---------------------------------------
    @testset "error on unknown test name" begin
        sol = Dict(
            "termination" => Dict(
                "converge when any" => [
                    Dict("bogus test" => 1.0),
                ],
            ),
        )
        @test_throws ErrorException Carina._parse_termination(sol)
    end

    # ----- Error on unknown when-key ----------------------------------------
    @testset "error on unknown when-key" begin
        sol = Dict(
            "termination" => Dict(
                "stop when maybe" => [
                    Dict("absolute residual" => 1e-6),
                ],
            ),
        )
        @test_throws ErrorException Carina._parse_termination(sol)
    end

    # ----- All test types recognized ----------------------------------------
    @testset "all test types parse" begin
        items = [
            (Dict("absolute residual"  => 1e-6),  Carina.AbsResidualTest),
            (Dict("relative residual"  => 1e-10), Carina.RelResidualTest),
            (Dict("absolute update"    => 1e-8),  Carina.AbsUpdateTest),
            (Dict("relative update"    => 1e-6),  Carina.RelUpdateTest),
            (Dict("maximum iterations" => 20),    Carina.MaxIterationsTest),
            (Dict("minimum iterations" => 2),     Carina.MinIterationsTest),
            (Dict("finite value"       => 0),     Carina.FiniteValueTest),
            (Dict("divergence"         => 1e6),   Carina.DivergenceTest),
            (Dict("stagnation"         => 5),     Carina.StagnationTest),
        ]
        for (entry, expected_type) in items
            t = Carina._parse_termination_item(entry)
            @test t isa expected_type
        end
    end
end
