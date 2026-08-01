@testset "Solver status tests" begin
    # StagnationTest and ModelFlagTest are the two status tests nothing
    # constructed: the existing termination-parsing tests cover the residual and
    # iteration-count tests, but a stalled solve and a constitutive-model failure
    # are exactly the situations where a status test has to work.

    info(; norm_R=1.0, iter=1) = Carina.SolverInfo(
        iter, norm_R, 1.0, norm_R, 1.0e-3, 1.0)

    @testset "StagnationTest" begin
        # A run that is still reducing its residual must not be declared stalled.
        t = Carina.StagnationTest(; window=3, tol=0.95)
        for k in 0:8
            @test Carina.check(t, info(; norm_R=10.0^(-k))) == Carina.Unconverged
        end

        # Below `window` samples there is no history to compare against yet.
        t = Carina.StagnationTest(; window=3, tol=0.95)
        for _ in 1:3
            @test Carina.check(t, info(; norm_R=1.0)) == Carina.Unconverged
        end

        # A residual that stops moving trips the detector, but only after the
        # stall has persisted for a full window -- one flat iteration is normal.
        t = Carina.StagnationTest(; window=2, tol=0.95)
        results = [Carina.check(t, info(; norm_R=1.0)) for _ in 1:8]
        @test Carina.Failed in results
        @test results[1] == Carina.Unconverged

        # reset! must clear both the history and the stall counter, so a reused
        # test does not inherit the previous solve's verdict.
        Carina.reset!(t)
        @test Carina.check(t, info(; norm_R=1.0)) == Carina.Unconverged
    end

    @testset "ModelFlagTest" begin
        # The push-based channel a constitutive model uses to abort a solve.
        t = Carina.ModelFlagTest()
        @test Carina.check(t, info()) == Carina.Unconverged

        t.status  = Carina.Failed
        t.message = "material point failed to converge"
        @test Carina.check(t, info()) == Carina.Failed

        Carina.reset!(t)
        @test Carina.check(t, info()) == Carina.Unconverged
        @test isempty(t.message)
    end

    # SolverInfo(iteration, norm_R, norm_R_init, norm_R_prev, norm_step, norm_solution)
    step_info(; step=1.0e-3, U=1.0) = Carina.SolverInfo(1, 1.0, 1.0, 1.0, step, U)

    @testset "update-norm tests" begin
        # Step-size convergence criteria: parseable from YAML since the
        # termination rework, but nothing ever evaluated one.
        t = Carina.AbsUpdateTest(1.0e-6)
        @test Carina.check(t, step_info(; step=1.0e-8)) == Carina.Converged
        @test Carina.check(t, step_info(; step=1.0e-3)) == Carina.Unconverged

        t = Carina.RelUpdateTest(1.0e-4)
        @test Carina.check(t, step_info(; step=1.0e-6, U=1.0)) == Carina.Converged
        @test Carina.check(t, step_info(; step=1.0e-2, U=1.0)) == Carina.Unconverged
        # A zero solution norm must not divide; the test just stays unconverged.
        @test Carina.check(t, step_info(; step=1.0e-6, U=0.0)) == Carina.Unconverged
    end

    @testset "DivergenceTest" begin
        t = Carina.DivergenceTest(1.0e3)
        grew(factor) = Carina.SolverInfo(3, factor, 1.0, factor / 2, 1.0e-3, 1.0)
        @test Carina.check(t, grew(10.0))   == Carina.Unconverged
        @test Carina.check(t, grew(1.0e4)) == Carina.Failed
    end

    @testset "default status-test builders" begin
        # The exported convenience constructors for embedding Carina solvers
        # programmatically (no YAML, hence no parsed termination block).
        t = Carina.default_nonlinear_status_test(;
            abs_tol=1.0e-8, rel_tol=1.0e-12, max_iters=7)
        @test t isa Carina.ComboOrTest
        # Convergence is AND(abs, rel): the sample must satisfy both tolerances.
        @test Carina.check(t, Carina.SolverInfo(1, 1.0e-13, 1.0, 1.0, 1.0, 1.0)) ==
              Carina.Converged
        @test Carina.check(t, Carina.SolverInfo(7, 1.0, 1.0, 1.0, 1.0, 1.0)) ==
              Carina.Failed
        @test Carina.check(t, Carina.SolverInfo(1, NaN, 1.0, 1.0, 1.0, 1.0)) ==
              Carina.Failed

        t = Carina.default_linear_status_test(; rtol=1.0e-6, max_iters=50)
        @test t isa Carina.ComboOrTest
        @test Carina.check(t, Carina.SolverInfo(1, 1.0e-8, 1.0, 1.0, 1.0, 1.0)) ==
              Carina.Converged
        @test Carina.check(t, Carina.SolverInfo(50, 1.0, 1.0, 1.0, 1.0, 1.0)) ==
              Carina.Failed

        # The in-solver fallback used when no termination block was parsed.
        ns = Carina.NewtonSolver(0, 20, 1.0e-8, 1.0e-8, 1.0e-12,
                                 Carina.NoLinearSolver(), true, 0.5, 1.0e-4, 10)
        t = Carina._build_status_test(ns)
        @test t isa Carina.ComboOrTest
        @test Carina.check(t, Carina.SolverInfo(1, 0.0, 1.0, 1.0, 1.0, 1.0)) ==
              Carina.Converged
    end
end
