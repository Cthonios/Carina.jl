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
end
