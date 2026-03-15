@testset "Mechanics Dynamic Cube (L-BFGS)" begin
    # Same problem as the Newmark-direct test, solved with L-BFGS instead of
    # Newton+direct.  Tolerances match the direct-solver test to within rtol=1e-2
    # on averages (residual dynamic effects dominate over solver tolerance).
    #
    # Tests that NewmarkLBFGSIntegrator with Jacobi H₀ preconditioner produces
    # the same solution as the direct-solver path.

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics", "implicit-dynamic", "cube")
    mktempdir() do dir
        cp(joinpath(example_dir, "cube.g"),          joinpath(dir, "cube.g"))
        cp(joinpath(example_dir, "cube_lbfgs.yaml"), joinpath(dir, "cube_lbfgs.yaml"))
        sim = Carina.run(joinpath(dir, "cube_lbfgs.yaml"))
        avg = average_components(sim)
        mx  = maximum_components(sim)

        @test mx[3]  ≈  1.00e-4 rtol=1e-4   # max u_z = prescribed BC (direct would be 1e-6)
        @test avg[3] ≈  5.00e-5 rtol=1e-2   # avg u_z (quasi-static limit)
        @test avg[1] ≈ -1.25e-5 rtol=1e-2   # avg u_x (Poisson)
        @test avg[2] ≈ -1.25e-5 rtol=1e-2   # avg u_y (Poisson)
    end
end
