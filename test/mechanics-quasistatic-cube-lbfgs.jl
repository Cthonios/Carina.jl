@testset "Mechanics Quasi-static Cube L-BFGS" begin
    # Same problem as mechanics-quasistatic-cube.jl but solved with
    # QuasiStaticLBFGSIntegrator instead of Newton + direct solver.
    # Verifies that L-BFGS produces the same answer as Newton to rtol=1e-4.
    #
    # Unit cube [0,1]³, neo-Hookean, E=10e9, ν=0.25.
    # BCs: u_x=0 on x=0, u_y=0 on y=0, u_z=0 on z=0, u_z=1e-3*t on z=1.
    # Applied strain ε_z = 1e-3 at t=1.
    #
    # Analytical solution (small strain, coincides with neo-Hookean at ε_z=1e-3):
    #   avg_uz =  5.00e-4   avg_ux = avg_uy = -1.25e-4   max_uz = 1.00e-3

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics", "quasistatic", "cube")
    mktempdir() do dir
        cp_example(joinpath(example_dir, "cube.g"),          joinpath(dir, "cube.g"))
        cp_example(joinpath(example_dir, "cube_lbfgs.yaml"), joinpath(dir, "cube_lbfgs.yaml"))
        sim = Carina.run(joinpath(dir, "cube_lbfgs.yaml"); device="cpu")
        avg = average_components(sim)
        mx  = maximum_components(sim)

        @test avg[3] ≈  5.00e-4 rtol=1e-4
        @test avg[1] ≈ -1.25e-4 rtol=1e-2
        @test avg[2] ≈ -1.25e-4 rtol=1e-2
        @test mx[3]  ≈  1.00e-3 rtol=1e-6
    end
end
