@testset "Mechanics Quasi-static Cube (NLCG)" begin
    # Same problem as cube.yaml but solved with nonlinear CG (matrix-free).
    # Linear elastic, E=10e9, ν=0.25, uniaxial stress via DBC u_z = 1e-3*t.
    # At t=1: avg_uz = 5e-4, avg_ux = avg_uy = -1.25e-4.

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics", "quasistatic", "cube")
    mktempdir() do dir
        cp_example(joinpath(example_dir, "cube.g"),          joinpath(dir, "cube.g"))
        cp_example(joinpath(example_dir, "cube_nlcg.yaml"),  joinpath(dir, "cube_nlcg.yaml"))
        sim = Carina.run(joinpath(dir, "cube_nlcg.yaml"))
        avg = average_components(sim)

        @test avg[3] ≈  5.0e-4  rtol=1e-4
        @test avg[1] ≈ -1.25e-4 rtol=1e-4
        @test avg[2] ≈ -1.25e-4 rtol=1e-4
    end
end
