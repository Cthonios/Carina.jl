@testset "Mechanics Dynamic Cube (NLCG)" begin
    # Same problem as cube_lbfgs.toml but solved with nonlinear CG (matrix-free).
    # Implicit Newmark, neohookean, DBC u_z = 1e-3*t.
    # Compared against the Newton+direct solution for correctness.

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics", "implicit-dynamic", "cube")
    mktempdir() do dir
        cp_example(joinpath(example_dir, "cube.g"),           joinpath(dir, "cube.g"))
        cp_example(joinpath(example_dir, "cube_direct.toml"), joinpath(dir, "cube_direct.toml"))
        cp_example(joinpath(example_dir, "cube_nlcg.toml"),   joinpath(dir, "cube_nlcg.toml"))

        sim_ref  = Carina.run(joinpath(dir, "cube_direct.toml"))
        sim_nlcg = Carina.run(joinpath(dir, "cube_nlcg.toml"))

        avg_ref  = average_components(sim_ref)
        avg_nlcg = average_components(sim_nlcg)

        @test avg_nlcg[1] ≈ avg_ref[1] rtol=0.01
        @test avg_nlcg[2] ≈ avg_ref[2] rtol=0.01
        @test avg_nlcg[3] ≈ avg_ref[3] rtol=0.01
    end
end
