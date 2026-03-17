@testset "Mechanics Quasi-static Cube (Neumann BC)" begin
    # Unit cube [0,1]³, linear elastic (infinitesimal strain), E=1e9, ν=0.25.
    # BCs: u_x=0 on ssx-, u_y=0 on ssy-, u_z=0 on ssz-.
    # Neumann: traction t_z = +1e9*t on ssz+ (FEC sign convention: g = -traction).
    # Final time t=1.0 → applied traction 1e9 Pa.
    #
    # Analytical solution (uniaxial stress, small strain):
    #   ε_z = t_z / E = 1e6 / 1e9 = 1e-3    avg_uz = ε_z * 0.5 = 5e-4
    #   ε_x = ε_y = -ν * ε_z = -2.5e-4      avg_ux = avg_uy = -2.5e-4 * 0.5 = -1.25e-4

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics", "quasistatic", "cube-neumann-bc")
    mktempdir() do dir
        cp_example(joinpath(example_dir, "cube.g"),    joinpath(dir, "cube.g"))
        cp_example(joinpath(example_dir, "cube.yaml"), joinpath(dir, "cube.yaml"))
        sim = Carina.run(joinpath(dir, "cube.yaml"))
        avg = average_components(sim)

        @test avg[3] ≈  5.0e-4  rtol=1e-3   # avg u_z
        @test avg[1] ≈ -1.25e-4 rtol=1e-3   # avg u_x (Poisson contraction)
        @test avg[2] ≈ -1.25e-4 rtol=1e-3   # avg u_y (Poisson contraction)
    end
end
