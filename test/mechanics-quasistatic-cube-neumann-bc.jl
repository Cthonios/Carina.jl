@testset "Mechanics Quasi-static Cube (Neumann BC)" begin
    # Unit cube [0,1]³, linear elastic (infinitesimal strain), E=1e9, ν=0.25.
    # BCs: u_x=0 on ssx-, u_y=0 on ssy-, u_z=0 on ssz-.
    # Neumann: traction t_z = +1e9*t on ssz+ (FEC sign convention: g = -traction).
    # Final time t=1.0 → applied traction 1e9 Pa.
    #
    # Matches Norma single-static-solid-neumann-bc (same E, traction, time stepping).
    #
    # Analytical solution (uniaxial stress, small strain):
    #   ε_z = t_z / E = 1e9 / 1e9 = 1.0    avg_uz = ε_z * 0.5 = 0.5
    #   ε_x = ε_y = -ν * ε_z = -0.25       avg_ux = avg_uy = -0.25 * 0.5 = -0.125

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics", "quasistatic", "cube-neumann-bc")
    mktempdir() do dir
        cp_example(joinpath(example_dir, "cube.g"),    joinpath(dir, "cube.g"))
        cp_example(joinpath(example_dir, "cube.yaml"), joinpath(dir, "cube.yaml"))
        sim = Carina.run(joinpath(dir, "cube.yaml"))
        avg = average_components(sim)

        @test avg[3] ≈  0.5   rtol=1e-6   # avg u_z (Norma: 0.500)
        @test avg[1] ≈ -0.125 rtol=1e-6   # avg u_x (Poisson; Norma: -0.125)
        @test avg[2] ≈ -0.125 rtol=1e-6   # avg u_y (Poisson; Norma: -0.125)
    end
end
