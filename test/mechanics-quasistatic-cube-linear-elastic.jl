@testset "Mechanics Quasi-static Cube (Linear Elastic)" begin
    # Unit cube [0,1]³, linear elastic (infinitesimal strain), E=1e9, ν=0.25.
    # BCs: u_x=0 on x=0, u_y=0 on y=0, u_z=0 on z=0, u_z=1.0*t on z=1.
    # Final time t=1.0 → applied strain ε_z = 1.0.
    #
    # Matches Norma single-static-solid-cube (E=1e9, same BCs, same time stepping).
    #
    # Analytical solution (exact for linear elastic FEM on any mesh):
    #   u_z(z) = z              avg_uz = 0.5
    #   u_x(x) = -0.25*x       avg_ux = -0.125
    #   u_y(y) = -0.25*y       avg_uy = -0.125
    #
    # Linear elastic FEM is exact for linear displacement fields on any mesh;
    # tolerances match Norma's (rtol=1e-6).

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics", "quasistatic", "cube-linear-elastic")
    mktempdir() do dir
        cp_example(joinpath(example_dir, "cube.g"),    joinpath(dir, "cube.g"))
        cp_example(joinpath(example_dir, "cube.yaml"), joinpath(dir, "cube.yaml"))
        sim = Carina.run(joinpath(dir, "cube.yaml"))
        avg = average_components(sim)
        mx  = maximum_components(sim)

        @test avg[3] ≈  0.5   rtol=1e-6   # avg u_z (exact; Norma: 0.500)
        @test avg[1] ≈ -0.125 rtol=1e-6   # avg u_x (Poisson, exact; Norma: -0.125)
        @test avg[2] ≈ -0.125 rtol=1e-6   # avg u_y (Poisson, exact; Norma: -0.125)
        @test mx[3]  ≈  1.0   rtol=1e-6   # max u_z = prescribed BC (exact)
    end
end
