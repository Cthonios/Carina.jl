@testset "Mechanics Quasi-static Cube (Linear Elastic)" begin
    # Unit cube [0,1]³, linear elastic (infinitesimal strain), E=10e9, ν=0.25.
    # BCs: u_x=0 on x=0, u_y=0 on y=0, u_z=0 on z=0, u_z=1e-3*t on z=1.
    # Final time t=1.0 → applied strain ε_z = 1e-3.
    #
    # Analytical solution (exact for linear elastic):
    #   Uniaxial stress in z, lateral faces free → Poisson contraction.
    #
    #   u_z(z) = ε_z * z = 1e-3 * z       avg_uz = 1e-3 * 0.5 = 5.00e-4
    #   u_x(x) = -ν * ε_z * x             avg_ux = -0.25 * 1e-3 * 0.5 = -1.25e-4
    #   u_y(y) = -ν * ε_z * y             avg_uy = -1.25e-4
    #
    # Linear elastic FEM is exact for linear displacement fields on any mesh,
    # so all tolerances are tight.

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics", "quasistatic", "cube-linear-elastic")
    mktempdir() do dir
        cp(joinpath(example_dir, "cube.g"),    joinpath(dir, "cube.g"))
        cp(joinpath(example_dir, "cube.yaml"), joinpath(dir, "cube.yaml"))
        sim = Carina.run(joinpath(dir, "cube.yaml"))
        avg = average_components(sim)
        mx  = maximum_components(sim)

        @test avg[3] ≈  5.00e-4 rtol=1e-6   # avg u_z (exact)
        @test avg[1] ≈ -1.25e-4 rtol=1e-4   # avg u_x (Poisson, exact)
        @test avg[2] ≈ -1.25e-4 rtol=1e-4   # avg u_y (Poisson, exact)
        @test mx[3]  ≈  1.00e-3 rtol=1e-6   # max u_z = prescribed BC (exact)
    end
end
