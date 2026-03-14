@testset "Mechanics Dynamic Cube" begin
    # Same unit cube as the quasi-static test, driven by Newmark-β (β=0.25, γ=0.5).
    # E=10e9, ν=0.25, density=1000.  Applied u_z = 1e-3*t on z=1 face.
    # Final time t=0.1, Δt=0.01.
    #
    # Wave speed and time scales:
    #   c_s = sqrt(E/ρ) = sqrt(10e9/1000) ≈ 3162 m/s
    #   Lowest mode period: T₁ = 2L/c_s ≈ 6.3e-4 s
    #   Δt/T₁ ≈ 16  →  Newmark is in the quasi-static regime; response tracks
    #   the static solution at each time step.
    #
    # At t=0.1:  applied ε_z = 1e-3 * 0.1 = 1e-4
    #   max u_z = 1e-4  (prescribed BC — exact by construction)
    #   avg u_z ≈ 5e-5  (quasi-static: ε_z * avg(z) = 1e-4 * 0.5)
    #   avg u_x ≈ -1.25e-5, avg u_y ≈ -1.25e-5  (Poisson)
    #
    # Tolerances: max u_z is tight (rtol=1e-6, set by BC); averages are loose
    # (rtol=1e-2) to allow for residual dynamic effects.

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics", "implicit-dynamic", "cube")
    mktempdir() do dir
        cp(joinpath(example_dir, "cube.g"),    joinpath(dir, "cube.g"))
        cp(joinpath(example_dir, "cube.yaml"), joinpath(dir, "cube.yaml"))
        sim = Carina.run(joinpath(dir, "cube.yaml"))
        avg = average_components(sim)
        mx  = maximum_components(sim)

        @test mx[3]  ≈  1.00e-4 rtol=1e-6   # max u_z = prescribed BC (exact)
        @test avg[3] ≈  5.00e-5 rtol=1e-2   # avg u_z (quasi-static limit)
        @test avg[1] ≈ -1.25e-5 rtol=1e-2   # avg u_x (Poisson)
        @test avg[2] ≈ -1.25e-5 rtol=1e-2   # avg u_y (Poisson)
    end
end
