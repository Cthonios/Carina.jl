@testset "Mechanics Quasi-static Cube (Gravity)" begin
    # Unit cube [-0.5, 0.5]³, linear elastic, E=1e9, ν=0.
    # Symmetry BCs on all faces except z=+0.5 (free).
    # Body force: b_z = -ρg = -9810 N/m³.
    #
    # With ν=0, the problem is 1-D (uniaxial strain in z).
    # The hex8 basis represents the quadratic solution exactly.
    #
    # Analytical (ζ = z + 0.5):
    #   u_z(ζ) = -ρg·ζ·(1 - ζ/2) / E
    #   u_z(z=+0.5) = -ρgL²/(2E) = -4.905e-6 m
    #   u_x = u_y = 0

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics", "quasistatic", "cube-gravity")
    mktempdir() do dir
        cp_example(joinpath(example_dir, "cube.g"),    joinpath(dir, "cube.g"))
        cp_example(joinpath(example_dir, "cube.yaml"), joinpath(dir, "cube.yaml"))
        sim = Carina.run(joinpath(dir, "cube.yaml"))

        uz_min = minimum_components(sim)[3]
        avg    = average_components(sim)

        @test uz_min ≈ -4.905e-6 atol=1e-14   # exact to machine precision
        @test avg[1] ≈  0.0      atol=1e-14   # no lateral displacement
        @test avg[2] ≈  0.0      atol=1e-14
    end
end
