@testset "Mechanics Torsion" begin
    # Cylinder: R = 25 mm, L = 1 m, neo-Hookean (E=1e9 Pa, ν=0.25, ρ=1000 kg/m³).
    # Initial velocity: rigid-body torsion about z, ω(z) = a·z with a = 8000 s⁻¹.
    #   v_x = -a·y·z,  v_y = a·x·z,  v_z = 0
    #
    # Both integrators run to t = 2e-6 s (small enough that elastic waves have
    # barely propagated, so explicit and implicit should agree closely).
    #
    # Explicit (central difference, Δt = 5e-7 s, 4 steps) is the reference.
    # Implicit (Newmark L-BFGS, Δt = 2e-6 s, 1 step) is compared against it.
    #
    # Analytical estimate at t = 2e-6 (pure kinematic, no wave reflection):
    #   max|u_x| = max|u_y| = a·R·(L/2)·t = 8000·0.025·0.5·2e-6 = 2.0e-4 m
    #   max displacement magnitude ≈ √2 · 2e-4 ≈ 2.828e-4 m
    #
    # The test runs on the best available device (ROCm → CUDA → CPU).

    device = Carina.best_device()

    explicit_dir = joinpath(@__DIR__, "..", "examples", "mechanics",
                            "explicit-dynamic", "torsion")
    implicit_dir = joinpath(@__DIR__, "..", "examples", "mechanics",
                            "implicit-dynamic", "torsion")

    # ------------------------------------------------------------------ #
    # Reference: explicit central difference
    # ------------------------------------------------------------------ #
    ref_max_ux = ref_max_uy = ref_max_mag = 0.0

    @testset "Explicit central difference (reference)" begin
        mktempdir() do dir
            cp_example(joinpath(explicit_dir, "torsion.g"),          joinpath(dir, "torsion.g"))
            cp_example(joinpath(explicit_dir, "torsion_explicit.yaml"), joinpath(dir, "torsion_explicit.yaml"))
            sim = Carina.run(joinpath(dir, "torsion_explicit.yaml"); device=device)

            mx  = maximum_components(sim)
            mag = maximum_magnitude(sim)

            # Kinematic estimate: max|u_x| = max|u_y| = a·R·(L/2)·t = 2e-4 m
            @test mx[1] ≈ 2.0e-4 rtol=1e-2
            @test mx[2] ≈ 2.0e-4 rtol=1e-2
            @test abs(mx[3]) < 1e-6          # u_z ≈ 0 for pure torsion

            # Store for cross-integrator comparison below.
            ref_max_ux  = mx[1]
            ref_max_uy  = mx[2]
            ref_max_mag = mag
        end
    end

    # ------------------------------------------------------------------ #
    # Implicit: Newmark L-BFGS
    # ------------------------------------------------------------------ #
    @testset "Implicit Newmark L-BFGS vs explicit" begin
        mktempdir() do dir
            cp_example(joinpath(implicit_dir, "torsion.g"),        joinpath(dir, "torsion.g"))
            cp_example(joinpath(implicit_dir, "torsion_lbfgs.yaml"), joinpath(dir, "torsion_lbfgs.yaml"))
            sim = Carina.run(joinpath(dir, "torsion_lbfgs.yaml"); device=device)

            mx  = maximum_components(sim)
            mag = maximum_magnitude(sim)

            @test mx[1] ≈ ref_max_ux  rtol=1e-2
            @test mx[2] ≈ ref_max_uy  rtol=1e-2
            @test mag   ≈ ref_max_mag rtol=1e-2
        end
    end
end
