@testset "Mechanics Sphere" begin
    # Unit sphere (R=1m), 997 nodes, 864 HEX8 elements.
    # Neo-Hookean: E=1e4 Pa, ν=0.33, ρ=1000 kg/m³.
    # Initial velocity on sphere_surf: rigid-body torsion about z,
    #   v_x = -a·y·z,  v_y = a·x·z,  v_z = 0,  a = 10.0 s⁻¹
    # No Dirichlet BCs (fully unconstrained).
    #
    # Both integrators run a single step to t = 1e-2 s.
    # At this short time the wave has barely propagated (c_p ≈ 3.85 m/s,
    # so the elastic wave travels ~3.85e-2 m in 1e-2 s), so explicit and
    # implicit should agree to within ~1%.
    #
    # Kinematic estimate (first-order): max|U| ≈ max(|v|)·Δt
    #   max|v| = a·R·(R/2) = 10·1·0.5 = 5 m/s  →  max|U| ≈ 5e-2 m
    #
    # NOTE: HHT-α integrator (2nd-order algorithmic damping) is not yet
    # implemented in Carina.  Once added it should be used here in place of
    # undamped Newmark (β=0.25, γ=0.5) for production-quality comparison.
    #
    # Runs on CPU only (sphere mesh is small; GPU test is in benchmark scripts).

    explicit_dir = joinpath(@__DIR__, "..", "examples", "mechanics",
                            "explicit-dynamic", "sphere")
    implicit_dir = joinpath(@__DIR__, "..", "examples", "mechanics",
                            "implicit-dynamic", "sphere")

    # ------------------------------------------------------------------ #
    # Reference: explicit central difference
    # ------------------------------------------------------------------ #
    ref_max_mag = 0.0

    @testset "Explicit central difference (reference)" begin
        mktempdir() do dir
            cp_example(joinpath(explicit_dir, "sphere.g"),             joinpath(dir, "sphere.g"))
            cp_example(joinpath(explicit_dir, "sphere_explicit.yaml"), joinpath(dir, "sphere_explicit.yaml"))
            sim = Carina.run(joinpath(dir, "sphere_explicit.yaml"); device="cpu")

            mag = maximum_magnitude(sim)

            # Kinematic estimate: max|U| ≈ 5e-2 m
            @test mag ≈ 5.0e-2 rtol=5e-2

            ref_max_mag = mag
        end
    end

    # ------------------------------------------------------------------ #
    # Implicit: Newmark (β=0.25, γ=0.5) + direct LU
    # ------------------------------------------------------------------ #
    @testset "Implicit Newmark direct vs explicit" begin
        mktempdir() do dir
            cp_example(joinpath(implicit_dir, "sphere.g"),             joinpath(dir, "sphere.g"))
            cp_example(joinpath(implicit_dir, "sphere_implicit.yaml"), joinpath(dir, "sphere_implicit.yaml"))
            sim = Carina.run(joinpath(dir, "sphere_implicit.yaml"); device="cpu")

            mag = maximum_magnitude(sim)

            @test mag ≈ ref_max_mag rtol=1e-2
        end
    end
end
