@testset "Mechanics Explicit Dynamic Cube" begin
    # Free-free cube (no Dirichlet BCs), initial velocity v_z = 1.0 on all nodes.
    # E=1e3, ν=0.25, density=1000.  Central difference (β=0, γ=0.5).
    # Final time t_f = 1.0, Δt = 0.01.
    #
    # Since the initial velocity is uniform (rigid-body translation), there is no
    # deformation and no internal forces throughout.  The cube translates freely:
    #   u_z(t) = v_z * t   →   avg_u_z(t_f) = 1.0 * 1.0 = 1.0
    #   u_x = u_y = 0 (no x/y velocity or forces)
    #
    # CFL stability:
    #   c_s = sqrt(E/ρ) = sqrt(1e3/1e3) = 1 m/s,  h ≈ 0.5 m
    #   Δt_crit ≈ h/c_s = 0.5 s >> Δt = 0.01 s  (very conservative)

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics", "explicit-dynamic", "cube")
    mktempdir() do dir
        cp_example(joinpath(example_dir, "cube.g"),    joinpath(dir, "cube.g"))
        cp_example(joinpath(example_dir, "cube.yaml"), joinpath(dir, "cube.yaml"))
        sim = Carina.run(joinpath(dir, "cube.yaml"))
        avg = average_components(sim)

        @test avg[3] ≈ 1.0 atol=1e-6   # avg u_z = v_z * t_f (exact; Norma: 1.0 ± 1e-6)
        @test avg[1] ≈ 0.0 atol=1e-8   # no x displacement
        @test avg[2] ≈ 0.0 atol=1e-8   # no y displacement
    end
end
