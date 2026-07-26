@testset "Mechanics Quasi-static Cube (Steepest Descent)" begin
    # examples/mechanics/quasistatic/cube/cube_sd.yaml has existed alongside the
    # other cube variants, and test/input-validation.jl asserts that
    # `type: steepest descent` parses into a SteepestDescentSolver -- but nothing
    # ever ran one, so `solve!(::SteepestDescentSolver, ...)` was dead in the test
    # suite. That solver body holds the Armijo backtracking line search on energy,
    # and it is the only caller of `_compute_energy`, hence the only path that
    # evaluates `FEC.energy`.
    #
    # Writing this test is what exposed the example's unreachable tolerances: it
    # asked for a relative residual of 1e-14 while the energy line search stalls
    # near 1e-8 (the Armijo test cannot resolve an energy decrease below the
    # round-off of the energy itself). The example now asks for 1e-6.

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics", "quasistatic", "cube")

    mktempdir() do dir
        cp_example(joinpath(example_dir, "cube.g"),       joinpath(dir, "cube.g"))
        cp_example(joinpath(example_dir, "cube_sd.yaml"), joinpath(dir, "cube_sd.yaml"))
        path = joinpath(dir, "cube_sd.yaml")

        # Build first so the undeformed energy can be sampled before stepping;
        # one run serves both the solution and the energy assertions.
        dict = Carina.YAML.load_file(path; dicttype=Dict{String,Any})
        sim  = Carina.create_simulation(dict, dir)
        ig   = sim.integrator

        @test ig.nonlinear_solver isa Carina.SteepestDescentSolver

        # The Armijo condition the line search enforces is a decrease in total
        # potential energy, so the energy functional has to be sane: zero for the
        # undeformed state.
        W_undeformed = Carina._compute_energy(ig, sim.params)
        @test W_undeformed ≈ 0.0 atol=1e-8

        Carina.evolve!(sim)
        Carina.FEC.close(sim.post_processor)

        # The run completed rather than exhausting its iteration budget and
        # failing the step -- which is what the shipped tolerances used to do.
        @test ig.failed[] == false

        W_strained = Carina._compute_energy(ig, sim.params)
        @test isfinite(W_strained)
        @test W_strained > 0.0

        avg = average_components(sim)
        mx  = maximum_components(sim)

        # Same uniaxial stretch as the Newton cube tests: u_z = 1.0e-3 prescribed
        # on the +z face. Steepest descent solves the same problem and must reach
        # the same answer -- looser tolerances only because it converges linearly
        # and stops at a relative residual of 1e-6 rather than 1e-10.
        @test avg[3] ≈  5.00e-4 rtol=1e-3   # avg u_z (analytical)
        @test avg[1] ≈ -1.25e-4 rtol=5e-2   # avg u_x (Poisson)
        @test avg[2] ≈ -1.25e-4 rtol=5e-2   # avg u_y (Poisson)
        @test mx[3]  ≈  1.00e-3 rtol=1e-6   # max u_z = prescribed BC (exact)
    end
end
