@testset "Adaptive stepping and step-failure recovery" begin
    # Nothing in the suite ever made a time step fail, so the whole recovery
    # path was dead: the retry loop in `_advance_one_step!`, `_save_state!` /
    # `_restore_state!`, `_decrease_step!`, `_increase_step!`, and
    # `_parse_adaptive_stepping`. That is the code which runs precisely when a
    # simulation is in trouble, which is the worst place to have no coverage.

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics", "quasistatic", "cube")

    # A cube stretched to 50% engineering strain in a single step. Newton needs
    # more than three iterations to converge that, so the first attempt trips the
    # `maximum iterations: 3` failure test; after the step is halved the reduced
    # increment converges and the run completes. The margin is deliberate -- at
    # `maximum iterations: 4` this problem never fails at all, so the test is not
    # balanced on a one-iteration knife edge.
    adaptive_yaml = """
type: single

input mesh file: cube.g
output mesh file: cube_adaptive.e

model:
  type: solid mechanics
  material:
    blocks:
      cube: neohookean
    neohookean:
      elastic modulus: 1.0e9
      Poisson's ratio: 0.3
      density: 1000.0

time integrator:
  type: quasi static
  initial time: 0.0
  final time: 1.0
  time step: 1.0
  minimum time step: 0.001
  maximum time step: 1.0
  decrease factor: 0.5
  increase factor: 1.5

boundary conditions:
  dirichlet:
    - side set: ssx-
      component: x
      function: "0.0"
    - side set: ssy-
      component: y
      function: "0.0"
    - side set: ssz-
      component: z
      function: "0.0"
    - side set: ssz+
      component: z
      function: "0.5 * t"

solver:
  type: newton
  termination:
    fail when any:
      - maximum iterations: 3
    converge when any:
      - relative residual: 1.0e-8
  linear solver:
    type: direct
"""

    @testset "a failed step is retried at a smaller time step" begin
        mktempdir() do dir
            cp_example(joinpath(example_dir, "cube.g"), joinpath(dir, "cube.g"))
            path = joinpath(dir, "adaptive.yaml")
            open(io -> write(io, adaptive_yaml), path, "w")

            sim = Carina.run(path)

            # The run reached the end despite the first attempt failing.
            @test sim.integrator.failed[] == false

            # A reduction actually happened: the step never climbed back to the
            # nominal 1.0 within the remaining simulated time.
            @test sim.integrator.time_step < 1.0
            @test sim.integrator.time_step >= sim.integrator.min_time_step

            # Recovery must not corrupt the answer. The prescribed +z face
            # displacement is exact regardless of how the step was subdivided.
            mx = maximum_components(sim)
            @test mx[3] ≈ 0.5 rtol=1e-6
            @test all(isfinite, sim.params.field.data)
        end
    end

    @testset "adaptive parameters are validated as a group" begin
        # `_parse_adaptive_stepping` requires all four keys together, because a
        # partial specification silently disables adaptivity.
        base = replace(adaptive_yaml, "  minimum time step: 0.001\n" => "")
        mktempdir() do dir
            cp_example(joinpath(example_dir, "cube.g"), joinpath(dir, "cube.g"))
            path = joinpath(dir, "partial.yaml")
            open(io -> write(io, base), path, "w")
            @test_throws ErrorException Carina.run(path)
        end

        for (bad, sub) in (("decrease factor: 0.5" => "decrease factor: 1.5", "decrease"),
                           ("increase factor: 1.5" => "increase factor: 0.5", "increase"))
            mktempdir() do dir
                cp_example(joinpath(example_dir, "cube.g"), joinpath(dir, "cube.g"))
                path = joinpath(dir, "bad_$(sub).yaml")
                open(io -> write(io, replace(adaptive_yaml, bad)), path, "w")
                @test_throws ErrorException Carina.run(path)
            end
        end
    end

    @testset "step size helpers" begin
        # Direct unit coverage of the two helpers' guard rails, which the
        # integration test above cannot reach without deliberately wedging a run.
        mktempdir() do dir
            cp_example(joinpath(example_dir, "cube.g"), joinpath(dir, "cube.g"))
            path = joinpath(dir, "helpers.yaml")
            open(io -> write(io, adaptive_yaml), path, "w")
            dict = Carina.YAML.load_file(path; dicttype=Dict{String,Any})
            sim  = Carina.create_simulation(dict, dir)
            ig   = sim.integrator
            p    = sim.params

            # _increase_step! grows by the factor but never past the maximum.
            ig.time_step = 0.1
            Carina._increase_step!(ig, p)
            @test ig.time_step ≈ 0.15

            ig.time_step = 0.9
            Carina._increase_step!(ig, p)
            @test ig.time_step ≈ ig.max_time_step

            # _decrease_step! shrinks by its factor ...
            ig.time_step = 0.4
            Carina._decrease_step!(ig, p)
            @test ig.time_step ≈ 0.2

            # ... and refuses to go below the floor rather than grinding the run
            # down to a zero-length step.
            ig.time_step = ig.min_time_step
            @test_throws ErrorException Carina._decrease_step!(ig, p)

            # With adaptivity disabled there is no recovery to attempt, so a
            # failed step must be reported rather than silently retried forever.
            ig.decrease_factor = 1.0
            ig.time_step = 0.4
            @test_throws ErrorException Carina._decrease_step!(ig, p)

            # increase_factor == 1.0 means "no adaptivity": a no-op, not growth.
            ig.increase_factor = 1.0
            ig.time_step = 0.4
            Carina._increase_step!(ig, p)
            @test ig.time_step ≈ 0.4
        end
    end

    @testset "state is restored exactly on rollback" begin
        # The retry loop depends on `_restore_state!` putting the integrator back
        # where `_save_state!` left it -- displacement, field, and the internal
        # state variables that a constitutive model may already have advanced.
        mktempdir() do dir
            cp_example(joinpath(example_dir, "cube.g"), joinpath(dir, "cube.g"))
            path = joinpath(dir, "rollback.yaml")
            open(io -> write(io, adaptive_yaml), path, "w")
            dict = Carina.YAML.load_file(path; dicttype=Dict{String,Any})
            sim  = Carina.create_simulation(dict, dir)
            ig   = sim.integrator
            p    = sim.params

            Carina._save_state!(ig, p)
            U_saved     = copy(ig.U)
            field_saved = copy(p.field.data)
            state_saved = copy(p.state_new.data)

            # Perturb everything a failed step would have dirtied.
            ig.U .+= 1.0e-3
            p.field.data .+= 1.0e-3
            p.state_new.data .+= 1.0

            Carina._restore_state!(ig, p)

            @test ig.U == U_saved
            @test p.field.data == field_saved
            @test p.state_new.data == state_saved
        end
    end
end
