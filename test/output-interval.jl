# Output on a time period: the top-level `output interval` key (seconds).
#
# The controller places its stops on the output grid and the integrator
# subcycles between them with its own time step, trimming the last substep
# to land exactly on each stop.  Exodus writes are host-serial and cost
# ~1-1.5 s/step on the large GPU benchmarks, so long runs keep a sparse
# frame record by setting the interval well above the time step.
#
# Two guarantees exercised here:
#   * a period that does not divide the span never truncates the run --
#     the last interval is partial and its stop is clamped to final time
#     (`round` in the old stop count ended a 4e-4 s run at 3e-4 s);
#   * bad values (zero, negative, non-numeric, Bool) fail loudly.

@testset "Output interval" begin

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics",
                           "quasistatic", "cube")

    interval_yaml(interval_line) = """
type: single
input mesh file: cube.g
output mesh file: cube_oi.e
$interval_line
output:
  stress: false
  deformation gradient: false
model:
  type: solid mechanics
  material:
    blocks:
      cube: neohookean
    neohookean:
      elastic modulus: 1.0e9
      Poisson's ratio: 0.25
      density: 1000.0
time integrator:
  type: quasi static
  initial time: 0.0
  final time: 1.0
  time step: 0.1
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
      function: "1.0e-3 * t"
solver:
  type: newton
  termination:
    fail when any:
      - maximum iterations: 30
    converge when any:
      - absolute residual: 1.0e-6
      - relative residual: 1.0e-10
  linear solver:
    type: direct
"""

    function run_interval(interval_line)
        mktempdir() do dir
            cp_example(joinpath(example_dir, "cube.g"), joinpath(dir, "cube.g"))
            path = joinpath(dir, "cube_oi.yaml")
            open(io -> write(io, interval_yaml(interval_line)), path, "w")
            dict = Carina.YAML.load_file(path; dicttype = Dict{String, Any})
            sim  = Carina.create_simulation(dict, dir)
            Carina.evolve!(sim)
            Carina.FEC.close(sim.post_processor)

            out = joinpath(dir, "cube_oi.e")
            exo = Carina.Exodus.ExodusDatabase(out, "r")
            try
                times = Carina.Exodus.read_times(exo)
                last_step = length(times)
                uz = Carina.Exodus.read_values(
                    exo, Carina.Exodus.NodalVariable, last_step, "displ_z")
                return sim, times, uz
            finally
                Carina.close(exo)
            end
        end
    end

    @testset "default: one frame per time step" begin
        _, times, _ = run_interval("")
        @test times ≈ collect(0.0:0.1:1.0) atol = 1.0e-12
    end

    @testset "commensurate period groups steps between frames" begin
        sim, times, uz = run_interval("output interval: 0.5")
        @test times ≈ [0.0, 0.5, 1.0] atol = 1.0e-12
        # The last frame is the converged final state, not a stale one.
        u_sim = reshape(Vector(sim.params.field.data), 3, :)
        @test uz ≈ u_sim[3, :] atol = 1.0e-14 * max(1.0, maximum(abs, u_sim))
    end

    @testset "non-commensurate period reaches final time via a partial interval" begin
        sim, times, uz = run_interval("output interval: 0.4")
        @test times ≈ [0.0, 0.4, 0.8, 1.0] atol = 1.0e-12
        # The whole span was simulated: prescribed stretch 1e-3 * t at t = 1.
        @test maximum(abs, uz) ≈ 1.0e-3 rtol = 1.0e-6
    end

    @testset "period past the end writes initial and final only" begin
        _, times, _ = run_interval("output interval: 100.0")
        @test times ≈ [0.0, 1.0] atol = 1.0e-12
    end

    @testset "bad intervals fail loudly" begin
        function parse_only(interval_line)
            mktempdir() do dir
                cp_example(joinpath(example_dir, "cube.g"),
                           joinpath(dir, "cube.g"))
                path = joinpath(dir, "cube_oi.yaml")
                open(io -> write(io, interval_yaml(interval_line)), path, "w")
                dict = Carina.YAML.load_file(path; dicttype = Dict{String, Any})
                Carina.create_simulation(dict, dir)
            end
        end
        @test_throws ErrorException parse_only("output interval: 0.0")
        @test_throws ErrorException parse_only("output interval: -0.5")
        @test_throws ErrorException parse_only("output interval: \"0.5\"")
        @test_throws ErrorException parse_only("output interval: true")
    end

end
