@testset "Mechanics Quasi-static Cube (Point Load)" begin
    # Same problem as the Neumann BC test but with the traction replaced by
    # equivalent point loads on node set nsz+.
    #
    # Single hex8 element [−0.5,0.5]³.  The z+ face has 4 nodes, each with
    # tributary area 0.25 (bilinear shape functions, uniform quad).
    #
    # Original traction: function = "1e9·t" on ssz+ (positive = tension in +z).
    # Equivalent point load per node: 0.25e9·t N (total force / 4 nodes).
    #
    # Analytical: ε_z = 1, avg u_z = 0.5, avg u_x = avg u_y = −0.125.

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics", "quasistatic", "cube-neumann-bc")
    mktempdir() do dir
        cp_example(joinpath(example_dir, "cube.g"), joinpath(dir, "cube.g"))

        yaml = """
type: single

input mesh file:  cube.g
output mesh file: cube.e

model:
  type: solid mechanics
  material:
    blocks:
      cube: linear elastic
    linear elastic:
      elastic modulus: 1.0e9
      Poisson's ratio: 0.25
      density: 1000.0

time integrator:
  type: quasi static
  initial time: 0.0
  final time:   1.0
  time step:    0.1

boundary conditions:
  Dirichlet:
    - side set:   ssx-
      component: x
      function:  "0.0"
    - side set:   ssy-
      component: y
      function:  "0.0"
    - side set:   ssz-
      component: z
      function:  "0.0"
  Neumann:
    - node set:  nsz+
      component: z
      function:  "0.25e+09 * t"

solver:
  type: newton
  termination:
    - type: combo
      combo: or
      tests:
        - type: absolute residual
          tolerance: 1.0e-08
        - type: relative residual
          tolerance: 1.0e-14
    - type: maximum iterations
      value: 16
  linear solver:
    type: direct
"""
        write(joinpath(dir, "cube_point_load.yaml"), yaml)
        sim = Carina.run(joinpath(dir, "cube_point_load.yaml"))
        avg = average_components(sim)

        @test avg[3] ≈  0.5   rtol=1e-6   # avg u_z (tension)
        @test avg[1] ≈ -0.125 rtol=1e-6   # avg u_x (Poisson contraction)
        @test avg[2] ≈ -0.125 rtol=1e-6   # avg u_y (Poisson contraction)
    end
end
