@testset "Mechanics Quasi-static Cube (Chebyshev Preconditioner)" begin
    # Same problem as mechanics-quasistatic-cube.jl but solved with
    # CG + Chebyshev polynomial preconditioner (degree 5) instead of direct.
    #
    # Verifies that the Chebyshev preconditioner produces the same physical
    # result to within discretization tolerance.

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics", "quasistatic", "cube")
    mktempdir() do dir
        cp_example(joinpath(example_dir, "cube.g"), joinpath(dir, "cube.g"))

        # Write a YAML config with iterative solver + Chebyshev preconditioner.
        yaml_path = joinpath(dir, "cube_chebyshev.yaml")
        open(yaml_path, "w") do io
            write(io, """
type: single

input mesh file:  cube.g
output mesh file: cube_chebyshev.e

model:
  type: solid mechanics
  material:
    blocks:
      cube: neohookean
    neohookean:
      elastic modulus: 10.0e9
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
    - side set:   ssz+
      component: z
      function:  "1.0e-3 * t"

solver:
  type: newton
  termination:
    - type: combo
      combo: or
      tests:
        - type: absolute residual
          tolerance: 1.0e-06
        - type: relative residual
          tolerance: 1.0e-10
    - type: maximum iterations
      value: 16
  linear solver:
    type: iterative
    tolerance: 1.0e-10
    maximum iterations: 500
    preconditioner:
      type: chebyshev
      degree: 5
""")
        end

        sim = Carina.run(yaml_path)
        avg = average_components(sim)
        mx  = maximum_components(sim)

        @test avg[3] ≈  5.00e-4 rtol=1e-4   # avg u_z (analytical)
        @test avg[1] ≈ -1.25e-4 rtol=1e-2   # avg u_x (Poisson)
        @test avg[2] ≈ -1.25e-4 rtol=1e-2   # avg u_y (Poisson)
        @test mx[3]  ≈  1.00e-3 rtol=1e-6   # max u_z = prescribed BC (exact)
    end
end
