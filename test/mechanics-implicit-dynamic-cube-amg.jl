@testset "Mechanics Dynamic Cube (AMG Preconditioner)" begin
    # Same problem as "Mechanics Dynamic Cube" (mechanics-implicit-dynamic-cube.jl)
    # but solved with CG + smoothed-aggregation AMG (rigid-body-mode
    # near-nullspace) instead of MINRES + Jacobi.
    #
    # Verifies the AMG preconditioner produces the same physical result and
    # exercises the lagged hierarchy build across Newton steps.

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics", "implicit-dynamic", "cube")
    mktempdir() do dir
        cp_example(joinpath(example_dir, "cube.g"), joinpath(dir, "cube.g"))

        yaml_path = joinpath(dir, "cube_amg.yaml")
        open(yaml_path, "w") do io
            write(io, """
type: single
input mesh file: cube.g
output mesh file: cube_amg.e
model:
  type: solid mechanics
  material:
    blocks:
      cube: neohookean
    neohookean:
      elastic modulus: 1.0e10
      density: 1000.0
      Poisson's ratio: 0.25
time integrator:
  type: newmark
  time step: 0.01
  gamma: 0.5
  final time: 0.1
  initial time: 0.0
  beta: 0.25
boundary conditions:
  dirichlet:
    - function: "0.0"
      side set: ssx-
      component: x
    - function: "0.0"
      side set: ssy-
      component: y
    - function: "0.0"
      side set: ssz-
      component: z
    - function: "1.0e-3 * t"
      side set: ssz+
      component: z
solver:
  type: newton
  linear solver:
    type: cg
    preconditioner:
      type: amg
    maximum iterations: 500
    tolerance: 1.0e-8
  termination:
    fail when any:
      - maximum iterations: 16
    converge when any:
      - absolute residual: 1.0e-6
      - relative residual: 1.0e-10
""")
        end

        sim = Carina.run(yaml_path)
        avg = average_components(sim)
        mx  = maximum_components(sim)

        @test mx[3]  ≈  1.00e-4 rtol=1e-6   # max u_z = prescribed BC (exact)
        @test avg[3] ≈  5.00e-5 rtol=1e-2   # avg u_z (quasi-static limit)
        @test avg[1] ≈ -1.25e-5 rtol=1e-2   # avg u_x (Poisson)
        @test avg[2] ≈ -1.25e-5 rtol=1e-2   # avg u_y (Poisson)

        # The lagged build policy must have built the hierarchy exactly once
        # (constant Δt, mild nonlinearity — no staleness trigger expected).
        precond = sim.integrator.nonlinear_solver.linear_solver.precond
        @test precond isa Carina.AMGPreconditioner
        @test precond.nbuilds == 1
    end
end

@testset "Mechanics Quasi-static Cube (AMG Preconditioner)" begin
    # Same problem as mechanics-quasistatic-cube.jl but solved with
    # CG + smoothed-aggregation AMG, exercising the quasi-static assembled
    # path (_build_precond_op with c_M = 0).

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics", "quasistatic", "cube")
    mktempdir() do dir
        cp_example(joinpath(example_dir, "cube.g"), joinpath(dir, "cube.g"))

        yaml_path = joinpath(dir, "cube_qs_amg.yaml")
        open(yaml_path, "w") do io
            write(io, """
type: single
input mesh file: cube.g
output mesh file: cube_qs_amg.e
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
    - type: combo
      combo: or
      tests:
        - type: absolute residual
          tolerance: 1.0e-6
        - type: relative residual
          tolerance: 1.0e-10
    - type: maximum iterations
      value: 16
  linear solver:
    type: iterative
    tolerance: 1.0e-10
    maximum iterations: 500
    preconditioner:
      type: amg
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
