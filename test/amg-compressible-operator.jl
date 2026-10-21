# Carina: Copyright 2026 National Technology & Engineering Solutions of
# Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS,
# the U.S. Government retains certain rights in this software. This software
# is released under the BSD license detailed in the file LICENSE.md in the
# top-level Carina.jl directory.

@testset "AMG compressible operator" begin
    # The key `operator: compressible` of the AMG preconditioner builds the
    # hierarchy on the tangent of the material with its bulk modulus set to
    # the shear modulus (λ' = μ/3) and applies it to the tangent itself, so
    # the converged solution is unchanged (research/tet15-p1/precond).  The
    # quasi-static cube is solved on the CPU, assembled CG with AMG, with
    # `tangent` and with `compressible`, at the near-incompressible J2
    # material of the punch benchmark (κ/μ = 1e4); bad input fails loudly;
    # the Newmark integrator rejects the key.
    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics", "quasistatic", "cube")
    deck(operator) = """
type: single
input mesh file: cube.g
output mesh file: cube-amg-$operator.e
model:
  type: solid mechanics
  material:
    blocks:
      cube: j2 plasticity
    j2 plasticity:
      elastic modulus: 2.9999e9
      Poisson's ratio: 0.49995
      density: 8930.0
      yield stress: 1.0e6
      hardening modulus: 1.0e7
time integrator:
  type: quasi static
  time step: 0.5
  final time: 1.0
  initial time: 0.0
  predictor: tangent
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
  termination:
    fail when any:
      - maximum iterations: 30
    converge when any:
      - absolute residual: 1.0e-2
      - relative residual: 1.0e-8
  linear solver:
    type: iterative
    tolerance: 1.0e-10
    maximum iterations: 5000
    preconditioner:
      type: amg
      operator: $operator
"""
    mktempdir() do dir
        cp_example(joinpath(example_dir, "cube.g"), joinpath(dir, "cube.g"))
        results = Dict{String, Any}()
        for operator in ("tangent", "compressible")
            path = joinpath(dir, "cube-amg-$operator.yaml")
            write(path, deck(operator))
            sim = Carina.run(path)
            results[operator] = (average_components(sim), maximum_components(sim))
            @test sim.integrator.nonlinear_solver.linear_solver.precond isa Carina.AMGPreconditioner
            @test sim.integrator.nonlinear_solver.linear_solver.precond.compressible ==
                  (operator == "compressible")
        end
        avg_t, mx_t = results["tangent"]
        avg_c, mx_c = results["compressible"]
        @test avg_c[3] ≈ avg_t[3] rtol=1e-8
        @test mx_c[3]  ≈ mx_t[3]  rtol=1e-8
        @test any(!iszero, avg_t)

        # Bad input fails loudly.
        bad = joinpath(dir, "cube-amg-bad.yaml")
        write(bad, deck("volumetric"))
        @test_throws ErrorException Carina.run(bad)

        # The Newmark integrator rejects the key.
        newmark = replace(deck("compressible"),
                          "  type: quasi static\n  time step: 0.5\n  final time: 1.0\n  initial time: 0.0\n  predictor: tangent\n" =>
                          "  type: newmark\n  time step: 0.01\n  final time: 0.02\n  initial time: 0.0\n  beta: 0.25\n  gamma: 0.5\n")
        path = joinpath(dir, "cube-amg-newmark.yaml")
        write(path, newmark)
        @test_throws ErrorException Carina.run(path)
    end
end
