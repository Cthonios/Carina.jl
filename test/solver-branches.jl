# Solver and integrator branches no bundled input selects.
#
# Every YAML in examples/ picks the same happy path per solver: Newton always
# line-searches, NLCG always has a Jacobi preconditioner and never restarts,
# nothing requests incomplete Cholesky, HHT damping, or a CFL-driven stable
# time step, and no *dynamic* step ever failed (the adaptive-stepping test is
# quasi-static).  Each testset here flips exactly one of those switches on a
# small cube and asserts the run still reaches the documented answer.

@testset "Solver and integrator branches" begin

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics",
                           "quasistatic", "cube")

    # Uniaxial 1e-3 stretch of the unit cube: the analytic answers the other
    # cube tests use (max u_z = prescribed, avg u_z = half of it).
    qs_yaml(solver_block) = """
type: single
input mesh file: cube.g
output mesh file: cube_branch.e
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
  time step: 1.0
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
$solver_block
"""

    function run_yaml(dir, name, yaml)
        cp_example(joinpath(example_dir, "cube.g"), joinpath(dir, "cube.g"))
        path = joinpath(dir, name)
        open(io -> write(io, yaml), path, "w")
        return Carina.run(path)
    end

    @testset "Newton without line search" begin
        # `use line search` defaults to true, so the plain
        # apply_increment!-then-evaluate! branch of the Newton loop never ran.
        yaml = qs_yaml("""
solver:
  type: newton
  use line search: false
  termination:
    fail when any:
      - maximum iterations: 20
    converge when any:
      - absolute residual: 1.0e-8
      - relative residual: 1.0e-10
  linear solver:
    type: direct
""")
        mktempdir() do dir
            sim = run_yaml(dir, "no_ls.yaml", yaml)
            @test sim.integrator.failed[] == false
            @test maximum_components(sim)[3] ≈ 1.0e-3 rtol = 1e-6
        end
    end

    @testset "line-search exhaustion falls back to the full step" begin
        # With a zero backtracking budget the Armijo loop body never runs and
        # the restore-and-accept-α=1 tail executes on every Newton iteration.
        # The problem is mildly nonlinear, so full steps still converge.
        yaml = qs_yaml("""
solver:
  type: newton
  use line search: true
  line search maximum iterations: 0
  termination:
    fail when any:
      - maximum iterations: 20
    converge when any:
      - absolute residual: 1.0e-8
      - relative residual: 1.0e-10
  linear solver:
    type: direct
""")
        mktempdir() do dir
            sim = run_yaml(dir, "ls_exhausted.yaml", yaml)
            @test sim.integrator.failed[] == false
            @test maximum_components(sim)[3] ≈ 1.0e-3 rtol = 1e-6
        end
    end

    @testset "NLCG without preconditioner, with periodic restart" begin
        # Covers the identity-preconditioner apply and the restart_interval
        # branch that zeroes β.
        yaml = qs_yaml("""
solver:
  type: nonlinear cg
  preconditioner:
    type: none
  restart interval: 2
  line search maximum iterations: 30
  use line search: true
  termination:
    fail when any:
      - maximum iterations: 500
    converge when any:
      - absolute residual: 1.0e-6
      - relative residual: 1.0e-12
""")
        mktempdir() do dir
            sim = run_yaml(dir, "nlcg_none.yaml", yaml)
            @test sim.integrator.failed[] == false
            @test maximum_components(sim)[3] ≈ 1.0e-3 rtol = 1e-5
        end
    end

    @testset "incomplete Cholesky preconditioner (quasi-static)" begin
        yaml = qs_yaml("""
solver:
  type: newton
  termination:
    fail when any:
      - maximum iterations: 20
    converge when any:
      - absolute residual: 1.0e-8
      - relative residual: 1.0e-10
  linear solver:
    type: iterative
    tolerance: 1.0e-10
    maximum iterations: 500
    preconditioner:
      type: ic
""")
        mktempdir() do dir
            sim = run_yaml(dir, "qs_ic.yaml", yaml)
            @test sim.integrator.failed[] == false
            @test maximum_components(sim)[3] ≈ 1.0e-3 rtol = 1e-6
        end
    end

    # ---- dynamic variants ---------------------------------------------------

    newmark_yaml(ti_extra, solver_block) = """
type: single
input mesh file: cube.g
output mesh file: cube_branch_dyn.e
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
  type: newmark
  initial time: 0.0
  final time: 2.0e-5
  time step: 1.0e-5
  beta: 0.25
  gamma: 0.5
$ti_extra
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
      function: "1.0e-3 * t / 2.0e-5"
$solver_block
"""

    newton_direct = """
solver:
  type: newton
  termination:
    fail when any:
      - maximum iterations: 20
    converge when any:
      - absolute residual: 1.0e-6
      - relative residual: 1.0e-10
  linear solver:
    type: direct
"""

    @testset "incomplete Cholesky preconditioner (Newmark)" begin
        solver = """
solver:
  type: newton
  termination:
    fail when any:
      - maximum iterations: 20
    converge when any:
      - absolute residual: 1.0e-6
      - relative residual: 1.0e-10
  linear solver:
    type: iterative
    tolerance: 1.0e-10
    maximum iterations: 500
    preconditioner:
      type: ic
"""
        mktempdir() do dir
            sim = run_yaml(dir, "newmark_ic.yaml", newmark_yaml("", solver))
            @test sim.integrator.failed[] == false
            @test all(isfinite, sim.params.field.data)
            @test maximum_components(sim)[3] ≈ 1.0e-3 rtol = 1e-3
        end
    end

    @testset "HHT damping (alpha != 0)" begin
        # alpha != 0 blends the previous internal force into the residual and
        # keeps F_int_n current in _finalize_step! -- both paths off with the
        # default alpha = 0.
        mktempdir() do dir
            sim = run_yaml(dir, "newmark_hht.yaml",
                           newmark_yaml("  alpha: -0.05", newton_direct))
            @test sim.integrator.failed[] == false
            @test sim.integrator.α_hht == -0.05
            @test all(isfinite, sim.params.field.data)
            @test maximum_components(sim)[3] ≈ 1.0e-3 rtol = 1e-3
        end
    end

    @testset "CFL-driven stable time step (central difference)" begin
        # `CFL` + `stable time step interval` re-estimates the stable step
        # every N steps inside _pre_step_hook!; nothing ever set the interval.
        # c = sqrt(E/rho) = 1000 m/s, h = 0.25 => stable dt ~ CFL * 2.5e-4.
        # Start well below it so the hook is what grows the step.
        yaml = """
type: single
input mesh file: cube.g
output mesh file: cube_cfl.e
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
  type: central difference
  initial time: 0.0
  final time: 2.0e-5
  time step: 1.0e-6
  minimum time step: 1.0e-6
  maximum time step: 1.0e-4
  decrease factor: 0.5
  increase factor: 1.1
  gamma: 0.5
  CFL: 0.5
  stable time step interval: 3
boundary conditions:
  dirichlet:
    - side set: ssz-
      component: z
      function: "0.0"
    - side set: ssz+
      component: z
      function: "1.0e-4 * t"
"""
        mktempdir() do dir
            sim = run_yaml(dir, "cfl.yaml", yaml)
            @test sim.integrator.failed[] == false
            # The hook jumped the step to min(stable, max) at its interval.
            # The adaptive increase factor alone could only have reached
            # 1e-6 * 1.1^20 ~ 6.7e-6 in this window, so a step above that is
            # unambiguous evidence the CFL hook ran.
            @test sim.integrator.time_step > 1.0e-5
            @test sim.integrator.time_step <= 1.0e-4
            @test all(isfinite, sim.params.field.data)
        end
    end

    @testset "Newmark step failure rolls back and retries" begin
        # The dynamic _save_state!/_restore_state! (U, V, A) never executed:
        # the adaptive-stepping test is quasi-static.  Same recipe -- a stretch
        # Newton cannot converge in 3 iterations at the nominal step -- on a
        # Newmark integrator with adaptive stepping.
        yaml = """
type: single
input mesh file: cube.g
output mesh file: cube_newmark_adapt.e
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
  type: newmark
  initial time: 0.0
  final time: 1.0
  time step: 1.0
  minimum time step: 0.001
  maximum time step: 1.0
  decrease factor: 0.5
  increase factor: 1.5
  beta: 0.25
  gamma: 0.5
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
        mktempdir() do dir
            sim = run_yaml(dir, "newmark_adapt.yaml", yaml)
            @test sim.integrator.failed[] == false
            # A rollback happened: the step was reduced and never recovered
            # to the nominal 1.0 within the simulated window.
            @test sim.integrator.time_step < 1.0
            @test maximum_components(sim)[3] ≈ 0.5 rtol = 1e-5
            @test all(isfinite, sim.params.field.data)
        end
    end

    @testset "element eversion fails loudly, never converges silently" begin
        # The +z face is driven 1.2 below the fixed -z face -- a target state
        # that everts the cube.  Neither formulation objects on its own:
        # neohookean is algebraically smooth through J = 0 (cbrt accepts
        # negative arguments), and the small-strain linear operator is well
        # defined for ANY displacement -- so before the eversion guard both
        # CONVERGED to an inside-out equilibrium.  The guard NaN-poisons the
        # residual at any J <= 0 quadrature point, so each attempt is a step
        # failure; adaptive stepping shrinks dt until the loud cannot-reduce
        # error ends the run.
        eversion_yaml(material) = """
type: single
input mesh file: cube.g
output mesh file: cube_eversion.e
model:
  type: solid mechanics
  material:
    blocks:
      cube: $material
    $material:
      elastic modulus: 1.0e9
      Poisson's ratio: 0.3
      density: 1000.0
time integrator:
  type: quasi static
  initial time: 0.0
  final time: 1.0
  time step: 1.0
  minimum time step: 0.05
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
      function: "-1.2 * t"
solver:
  type: newton
  termination:
    fail when any:
      - maximum iterations: 15
    converge when any:
      - relative residual: 1.0e-8
  linear solver:
    type: direct
"""
        for material in ("neohookean", "linear elastic")
            mktempdir() do dir
                cp_example(joinpath(example_dir, "cube.g"), joinpath(dir, "cube.g"))
                path = joinpath(dir, "eversion.yaml")
                open(io -> write(io, eversion_yaml(material)), path, "w")
                err = try
                    Carina.run(path)
                    nothing
                catch e
                    sprint(showerror, e)
                end
                @test err !== nothing
                @test err !== nothing && occursin("Cannot reduce time step", err)
            end
        end
    end

end
