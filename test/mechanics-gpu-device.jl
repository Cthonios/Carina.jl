@testset "GPU Device Verification" begin
    # ---- BC function GPU compatibility (runs on all platforms) ----
    # @eval closures must be isbits so they can be passed as GPU kernel arguments.
    # This guards against regressions (e.g. switching to RuntimeGeneratedFunctions
    # which stores Expr and is non-isbits).
    @testset "BC functions are isbits (GPU-compatible)" begin
        f = Carina._make_function("0.005 * t")
        @test isbitstype(typeof(f))
        dbcf = Carina.FEC.DirichletBCFunction(f)
        @test isbitstype(typeof(dbcf))
    end

    # Run the explicit sphere for 1 step on the best available device.
    # If a GPU is present, verify that sim.backend is a GPU backend
    # and that the result matches CPU to machine precision.
    #
    # Uses the explicit solver (no linear solver needed) so the same
    # YAML works on both CPU and GPU.

    backend = test_best_device()
    has_gpu = !(backend isa Carina.KA.CPU)

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics",
                           "explicit-dynamic", "sphere")

    # ---- CPU baseline ----
    cpu_mag = 0.0
    mktempdir() do dir
        cp_example(joinpath(example_dir, "sphere.g"),             joinpath(dir, "sphere.g"))
        cp_example(joinpath(example_dir, "sphere_explicit.yaml"), joinpath(dir, "sphere_explicit.yaml"))
        sim = Carina.run(joinpath(dir, "sphere_explicit.yaml"); backend=Carina.KA.CPU())
        @test sim.backend isa Carina.KA.CPU
        cpu_mag = maximum_magnitude(sim)
    end

    # ---- GPU run (if available) ----
    if has_gpu
        @testset "GPU runs on $backend" begin
            mktempdir() do dir
                cp_example(joinpath(example_dir, "sphere.g"),             joinpath(dir, "sphere.g"))
                cp_example(joinpath(example_dir, "sphere_explicit.yaml"), joinpath(dir, "sphere_explicit.yaml"))
                sim = Carina.run(joinpath(dir, "sphere_explicit.yaml"); backend=backend)

                # Confirm the simulation actually ran on the GPU backend
                @test !(sim.backend isa Carina.KA.CPU)
                @test sim.backend === backend

                # Results should match CPU
                gpu_mag = maximum_magnitude(sim)
                @test gpu_mag ≈ cpu_mag rtol=1e-6
            end
        end
    else
        @info "No GPU detected — skipping GPU verification (CPU-only run confirmed)"
    end
end

@testset "GPU Dirichlet BCs" begin
    # The verification above runs the explicit sphere, which is initial-condition
    # driven and carries no Dirichlet BCs, and the isbits check only inspects the
    # BC function's type. Neither compiles a BC kernel for a GPU backend, so a
    # defect that made *every* GPU run with boundary conditions impossible went
    # unnoticed here: Carina builds BC functions as FEC ScalarExpressionFunctions,
    # which are evaluated inside a KernelAbstractions kernel on every time step,
    # and that evaluator could not be compiled for the GPU.
    #
    # These cases pin the combination that was broken -- a real solve, driven by
    # Dirichlet BCs, on whatever device is available -- and use a spatially
    # varying function so the packed coordinate path is exercised too, not just
    # a constant or a function of time alone.

    backend = test_best_device()
    has_gpu = !(backend isa Carina.KA.CPU)
    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics", "quasistatic", "cube")

    # Explicit needs no linear solver, so the same input runs on both devices.
    explicit_yaml = """
type: single
input mesh file: cube.g
output mesh file: cube_gpu_bc.e
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
  final time: 2.0e-7
  time step: 1.0e-7
  gamma: 0.5
boundary conditions:
  dirichlet:
    - side set: ssz-
      component: z
      function: "0.0"
    - side set: ssz+
      component: z
      function: "1.0e-4 * t + 1.0e-6 * x"
"""

    # Quasi-static with an iterative solver: on GPU this is the matrix-free path,
    # so it covers BC evaluation inside a Newton/CG loop rather than a single
    # explicit update.
    implicit_yaml = """
type: single
input mesh file: cube.g
output mesh file: cube_gpu_bc_qs.e
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
  time step: 0.5
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
      function: "1.0e-4 * t + 1.0e-6 * x"
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
      type: jacobi
"""

    function run_with(yaml_text, name, dev)
        mktempdir() do dir
            cp_example(joinpath(example_dir, "cube.g"), joinpath(dir, "cube.g"))
            path = joinpath(dir, name)
            open(io -> write(io, yaml_text), path, "w")
            sim = Carina.run(path; backend=dev)
            return copy(Array(sim.params.field.data))
        end
    end

    for (yaml_text, name) in ((explicit_yaml, "gpu_bc_explicit.yaml"),
                              (implicit_yaml, "gpu_bc_implicit.yaml"))
        u_cpu = run_with(yaml_text, name, Carina.KA.CPU())
        @test all(isfinite, u_cpu)
        @test any(!iszero, u_cpu)   # the BC must actually move the mesh

        if has_gpu
            u_gpu = run_with(yaml_text, name, backend)
            @test u_gpu ≈ u_cpu rtol=1e-8
        end
    end

    has_gpu || @info "No GPU detected — GPU Dirichlet BC comparison skipped"
end
