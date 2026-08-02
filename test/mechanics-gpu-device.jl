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

    # The device copy of the assembler must be stripped of the assembled-matrix
    # machinery (sparse pattern + matrix value buffers): device solves are
    # matrix-free by construction, and that storage dominated VRAM (~5.6 GB of
    # a 5.8 GB footprint at 530k DOFs) while never being read.  The CPU
    # assembler must keep its pattern for host-side work (initial acceleration,
    # GPU Cholesky factorization, AMG).
    if has_gpu
        @testset "device assembler is stripped" begin
            mktempdir() do dir
                cp_example(joinpath(example_dir, "cube.g"), joinpath(dir, "cube.g"))
                path = joinpath(dir, "gpu_bc_implicit_strip.yaml")
                open(io -> write(io, implicit_yaml), path, "w")
                sim = Carina.run(path; backend=backend)
                @test Carina.FEC._is_matrix_free(sim.integrator.asm)
                @test !Carina.FEC._is_matrix_free(sim.asm_cpu)
            end
        end
    end

    has_gpu || @info "No GPU detected — GPU Dirichlet BC comparison skipped"
end

@testset "GPU AMG preconditioner" begin
    # The device-resident AMG V-cycle (src/gpu_amg.jl) against ground truth:
    # the same quasistatic problem solved by CPU direct.  This is the
    # checked-in artifact for the solution-agreement claim in
    # benchmark_report.md §1 — CG+AMG on the device must reproduce the
    # direct answer, not merely converge.
    backend = test_best_device()
    if backend isa Carina.KA.CPU
        @info "No GPU detected — GPU AMG verification skipped"
    else
        # A 16^3 generated cube (14,739 DOF): large enough that the SA
        # hierarchy has assembled intermediate levels, so the device
        # descend/ascend CSR loops are exercised — the 8-element cube.g
        # coarsens straight to the dense coarse solve and tests nothing.
        include(joinpath(@__DIR__, "..", "benchmark", "meshgen.jl"))
        amg_yaml(solver) = """
type: single
input mesh file: cube16.g
output mesh file: cube_gpu_amg.e
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
    - node set: nsx-
      component: x
      function: "0.0"
    - node set: nsy-
      component: y
      function: "0.0"
    - node set: nsz-
      component: z
      function: "0.0"
    - node set: nsz+
      component: z
      function: "1.0e-3 * t"
solver:
  type: newton
  termination:
    fail when any:
      - maximum iterations: 20
    converge when any:
      - absolute residual: 1.0e-8
      - relative residual: 1.0e-10
$solver
"""
        direct = "  linear solver:\n    type: direct\n"
        amg    = "  linear solver:\n    type: iterative\n    tolerance: 1.0e-10\n" *
                 "    maximum iterations: 2000\n    preconditioner:\n      type: amg\n"

        function run_with(solver, dev)
            mktempdir() do dir
                generate(16, joinpath(dir, "cube16.g"))
                path = joinpath(dir, "run.yaml")
                open(io -> write(io, amg_yaml(solver)), path, "w")
                sim = Carina.run(path; backend=dev)
                @test !sim.integrator.failed[]
                return sim, copy(Array(sim.params.field.data))
            end
        end

        _, u_ref = run_with(direct, Carina.KA.CPU())
        sim_amg, u_amg = run_with(amg, backend)
        @test u_amg ≈ u_ref rtol = 1e-7

        # The hierarchy really was built, lives on the device, and has at
        # least one assembled intermediate level (multilevel V-cycle, not a
        # degenerate fine→coarse-solve shortcut).
        ls = sim_amg.integrator.nonlinear_solver.linear_solver
        @test ls.precond isa Carina.GPUAMGPreconditioner
        @test ls.precond.hierarchy !== nothing
        @test length(ls.precond.hierarchy.levels) >= 1
        @test ls.precond.nbuilds >= 1

        # The V-cycle apply path must not allocate device memory: repeated
        # applications leave live VRAM exactly unchanged (ROCm only — the
        # portable KA layer has no allocation counter).
        # `isdefined(Main, :AMDGPU)` is not enough.  AMDGPU is a test-only
        # extra, so `Pkg.test()` makes it importable on every machine —
        # including NVIDIA-only ones, where the HIP runtime is absent and the
        # first call into it dies with `undefined symbol: hipDeviceGet`.  That
        # throw lands outside any `@test`, so it aborts the whole file instead
        # of failing softly.  `_TEST_AMDGPU` (helpers.jl) adds the
        # `AMDGPU.functional()` check, and is what `test_best_device()` used to
        # pick `backend` above, so this now tracks the backend actually in use.
        if _TEST_AMDGPU
            h = ls.precond.hierarchy
            ig = sim_amg.integrator
            n = length(ls.precond.inv_diag)
            z = Carina.KA.allocate(backend, Float64, n); fill!(z, 0.0)
            r = Carina.KA.allocate(backend, Float64, n); fill!(r, 1.0)
            # The REAL matrix-free fine action, as production V-cycles use —
            # a placeholder here would exempt 5 of the ~7 kernel launches per
            # application from the allocation check.
            mv!(y, x) = Carina._stiffness_matvec_qs!(y, x, ig.asm, ig.U,
                                                     sim_amg.params)
            Carina._amg_vcycle!(z, r, h, mv!, backend)   # warm-up/compile
            Main.AMDGPU.synchronize()
            live0 = Main.AMDGPU.memory_stats().live
            for _ in 1:20
                Carina._amg_vcycle!(z, r, h, mv!, backend)
            end
            Main.AMDGPU.synchronize()
            @test Main.AMDGPU.memory_stats().live == live0
        end
    end
end
