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
        cp_example(joinpath(example_dir, "sphere_explicit.toml"), joinpath(dir, "sphere_explicit.toml"))
        sim = Carina.run(joinpath(dir, "sphere_explicit.toml"); backend=Carina.KA.CPU())
        @test sim.backend isa Carina.KA.CPU
        cpu_mag = maximum_magnitude(sim)
    end

    # ---- GPU run (if available) ----
    if has_gpu
        @testset "GPU runs on $backend" begin
            mktempdir() do dir
                cp_example(joinpath(example_dir, "sphere.g"),             joinpath(dir, "sphere.g"))
                cp_example(joinpath(example_dir, "sphere_explicit.toml"), joinpath(dir, "sphere_explicit.toml"))
                sim = Carina.run(joinpath(dir, "sphere_explicit.toml"); backend=backend)

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
