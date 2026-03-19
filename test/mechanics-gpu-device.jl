@testset "GPU Device Verification" begin
    # Run the explicit sphere for 1 step on the best available device.
    # If a GPU is present, verify that sim.device is :rocm or :cuda
    # and that the result matches CPU to machine precision.
    #
    # Uses the explicit solver (no linear solver needed) so the same
    # YAML works on both CPU and GPU.

    device = Carina.best_device()
    has_gpu = device != "cpu"

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics",
                           "explicit-dynamic", "sphere")

    # ---- CPU baseline ----
    cpu_mag = 0.0
    mktempdir() do dir
        cp_example(joinpath(example_dir, "sphere.g"),             joinpath(dir, "sphere.g"))
        cp_example(joinpath(example_dir, "sphere_explicit.yaml"), joinpath(dir, "sphere_explicit.yaml"))
        sim = Carina.run(joinpath(dir, "sphere_explicit.yaml"); device="cpu")
        @test sim.device == :cpu
        cpu_mag = maximum_magnitude(sim)
    end

    # ---- GPU run (if available) ----
    if has_gpu
        @testset "GPU runs on $device" begin
            mktempdir() do dir
                cp_example(joinpath(example_dir, "sphere.g"),             joinpath(dir, "sphere.g"))
                cp_example(joinpath(example_dir, "sphere_explicit.yaml"), joinpath(dir, "sphere_explicit.yaml"))
                sim = Carina.run(joinpath(dir, "sphere_explicit.yaml"); device=device)

                # Confirm the simulation actually ran on GPU
                @test sim.device in (:rocm, :cuda)
                @test sim.device != :cpu

                # Results should match CPU
                gpu_mag = maximum_magnitude(sim)
                @test gpu_mag ≈ cpu_mag rtol=1e-6
            end
        end
    else
        @info "No GPU detected — skipping GPU verification (CPU-only run confirmed)"
    end
end
