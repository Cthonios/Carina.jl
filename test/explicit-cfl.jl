# Explicit dynamics with a CFL-driven stable time step.
#
# `_compute_stable_dt` runs only when `cfl > 0` is set in the input.  Every
# explicit test in the suite used a fixed `time step`, so the whole code path
# was unexercised -- and when `SolidMechanics` stopped carrying a `density`
# field, `_compute_stable_dt` kept reading `block_physics.density` and threw
# `FieldError: type SolidMechanics has no field density`.  CI stayed green while
# the shipped examples/mechanics/explicit-dynamic/cantilever example crashed on
# startup.
#
# Density now comes from props[1], per the ConstitutiveModels contract.

@testset "Explicit CFL stable time step" begin

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics", "explicit-dynamic", "cube")

    # Free-free cube, uniform initial v_z: rigid translation, no deformation, so
    # u_z(t_f) = v_z * t_f exactly regardless of how the step size is chosen.
    # E = 1e3, ρ = 1e3  →  c ≈ 1 m/s, h = 0.5 m  →  Δt_stable ≈ CFL * 0.5.
    function run_with(extra_integrator_keys)
        mktempdir() do dir
            cp_example(joinpath(example_dir, "cube.g"), joinpath(dir, "cube.g"))
            dict = Carina.YAML.load_file(joinpath(example_dir, "cube.yaml");
                                         dicttype = Dict{String, Any})
            merge!(dict["time integrator"], extra_integrator_keys)
            dict["input mesh file"]  = joinpath(dir, "cube.g")
            dict["output mesh file"] = joinpath(dir, "cube.e")
            sim = Carina.create_simulation(dict, dir)
            Carina.evolve!(sim)
            return sim
        end
    end

    @testset "cfl-driven run completes and translates rigidly" begin
        sim = run_with(Dict{String, Any}("cfl" => 0.2, "final time" => 0.2))
        avg = average_components(sim)
        @test avg[3] ≈ 0.2 rtol = 1e-6      # u_z = v_z * t_f
        @test isapprox(avg[1], 0.0; atol = 1e-10)
        @test isapprox(avg[2], 0.0; atol = 1e-10)
    end

    @testset "stable dt is computed from props[1], not a physics field" begin
        # Direct unit check on the estimator: it must run at all (the regression
        # was an exception, not a wrong number) and land near CFL * h / c.
        mktempdir() do dir
            cp_example(joinpath(example_dir, "cube.g"), joinpath(dir, "cube.g"))
            dict = Carina.YAML.load_file(joinpath(example_dir, "cube.yaml");
                                         dicttype = Dict{String, Any})
            dict["input mesh file"]  = joinpath(dir, "cube.g")
            dict["output mesh file"] = joinpath(dir, "cube.e")
            sim = Carina.create_simulation(dict, dir)

            CFL = 0.2
            dt = Carina._compute_stable_dt(sim.integrator.asm, sim.params, CFL)
            @test isfinite(dt)
            @test dt > 0.0

            # ρ = 1000, E = 1000, ν = 0.25 → M = E(1-ν)/((1+ν)(1-2ν)) = 1.2e3,
            # c_p = sqrt(M/ρ) ≈ 1.0954 m/s, h = 0.5 m  →  dt ≈ 0.2*0.5/1.0954.
            M   = 1000.0 * (1 - 0.25) / ((1 + 0.25) * (1 - 2 * 0.25))
            c_p = sqrt(M / 1000.0)
            @test dt ≈ CFL * 0.5 / c_p rtol = 1e-6
        end
    end

    @testset "the shipped cantilever example builds" begin
        # This is the example that regressed: it is the only bundled input that
        # sets `cfl`.  The failure was a `FieldError` raised while *building* the
        # integrator, so constructing the simulation is the whole test -- no need
        # to pay for time stepping.
        cdir = joinpath(@__DIR__, "..", "examples", "mechanics", "explicit-dynamic", "cantilever")
        mktempdir() do dir
            cp_example(joinpath(cdir, "cantilever.g"), joinpath(dir, "cantilever.g"))
            dict = Carina.YAML.load_file(joinpath(cdir, "cantilever.yaml");
                                         dicttype = Dict{String, Any})
            @test dict["time integrator"]["cfl"] > 0.0   # guard the premise
            dict["input mesh file"]  = joinpath(dir, "cantilever.g")
            dict["output mesh file"] = joinpath(dir, "cantilever.e")
            sim = Carina.create_simulation(dict, dir)
            # The integrator holds the CFL-derived step, not the input's 1e-6.
            @test sim.integrator.time_step < 1.0e-6
        end
    end

end
