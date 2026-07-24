# Regression test for `output.internal variables: true`.
#
# `_write_element_fields!` looked up per-block physics by NAME:
#
#     block_physics = p_cpu.physics[block_name]
#
# `block_name` comes from `fspace.ref_fes`, which keeps the Exodus block names.
# But `FEC.create_parameters` re-keys physics and properties POSITIONALLY as
# `region_1..N` whenever it is handed a bare `AbstractPhysics` -- which is
# always, because Carina applies a single material to the whole mesh.  The two
# NamedTuples therefore share no key names unless the mesh block is literally
# called "region_1", so the lookup threw
#
#     FieldError: type NamedTuple has no field `tension`, available fields: `region_1`
#
# for every other mesh.  Nothing caught it: the only bundled inputs that set
# `internal variables: true` are the two tension-specimen-j2 examples, and
# neither is in the test suite.
#
# Note the throw happens before anything that depends on the number of state
# variables, so a model with no state variables reproduces it just as well as a
# plastic one -- which keeps this test cheap.

@testset "Internal variables output" begin

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics", "quasistatic", "cube")

    # The cube mesh's block is named "cube", not "region_1" -- that mismatch is
    # the whole point.
    function run_with_internal_variables(flag)
        mktempdir() do dir
            cp_example(joinpath(example_dir, "cube.g"), joinpath(dir, "cube.g"))
            dict = Carina.YAML.load_file(joinpath(example_dir, "cube.yaml");
                                         dicttype = Dict{String, Any})
            out = get!(dict, "output", Dict{String, Any}())
            out["internal variables"] = flag
            dict["input mesh file"]  = joinpath(dir, "cube.g")
            dict["output mesh file"] = joinpath(dir, "cube.e")
            sim = Carina.create_simulation(dict, dir)
            Carina.evolve!(sim)
            return sim
        end
    end

    @testset "writing state output does not throw on a normally-named block" begin
        sim = run_with_internal_variables(true)
        @test sim isa Carina.SingleDomainSimulation
    end

    @testset "still fine with the flag off" begin
        sim = run_with_internal_variables(false)
        @test sim isa Carina.SingleDomainSimulation
    end

    @testset "physics is keyed positionally, not by mesh block name" begin
        # Carina now hands FEC a physics NamedTuple keyed by the real mesh block
        # names, so the mismatch that caused this bug no longer exists: the keys
        # agree.  Positional access remains the invariant FEC actually
        # guarantees (`foreach_block` pairs entry k with block k), so the output
        # path still indexes by position -- but assert the names line up too,
        # because a future change that reintroduced a separate key space would
        # bring the original failure mode back with it.
        mktempdir() do dir
            cp_example(joinpath(example_dir, "cube.g"), joinpath(dir, "cube.g"))
            dict = Carina.YAML.load_file(joinpath(example_dir, "cube.yaml");
                                         dicttype = Dict{String, Any})
            dict["input mesh file"]  = joinpath(dir, "cube.g")
            dict["output mesh file"] = joinpath(dir, "cube.e")
            sim = Carina.create_simulation(dict, dir)

            fspace      = Carina.FEC.function_space(sim.integrator.asm.dof)
            block_names = collect(keys(fspace.ref_fes))
            phys_keys   = collect(keys(sim.params.physics))

            @test length(block_names) == length(phys_keys)
            @test Symbol("cube") in block_names
            # Same names, same order -- physics entry k really is block k.
            @test phys_keys == block_names
            @test collect(keys(sim.params.properties)) == block_names
            @test values(sim.params.physics)[1] isa Carina.SolidMechanics
        end
    end

end
