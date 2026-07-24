# Per-block materials: each element block gets its own constitutive model.
#
# Carina previously applied one material to the entire mesh and errored on a
# `blocks:` mapping with more than one entry.  FEC always supported per-block
# physics -- `create_parameters` accepts a NamedTuple, `_setup_state_variables`
# sizes state per block, and `foreach_block` dispatches each block's own physics
# and properties -- so the restriction was Carina's alone.
#
# The hazard these tests exist for: FEC pairs the k-th physics entry with the
# k-th block POSITIONALLY and does not reorder (there is an explicit TODO to
# that effect in its Parameters.jl), while `blocks:` parses to an unordered
# `Dict`.  Getting the order wrong does not fail -- it runs to convergence with
# materials silently swapped between blocks.  So it is not enough to assert that
# a two-material run completes; the tests below assert that each block got the
# material it was actually assigned.

@testset "Per-block materials" begin

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics", "quasistatic", "two-block")

    load_dict() = Carina.YAML.load_file(joinpath(example_dir, "two-block.yaml");
                                        dicttype = Dict{String, Any})

    # Run and return mean u_z on the interface plane (z=1) and the top (z=2).
    function run_and_measure(dict)
        mktempdir() do dir
            cp_example(joinpath(example_dir, "two-block.g"), joinpath(dir, "two-block.g"))
            dict["input mesh file"]  = joinpath(dir, "two-block.g")
            dict["output mesh file"] = joinpath(dir, "two-block.e")
            sim = Carina.create_simulation(dict, dir)
            Carina.evolve!(sim)
            u = reshape(sim.params.field.data, 3, :)
            X = reshape(sim.params.coords.data, 3, :)
            z = X[3, :]
            iface = findall(x -> isapprox(x, 1.0; atol = 1e-9), z)
            topn  = findall(x -> isapprox(x, 2.0; atol = 1e-9), z)
            return (u_iface = sum(u[3, iface]) / length(iface),
                    u_top   = sum(u[3, topn])  / length(topn),
                    sim     = sim)
        end
    end

    # Two equal-length segments in series carry the same force, so their strains
    # go like 1/E.  With E_lower = 100 * E_upper the interface should sit at
    # ~1/101 of the applied end displacement.  Swapping the materials moves it to
    # ~100/101 -- a factor of 100 apart, so this cannot pass by accident.
    @testset "each block gets the material it was assigned" begin
        r = run_and_measure(load_dict())
        @test r.u_top ≈ 0.02 rtol = 1e-6
        @test r.u_iface / r.u_top < 0.05          # stiff block barely stretches
        @test r.u_iface / r.u_top ≈ 1 / 101 rtol = 0.15
    end

    @testset "swapping the assignment swaps the response" begin
        # Same materials, same mesh, only the block->material mapping flipped.
        # If materials were matched positionally rather than by name this would
        # be indistinguishable from the case above.
        dict = load_dict()
        dict["model"]["material"]["blocks"]["lower"] = "soft"
        dict["model"]["material"]["blocks"]["upper"] = "stiff"
        r = run_and_measure(dict)
        @test r.u_iface / r.u_top > 0.95          # now the LOWER block stretches
        @test r.u_iface / r.u_top ≈ 100 / 101 rtol = 0.15
    end

    @testset "physics and properties are keyed and ordered by mesh block" begin
        mktempdir() do dir
            cp_example(joinpath(example_dir, "two-block.g"), joinpath(dir, "two-block.g"))
            dict = load_dict()
            dict["input mesh file"]  = joinpath(dir, "two-block.g")
            dict["output mesh file"] = joinpath(dir, "two-block.e")
            sim = Carina.create_simulation(dict, dir)

            fspace      = Carina.FEC.function_space(sim.integrator.asm.dof)
            block_order = String.(collect(keys(fspace.ref_fes)))

            # The invariant FEC relies on: entry k belongs to block k.
            @test String.(collect(keys(sim.params.physics)))    == block_order
            @test String.(collect(keys(sim.params.properties))) == block_order
            @test length(sim.params.physics) == 2

            # props[1] is density; props[2:3] are the Lamé constants, so the
            # stiff block must have the larger shear modulus.
            props = values(sim.params.properties)
            lower_idx = findfirst(==("lower"), block_order)
            upper_idx = findfirst(==("upper"), block_order)
            @test props[lower_idx][3] > 10 * props[upper_idx][3]
        end
    end

    @testset "both material spellings work" begin
        # Legacy: the label IS the model name, no `model:` key.
        dict = load_dict()
        mat  = dict["model"]["material"]
        mat["neohookean"] = Dict{String, Any}("elastic modulus" => 1.0e9,
                                              "Poisson's ratio" => 0.25,
                                              "density"         => 1000.0)
        delete!(mat, "stiff"); delete!(mat, "soft")
        mat["blocks"]["lower"] = "neohookean"
        mat["blocks"]["upper"] = "neohookean"
        r = run_and_measure(dict)
        # Uniform material: the interface sits at half the end displacement.
        @test r.u_iface / r.u_top ≈ 0.5 rtol = 0.1
    end

    @testset "every mesh block must be assigned" begin
        dict = load_dict()
        delete!(dict["model"]["material"]["blocks"], "upper")
        err = try
            run_and_measure(dict); ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("does not assign a material", err)
        @test occursin("upper", err)
    end

    @testset "assigning a material to a non-existent block is an error" begin
        dict = load_dict()
        blocks = dict["model"]["material"]["blocks"]
        blocks["uppr"] = pop!(blocks, "upper")       # typo
        err = try
            run_and_measure(dict); ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("not an element block in the mesh", err)
        @test occursin("uppr", err)
        @test occursin("did you mean \"upper\"", err)   # Levenshtein suggestion
    end

    @testset "a dynamic run checks density on every block" begin
        dict = load_dict()
        dict["time integrator"] = Dict{String, Any}(
            "type" => "newmark", "initial time" => 0.0,
            "final time" => 0.02, "time step" => 0.01,
        )
        # Only the second block loses its density -- a check that looked at just
        # the first material would miss this.
        delete!(dict["model"]["material"]["soft"], "density")
        err = try
            run_and_measure(dict); ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("density 0.0", err)
        @test occursin("upper", err)
    end

    @testset "mixed state variables: element output yes, nodal recovery no" begin
        # An elastic block (0 state variables) next to a J2 block (10).  Nodes on
        # the interface belong to both, so projecting `eqps` there is meaningless
        # -- that must fail loudly rather than average over one side.
        function j2_upper(recovery)
            dict = load_dict()
            soft = dict["model"]["material"]["soft"]
            soft["model"]             = "j2 plasticity"
            soft["yield stress"]      = 1.0e6
            soft["hardening modulus"] = 1.0e8
            dict["output"] = Dict{String, Any}(
                "internal variables" => true, "stress" => false,
                "deformation gradient" => false, "recovery" => recovery,
            )
            return dict
        end

        err = try
            run_and_measure(j2_upper("lumped")); ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("same state variables", err)
        @test occursin("lower", err) && occursin("upper", err)

        # Per-block element output has no such ambiguity and must still work.
        r = run_and_measure(j2_upper("none"))
        @test r.u_top ≈ 0.02 rtol = 1e-6
    end

end
