# Coverage for EVERY material model Carina advertises.
#
# The suite previously only ever constructed `neohookean` and `linear elastic`,
# so three separate breakages reached a green CI at once:
#
#   1. A stray trailing comma in `_model_ctor` folded the Saint-Venant-Kirchhoff
#      arm into the preceding `return` as a tuple element.  Every SVK spelling
#      became unreachable (falling through to `return nothing`), and
#      `linear elasto plasticity` returned a `Tuple{LinearElastoplastic, Bool}`
#      instead of a model -- which then failed far away with a `MethodError` on
#      `SolidMechanics(::Tuple)`.
#   2. `_MODEL_NAMES` was maintained by hand alongside the dispatch chain, so
#      the "Unknown material model" error went on listing `svk` as supported
#      while rejecting it.
#   3. `CM.FiniteDefJ2Plasticity` was dropped from ConstitutiveModels, but
#      `_model_ctor` still referenced it -- `UndefVarError` at first use.
#
# The invariant that catches all three: every name Carina advertises must
# actually build a constitutive model, and the advertised list must be exactly
# the accepted list.

@testset "Material models" begin

    # Enough properties to satisfy any model in the table.
    base_props() = Dict{String, Any}(
        "elastic modulus"   => 1.0e9,
        "Poisson's ratio"   => 0.25,
        "density"           => 1000.0,
        "yield stress"      => 1.0e6,
        "hardening modulus" => 1.0e6,
        "m"                 => 2.0,
        "n"                 => 2.0,
    )

    @testset "every advertised name builds a model" begin
        # Directly guards defect 1 and 3: `nothing` means an unreachable arm,
        # a Tuple means an arm got folded into its neighbour's return.
        for name in Carina._MODEL_NAMES
            cm = Carina._model_ctor(name)
            @test cm !== nothing
            @test !(cm isa Tuple)
            @test cm isa Carina.CM.AbstractConstitutiveModel
        end
    end

    @testset "advertised list matches accepted list" begin
        # Guards defect 2: the error message must not name a model that
        # `_model_ctor` rejects.  `_MODEL_NAMES` is derived from `_MODEL_TABLE`,
        # so this holds by construction -- assert it so a future refactor that
        # reintroduces a hand-maintained list fails here.
        derived = collect(Iterators.flatten(a for (a, _) in Carina._MODEL_TABLE))
        @test sort(collect(Carina._MODEL_NAMES)) == sort(derived)
        @test length(derived) == length(unique(derived))
    end

    @testset "aliases of a model agree" begin
        # Every spelling of one model must produce the same concrete type.
        for (aliases, _) in Carina._MODEL_TABLE
            types = unique(typeof(Carina._model_ctor(a)) for a in aliases)
            @test length(types) == 1
        end
    end

    @testset "parse_material returns a model, never a tuple" begin
        for name in Carina._MODEL_NAMES
            cm, ρ, props_inputs = Carina.parse_material(name, base_props())
            @test cm isa Carina.CM.AbstractConstitutiveModel
            @test ρ == 1000.0
            # Density must be handed to CM as a property: `initialize_props`
            # reads inputs["density"] and emits it as props[1].
            @test props_inputs["density"] == 1000.0
        end
    end

    @testset "density is the first property of every model" begin
        # The mass-matrix kernels and the explicit stable-dt estimate both read
        # props[1] as density.  If CM ever reorders, this fails loudly here
        # rather than silently producing a wrong mass matrix.
        for name in Carina._MODEL_NAMES
            cm, ρ, props_inputs = Carina.parse_material(name, base_props())
            props = Carina.create_solid_mechanics_properties(cm, props_inputs)
            @test length(props) == Carina.CM.num_properties(cm)
            @test props[1] == ρ
        end
    end

    @testset "SolidMechanics builds for every model" begin
        for name in Carina._MODEL_NAMES
            cm, _, _ = Carina.parse_material(name, base_props())
            physics = Carina.SolidMechanics(cm)
            @test physics.constitutive_model === cm
            # Density is deliberately NOT stored on the physics object; it lives
            # in the property vector so the two cannot disagree.
            @test !hasfield(typeof(physics), :density)
        end
    end

    @testset "unknown models fail loudly and list the alternatives" begin
        err = try
            Carina.parse_material("neohookian", base_props())
        catch e
            sprint(showerror, e)
        end
        @test occursin("Unknown material model", err)
        # The suggestion list must be the real one -- this is the assertion that
        # would have caught the error advertising "svk" while rejecting it.
        for name in Carina._MODEL_NAMES
            @test occursin(name, err)
        end
    end

    @testset "documentation lists exactly the supported models" begin
        # Keeps docs/src/reference/materials.md from advertising a model that
        # was removed, or omitting one that was added.
        doc = read(joinpath(@__DIR__, "..", "docs", "src", "reference", "materials.md"), String)
        for name in Carina._MODEL_NAMES
            @test occursin("`$name`", doc)
        end
        # Every model must have a row in the models table, not merely be
        # mentioned somewhere in the prose.
        rows = filter(l -> startswith(strip(l), "| `"), split(doc, '\n'))
        for (aliases, _) in Carina._MODEL_TABLE
            @test any(r -> startswith(strip(r), "| `$(first(aliases))`"), rows)
        end
    end

end
