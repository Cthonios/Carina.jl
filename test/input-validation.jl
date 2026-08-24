# Regression tests for input validation.
#
# Every case here is a spelling or value mistake that Carina used to accept
# silently -- running to completion with the wrong material, no boundary
# conditions, no initial condition, or a different solver than the one asked
# for. The point of each test is that the mistake now *stops the run*, so these
# guard against a fallthrough being reintroduced.
#
# The mirror image matters just as much: every valid spelling and alias is
# asserted to still parse, because the cheapest way to break a loud check is to
# make it too strict.

@testset "Input validation" begin

    good_material() = Dict{String,Any}(
        "model" => Dict{String,Any}(
            "type" => "solid mechanics",
            "material" => Dict{String,Any}(
                "blocks" => Dict{String,Any}("cube" => "neohookean"),
                "neohookean" => Dict{String,Any}(
                    "elastic modulus" => 1.0e9,
                    "Poisson's ratio" => 0.25,
                    "density" => 1000.0,
                ),
            ),
        ),
    )

    # ----- model ------------------------------------------------------------
    @testset "model" begin
        # `_parse_material_section` takes the mesh's block order and returns one
        # BlockMaterial per block, in that order.  See test/per-block-materials.jl
        # for the assignment semantics; this set covers the parse-level checks.
        parse1(d) = Carina._parse_material_section(d, ["cube"])

        @test parse1(good_material())[1].block == "cube"

        # `model.type` names a physics that does not exist.  Previously the key
        # was read by nothing at all, so `thermal` ran as solid mechanics.
        d = good_material()
        d["model"]["type"] = "thermal"
        @test_throws ErrorException parse1(d)

        # Omitting `type` remains legal.
        d = good_material()
        delete!(d["model"], "type")
        @test parse1(d)[1].block == "cube"

        # Assigning a material to a block the mesh does not have.  This used to
        # be silently dropped: `first(blocks_dict)` picked a survivor in hash
        # order and applied it to the whole mesh.
        d = good_material()
        d["model"]["material"]["blocks"]["shell"] = "linear elastic"
        @test_throws ErrorException parse1(d)

        # ...and the reverse: a mesh block with no material assigned.  There is
        # no default, because a block inheriting another block's material would
        # converge to a plausible wrong answer.
        d = good_material()
        @test_throws ErrorException Carina._parse_material_section(d, ["cube", "shell"])

        d = good_material()
        empty!(d["model"]["material"]["blocks"])
        @test_throws ErrorException parse1(d)

        # `blocks` names a material with no matching property dict.
        d = good_material()
        d["model"]["material"]["blocks"]["cube"] = "hencky"
        @test_throws ErrorException parse1(d)

        # The property dict is resolved case-insensitively, matching the
        # case-insensitive validation of the same keys.
        d = good_material()
        d["model"]["material"]["blocks"]["cube"] = "NeoHookean"
        d["model"]["material"]["NeoHookean"] = pop!(d["model"]["material"], "neohookean")
        @test parse1(d)[1].density == 1000.0

        # An arbitrary label plus an explicit `model:` key -- the spelling that
        # lets two blocks share a model with different properties.
        d = good_material()
        d["model"]["material"]["blocks"]["cube"] = "my_steel"
        d["model"]["material"]["my_steel"] = pop!(d["model"]["material"], "neohookean")
        d["model"]["material"]["my_steel"]["model"] = "neohookean"
        m = parse1(d)[1]
        @test m.model_name == "neohookean"
        @test m.density == 1000.0
        # `model` is a selector, not a material property, so it must not leak
        # into the property dict handed to ConstitutiveModels.
        @test !haskey(m.props_inputs, "model")
    end

    # ----- quadrature -------------------------------------------------------
    @testset "quadrature" begin
        q(type, order) = Carina._parse_quadrature(Dict{String,Any}(
            "quadrature" => Dict{String,Any}("type" => type, "order" => order)))

        @test q("gauss legendre", 2) == (Carina.RFE.GaussLegendre, 2)
        @test q("GLL", 3)[2] == 3
        @test_throws ErrorException q("simpson", 2)

        # Omitting the section keeps the default rule.
        @test Carina._parse_quadrature(Dict{String,Any}()) == (Carina.RFE.GaussLegendre, 2)
    end

    # ----- initial conditions -----------------------------------------------
    @testset "initial conditions" begin
        ic(pairs...) = Dict{String,Any}("initial conditions" => Dict{String,Any}(pairs...))
        entry() = Dict{String,Any}("node set" => "nsall",
                                    "component" => "z", "function" => "0.0")

        @test length(Carina._parse_displacement_ics(ic("displacement" => [entry()]))) == 1
        @test length(Carina._parse_velocity_ics(ic("velocity" => [entry()]))) == 1
        @test isempty(Carina._parse_displacement_ics(Dict{String,Any}()))

        # A misspelled entry key used to reach `_apply_initial_*_ics!` and
        # surface as a bare KeyError naming neither section nor entry.
        bad = entry(); bad["nodeset"] = pop!(bad, "node set")
        @test_throws ErrorException Carina._parse_displacement_ics(ic("displacement" => [bad]))

        bad = entry(); delete!(bad, "function")
        @test_throws ErrorException Carina._parse_velocity_ics(ic("velocity" => [bad]))

        @test_throws ErrorException Carina._parse_displacement_ics(
            ic("displacement" => Dict{String,Any}()))

        # Section-level keys are validated from `create_simulation`, since each
        # of the three parsers reads only its own sub-key.
        @test Carina._validate_ic_section(ic("velocity" => [entry()])) === nothing
        @test Carina._validate_ic_section(Dict{String,Any}()) === nothing
    end

    # ----- nonlinear solver -------------------------------------------------
    @testset "solver type" begin
        ns(type) = Carina._parse_nonlinear_solver(
            Dict{String,Any}("type" => type), Carina.NoLinearSolver())

        @test ns("newton") isa Carina.NewtonSolver
        @test ns("hessian minimizer") isa Carina.NewtonSolver
        @test ns("NEWTON-RAPHSON") isa Carina.NewtonSolver
        @test ns("nlcg") isa Carina.NLCGSolver
        @test ns("sd") isa Carina.SteepestDescentSolver

        # Newton is still the default when `type` is absent.
        @test Carina._parse_nonlinear_solver(
            Dict{String,Any}(), Carina.NoLinearSolver()) isa Carina.NewtonSolver

        # `lbfgs` is a *linear* solver type.  It used to fall through to Newton,
        # so the run silently used an algorithm the user had not asked for.
        @test_throws ErrorException ns("lbfgs")
        @test_throws ErrorException ns("newtno")

        # The gate in `_read_solver_dicts` and the dispatch in
        # `_parse_nonlinear_solver` must accept exactly the same set, or a value
        # passes one and is rejected by the other.
        for t in Carina._SOLVER_TYPES
            @test Carina._read_solver_dicts(Dict{String,Any}(
                "solver" => Dict{String,Any}(
                    "type" => t,
                    "linear solver" => Dict{String,Any}("type" => "direct")))) isa Tuple
            @test Carina._parse_nonlinear_solver(
                Dict{String,Any}("type" => t), Carina.NoLinearSolver()) isa
                Carina.AbstractNonlinearSolver
        end
    end

    # ----- preconditioner and recovery values -------------------------------
    @testset "value fallthroughs" begin
        # Both of these used to degrade silently: an unknown preconditioner
        # became NoPreconditioner (a slow but converging solve), and an unknown
        # recovery became :none (an output file missing nodal fields).
        ls(precond) = Carina._parse_linear_solver(
            Dict{String,Any}("type" => "iterative",
                             "preconditioner" => Dict{String,Any}("type" => precond)),
            zeros(4), Carina.KA.CPU(), () -> Carina.NoPreconditioner())

        @test ls("none").precond isa Carina.NoPreconditioner
        @test ls("chebyshev").precond isa Carina.ChebyshevPreconditioner
        @test_throws ErrorException ls("jacoby")

        rec(v) = Carina._parse_output_spec(Dict{String,Any}(
            "output" => Dict{String,Any}("recovery" => v))).recovery
        @test rec("lumped") == :lumped
        @test rec("L2") == :consistent
        @test rec("none") == :none
        @test_throws ErrorException rec("lump")
    end

    # ----- legacy combo -----------------------------------------------------
    @testset "termination combo" begin
        legacy(combo) = Carina._parse_termination(Dict{String,Any}(
            "termination" => Any[Dict{String,Any}(
                "type" => "combo", "combo" => combo,
                "tests" => Any[Dict{String,Any}("type" => "absolute residual",
                                                 "tolerance" => 1.0e-8)])]))

        @test legacy("and").tests[1] isa Carina.ComboAndTest
        @test legacy("or").tests[1] isa Carina.ComboOrTest
        # Anything else used to be read as "or", inverting the group's meaning.
        @test_throws ErrorException legacy("nad")
    end

    # ----- BC and body-force entries ----------------------------------------
    @testset "entry required keys" begin
        bc(kind, entry) = Dict{String,Any}(
            "boundary conditions" => Dict{String,Any}(kind => Any[entry]))

        full = Dict{String,Any}("side set" => "ssz-", "component" => "z",
                                 "function" => "0.0")
        @test length(Carina._parse_dirichlet_bcs(bc("dirichlet", copy(full)))) == 1

        # Each of these used to surface as a bare KeyError naming neither the
        # section nor the entry.
        for missing_key in ("component", "function")
            e = copy(full); delete!(e, missing_key)
            @test_throws ErrorException Carina._parse_dirichlet_bcs(bc("dirichlet", e))
            @test_throws ErrorException Carina._parse_neumann_bcs(bc("neumann", e))
        end

        bf(entry) = Dict{String,Any}("body forces" => Any[entry])
        @test length(Carina._parse_body_forces(bf(Dict{String,Any}(
            "component" => "z", "function" => "-9.81")))) == 1
        @test_throws ErrorException Carina._parse_body_forces(
            bf(Dict{String,Any}("component" => "z")))
    end

    # ----- mesh entity names ------------------------------------------------
    @testset "mesh names" begin
        mesh_file = joinpath(@__DIR__, "..", "examples", "mechanics",
                              "quasistatic", "cube", "cube.g")
        mesh = Carina.FEC.UnstructuredMesh(mesh_file)

        base() = Dict{String,Any}(
            "boundary conditions" => Dict{String,Any}(
                "dirichlet" => Any[Dict{String,Any}(
                    "side set" => "ssz-", "component" => "z", "function" => "0.0")]))

        @test Carina._validate_mesh_names(base(), mesh, "cube") === nothing

        # The material block name is used only for the startup log line, so a
        # typo here used to produce a correct-looking run with a wrong label.
        @test_throws ErrorException Carina._validate_mesh_names(base(), mesh, "cubeTypo")

        # A bad side set previously reached FEC as a bare KeyError.
        d = base()
        d["boundary conditions"]["dirichlet"][1]["side set"] = "sszMinus"
        @test_throws ErrorException Carina._validate_mesh_names(d, mesh, "cube")

        # Casing of `dirichlet` must not smuggle an entry past the check.
        d = base()
        d["boundary conditions"]["Dirichlet"] = pop!(d["boundary conditions"], "dirichlet")
        d["boundary conditions"]["Dirichlet"][1]["side set"] = "nope"
        @test_throws ErrorException Carina._validate_mesh_names(d, mesh, "cube")

        d = base()
        d["initial conditions"] = Dict{String,Any}("velocity" => Any[Dict{String,Any}(
            "node set" => "nsallx", "component" => "z", "function" => "1.0")])
        @test_throws ErrorException Carina._validate_mesh_names(d, mesh, "cube")

        d = base()
        d["body forces"] = Any[Dict{String,Any}(
            "block" => "nosuchblock", "component" => "z", "function" => "-9.81")]
        @test_throws ErrorException Carina._validate_mesh_names(d, mesh, "cube")

        # `block: all` is the documented default and must stay legal.
        d = base()
        d["body forces"] = Any[Dict{String,Any}(
            "component" => "z", "function" => "-9.81")]
        @test Carina._validate_mesh_names(d, mesh, "cube") === nothing
    end

    # ----- scalar coercion ---------------------------------------------------
    @testset "_f64 accepts any Real" begin
        # YAML.jl only ever hands back Int64/Float64, but programmatic callers
        # can pass any Real; the typed fallback must coerce rather than error.
        @test Carina._f64(1.5)          === 1.5
        @test Carina._f64(Int64(3))     === 3.0
        @test Carina._f64(Float32(2.5)) === 2.5
        @test Carina._f64(3 // 4)       === 0.75
    end

    # ----- component / direction maps ----------------------------------------
    @testset "component and direction reject unknowns" begin
        @test Carina._component_to_string(" X ") == "displ_x"
        @test_throws ErrorException Carina._component_to_string("w")
        @test Carina._direction_to_idx("z") == 3
        @test_throws ErrorException Carina._direction_to_idx("w")
    end

    # ----- linear solver values ----------------------------------------------
    @testset "linear solver type and preconditioner arms" begin
        ls(d) = Carina._parse_linear_solver(
            d, zeros(4), Carina.KA.CPU(), () -> Carina.NoPreconditioner())
        iterative(precond) = ls(Dict{String,Any}(
            "type" => "iterative",
            "preconditioner" => Dict{String,Any}("type" => precond)))

        @test iterative("incomplete cholesky").precond isa Carina.ICPreconditioner
        @test iterative("ildl").precond isa Carina.ICPreconditioner

        # Unknown solver types must fail loudly, not fall through to a default.
        @test_throws ErrorException ls(Dict{String,Any}("type" => "gmres"))

        # An integrator that provides no AMG factory must reject `amg` loudly
        # (the default factory is the error thunk).
        @test_throws ErrorException iterative("amg")

        # `assembled` is honored, not just accepted by the key validator:
        # false on CPU forces the matrix-free path; true on a GPU backend is
        # a request for a device sparse matrix that does not exist.
        @test ls(Dict{String,Any}("type" => "iterative")).assembled == true
        @test ls(Dict{String,Any}("type" => "iterative",
                                  "assembled" => false)).assembled == false
        struct _FakeGPU <: Carina.KA.GPU end
        @test_throws ErrorException Carina._parse_linear_solver(
            Dict{String,Any}("type" => "iterative", "assembled" => true),
            zeros(4), _FakeGPU(), () -> Carina.NoPreconditioner())
    end

    # ----- solver-level preconditioner (NLCG / steepest descent) -------------
    @testset "solver preconditioner values fail loudly" begin
        mk = () -> Carina.JacobiPreconditioner(zeros(4))
        parse_pc(pc) = Carina._parse_solver_preconditioner(
            Dict{String,Any}("preconditioner" => pc), mk)

        # Only Jacobi exists at this level; it used to be handed out for ANY
        # content, so `type: chebyshev` silently ran Jacobi.
        @test parse_pc(Dict{String,Any}("type" => "jacobi")) isa
              Carina.JacobiPreconditioner
        @test parse_pc(Dict{String,Any}()) isa Carina.JacobiPreconditioner
        @test parse_pc(Dict{String,Any}("type" => "none")) isa
              Carina.NoPreconditioner
        @test_throws ErrorException parse_pc(
            Dict{String,Any}("type" => "chebyshev"))
        # No preconditioner section at all: none.
        @test Carina._parse_solver_preconditioner(Dict{String,Any}(), mk) isa
              Carina.NoPreconditioner
    end

    # ----- whole-input key walker (ported back from Norma) -------------------
    @testset "input key walker" begin
        # Minimal clean deck, built in memory so each test can break one key.
        clean() = Dict{String,Any}(
            "type"             => "single",
            "input mesh file"  => "cube.g",
            "output mesh file" => "cube.e",
            "output interval"  => 0.5,
            "output"           => Dict{String,Any}("stress" => true),
            "model" => Dict{String,Any}(
                "type" => "solid mechanics",
                "material" => Dict{String,Any}(
                    "blocks" => Dict{String,Any}("cube" => "neohookean"),
                    "neohookean" => Dict{String,Any}(
                        "elastic modulus" => 1.0e9,
                        "Poisson's ratio" => 0.25,
                        "density"         => 1000.0))),
            "time integrator" => Dict{String,Any}(
                "type" => "newmark", "final time" => 1.0, "time step" => 0.5,
                "beta" => 0.25, "gamma" => 0.5),
            "boundary conditions" => Dict{String,Any}(
                "dirichlet" => Any[Dict{String,Any}(
                    "side set" => "ssz-", "component" => "z",
                    "function" => "0.0")]),
            "solver" => Dict{String,Any}(
                "type" => "newton",
                "linear solver" => Dict{String,Any}(
                    "type" => "iterative",
                    "preconditioner" => Dict{String,Any}(
                        "type" => "chebyshev", "degree" => 5))),
        )
        walk = Carina.validate_input_keys

        @test isempty(walk(clean()))

        # Top-level typo, with suggestion.
        d = clean(); d["outpt interval"] = 0.5; delete!(d, "output interval")
        msgs = walk(d)
        @test length(msgs) == 1
        @test occursin("outpt interval", msgs[1])
        @test occursin("output interval", msgs[1])   # the suggestion

        # Integrator keys are validated per type: Newmark keys on a
        # quasi-static deck warn, and CFL on a Newmark deck warns.
        d = clean(); d["time integrator"]["type"] = "quasi static"
        msgs = walk(d)
        @test any(occursin("\"beta\"", m) for m in msgs)
        @test any(occursin("\"gamma\"", m) for m in msgs)
        d = clean(); d["time integrator"]["CFL"] = 0.9
        @test any(occursin("\"CFL\"", m) for m in walk(d))
        # Unknown type validates against the union: only foreign keys warn
        # (the integrator parser aborts on the type itself).
        d = clean(); d["time integrator"]["type"] = "newmar"
        @test isempty(walk(d))

        # `degree` belongs to Chebyshev only.
        d = clean()
        d["solver"]["linear solver"]["preconditioner"]["type"] = "jacobi"
        @test any(occursin("\"degree\"", m) for m in walk(d))
        d = clean()
        d["solver"]["linear solver"]["preconditioner"] =
            Dict{String,Any}("type" => "chebyshev", "degre" => 5)
        msgs = walk(d)
        @test length(msgs) == 1 && occursin("degree", msgs[1])

        # Solver-level preconditioner (NLCG): type only.
        d = clean()
        d["solver"]["preconditioner"] = Dict{String,Any}(
            "type" => "jacobi", "degree" => 5)
        @test any(occursin("solver preconditioner", m) for m in walk(d))

        # Material section: an unreferenced property dict is typically a
        # renamed block whose old material was left behind.
        d = clean()
        d["model"]["material"]["old_material"] = Dict{String,Any}(
            "elastic modulus" => 1.0)
        @test any(occursin("old_material", m) for m in walk(d))
        # Property-key typo inside a referenced material.
        d = clean()
        d["model"]["material"]["neohookean"]["elastic modulos"] = 1.0
        msgs = walk(d)
        @test any(occursin("elastic modulos", m) && occursin("elastic modulus", m)
                  for m in msgs)

        # BC entry and output typos.
        d = clean()
        d["boundary conditions"]["dirichlet"][1]["sideset"] = "ssz-"
        @test any(occursin("\"sideset\"", m) for m in walk(d))
        d = clean(); d["output"]["stres"] = true; delete!(d["output"], "stress")
        @test any(occursin("\"stres\"", m) for m in walk(d))

        # Non-ASCII keys must not throw in the suggestion machinery
        # (character-indexed Levenshtein; "β" and "lamé's first constant"
        # are in the key sets).
        d = clean(); d["time integrator"]["ß"] = 0.25
        @test !isempty(walk(d))
    end

    # ----- shipped inputs are clean ------------------------------------------
    # Guard against schema drift: a keyword added to a parser and used in a
    # shipped deck fails here until the matching known-key set learns it.
    # Sweeps examples/ and benchmark/inputs/ (Carina decks only — the
    # crosscode directory holds Norma and LCM inputs too, which legitimately
    # use foreign keys).
    @testset "shipped inputs are clean" begin
        roots = (joinpath(@__DIR__, "..", "examples"),
                 joinpath(@__DIR__, "..", "benchmark", "inputs"))
        checked = 0
        for top in roots
            isdir(top) || continue
            for (root, dirs, files) in walkdir(top)
                for f in files
                    endswith(f, ".yaml") || continue
                    path = joinpath(root, f)
                    d = try
                        Carina.YAML.load_file(path; dicttype = Dict{String, Any})
                    catch
                        continue
                    end
                    d isa Dict || continue
                    get(d, "type", "") == "single" || continue
                    checked += 1
                    messages = Carina.validate_input_keys(d)
                    isempty(messages) ||
                        @error "Unknown input keys in shipped deck" path messages
                    @test isempty(messages)
                end
            end
        end
        # A broken walkdir path or filter must not pass as an empty sweep.
        @test checked > 50
    end

    # ----- max-iteration extraction ------------------------------------------
    @testset "_extract_max_iters fallbacks" begin
        @test Carina._extract_max_iters(Carina.MaxIterationsTest(12)) == 12
        # Leaf tests other than MaxIterations contribute no bound.
        @test Carina._extract_max_iters(Carina.AbsResidualTest(1e-8)) == 0
        # Nested combos are searched; the first positive bound wins.
        combo = Carina.ComboOrTest(Carina.AbstractStatusTest[
            Carina.ComboAndTest(Carina.AbstractStatusTest[
                Carina.AbsResidualTest(1e-8)]),
            Carina.MaxIterationsTest(7)])
        @test Carina._extract_max_iters(combo) == 7
    end

    # ----- BC entries must name a set ----------------------------------------
    @testset "BC entries missing their set fail loudly" begin
        dbc = Dict{String,Any}("boundary conditions" => Dict{String,Any}(
            "dirichlet" => Any[Dict{String,Any}(
                "component" => "z", "function" => "0.0")]))
        @test_throws ErrorException Carina._parse_dirichlet_bcs(dbc)

        nbc = Dict{String,Any}("boundary conditions" => Dict{String,Any}(
            "neumann" => Any[Dict{String,Any}(
                "component" => "z", "function" => "1.0")]))
        @test_throws ErrorException Carina._parse_neumann_bcs(nbc)
    end

    # ----- output spec defaults ----------------------------------------------
    @testset "OutputSpec default construction" begin
        spec = Carina.OutputSpec()
        @test !spec.velocity && !spec.acceleration && !spec.stress
        @test !spec.deformation_gradient && !spec.internal_variables
        @test spec.recovery == :none
    end

    # ----- unknown integrator type -------------------------------------------
    @testset "unknown time integrator type" begin
        # This error sits past assembler construction, so it needs a real
        # parse: run a minimal input whose integrator type is misspelled.
        example_dir = joinpath(@__DIR__, "..", "examples", "mechanics",
                               "quasistatic", "cube")
        yaml = """
        type: single
        input mesh file: cube.g
        output mesh file: cube_bad_ti.e
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
          type: quasistatique
          initial time: 0.0
          final time: 1.0
          time step: 1.0
        boundary conditions:
          dirichlet:
            - side set: ssz-
              component: z
              function: "0.0"
        solver:
          type: newton
          linear solver:
            type: direct
        """
        mktempdir() do dir
            cp_example(joinpath(example_dir, "cube.g"), joinpath(dir, "cube.g"))
            path = joinpath(dir, "bad_integrator.yaml")
            open(io -> write(io, yaml), path, "w")
            err = try
                Carina.run(path)
                nothing
            catch e
                sprint(showerror, e)
            end
            @test err !== nothing
            @test occursin("Unknown time_integrator.type", err)
        end
    end

end
