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
        @test Carina._parse_material_section(good_material())[1] == "cube"

        # `model.type` names a physics that does not exist.  Previously the key
        # was read by nothing at all, so `thermal` ran as solid mechanics.
        d = good_material()
        d["model"]["type"] = "thermal"
        @test_throws ErrorException Carina._parse_material_section(d)

        # Omitting `type` remains legal.
        d = good_material()
        delete!(d["model"], "type")
        @test Carina._parse_material_section(d)[1] == "cube"

        # A second block used to be dropped, with `first(blocks_dict)` picking
        # the survivor in hash order and applying it to the whole mesh.
        d = good_material()
        d["model"]["material"]["blocks"]["shell"] = "linear elastic"
        @test_throws ErrorException Carina._parse_material_section(d)

        d = good_material()
        empty!(d["model"]["material"]["blocks"])
        @test_throws ErrorException Carina._parse_material_section(d)

        # `blocks` names a model with no matching property dict.
        d = good_material()
        d["model"]["material"]["blocks"]["cube"] = "hencky"
        @test_throws ErrorException Carina._parse_material_section(d)

        # The property dict is resolved case-insensitively, matching the
        # case-insensitive validation of the same keys.
        d = good_material()
        d["model"]["material"]["blocks"]["cube"] = "NeoHookean"
        d["model"]["material"]["NeoHookean"] = pop!(d["model"]["material"], "neohookean")
        @test Carina._parse_material_section(d)[3] == 1000.0
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

end
