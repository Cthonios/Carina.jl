# The general form of the projected volumetric formulation
# (model.volumetric form: general; src/projected_physics.jl, "General form"):
# any material is evaluated at F̃ = (J̃/J)^{1/3} F with J̃ = θ⁻¹(P_h θ(J)).
#
# Checks, on the TETRA15 cube:
#   1. selection of the form and of θ from the input, and the input errors;
#   2. with the neo-Hookean material (no split), θ = log J and J − 1, both
#      projections: the residual is the derivative of the energy, the tangent
#      is the derivative of the residual and is symmetric, the matrix-free
#      action, the diagonal kernels and the Newmark kernels reproduce the
#      assembled matrices, and Newton converges quadratically;
#   3. the linear patch test and an explicit run with the general form;
#   4. the J2 model through the general form with θ = J − 1 reproduces the
#      split form (energy, residual, tangent, Newton history, displacement and
#      internal variables) in an elastic and in a plastic case.  In plastic
#      flow the J2 tangent is the symmetric part of the derivative of its
#      stress (BOX 9.2 of Simo and Hughes), so the plastic tangent is
#      verified by this equality and not by differences of the residual;
#   5. on a GPU, when one is present: the general form runs, and its Newton
#      history and displacement agree with those of the CPU.

using LinearAlgebra: norm, diag

# A status test that records the residual norms Newton reports to it: the
# initial norm of every solve, then the norm after every iteration.
mutable struct _RecordingTest <: Carina.AbstractStatusTest
    inner::Carina.AbstractStatusTest
    norms::Vector{Float64}
end
function Carina.check(t::_RecordingTest, info::Carina.SolverInfo)
    info.iteration == 1 && push!(t.norms, info.norm_R_init)
    push!(t.norms, info.norm_R)
    return Carina.check(t.inner, info)
end
Carina.reset!(t::_RecordingTest) = Carina.reset!(t.inner)

@testset "Projected volumetric formulation, general form" begin
    FEC = Carina.FEC
    mesh_dir(v) = joinpath(@__DIR__, "..", "examples", "meshes", "cube-$v")

    neohookean = """
    blocks:
      cube: neohookean
    neohookean:
      elastic modulus: 1.0e9
      Poisson's ratio: 0.3
      density: 1000.0"""
    j2(yield) = """
    blocks:
      cube: j2 plasticity
    j2 plasticity:
      elastic modulus: 1.0e9
      Poisson's ratio: 0.3
      density: 1000.0
      yield stress: $yield
      hardening modulus: 1.0e8"""

    # Triple-quoted strings lose their common indentation; this restores it.
    indent(text, n) = join((" "^n * l for l in split(text, '\n')), '\n')
    linear_solver_direct = "type: direct"
    linear_solver_iterative = """
    type: iterative
    tolerance: 1.0e-12
    maximum iterations: 2000
    preconditioner:
      type: jacobi"""

    function deck(; material = neohookean, projection = "linear", keys = "",
                  integrator = "quasi static", linear_solver = linear_solver_direct,
                  time_step = 0.5, compression = 2.0e-2)
        return """
type: single
input mesh file: cube.g
output mesh file: general.e
model:
  type: solid mechanics
  volumetric projection: $projection
$keys  material:
$(indent(material, 4))
time integrator:
  type: $integrator
  initial time: 0.0
  final time: 1.0
  time step: $time_step
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
      function: "-$compression * t"
    - side set: ssx+
      component: x
      function: "1.0e-2 * t * z"
solver:
  type: newton
  linear solver:
$(indent(linear_solver, 4))
  termination:
    fail when any:
      - maximum iterations: 32
    converge when any:
      - absolute residual: 1.0e-9
      - relative residual: 1.0e-12
"""
    end
    model_keys(; strain = nothing, form = nothing) =
        (strain === nothing ? "" : "  volumetric strain: $strain\n") *
        (form === nothing ? "" : "  volumetric form: $form\n")

    function build(dir, yaml_text; backend = Carina.KA.CPU())
        cp_example(joinpath(mesh_dir("tet15"), "cube.g"), joinpath(dir, "cube.g"))
        path = joinpath(dir, "general.yaml")
        open(io -> write(io, yaml_text), path, "w")
        dict = Carina.YAML.load_file(path; dicttype = Dict{String, Any})
        return Carina.create_simulation(dict, dir; backend = backend)
    end

    # Evolve with the residual norms of every Newton iteration recorded.
    function run!(sim)
        rec = _RecordingTest(Carina._nonlinear_status_test[], Float64[])
        Carina._nonlinear_status_test[] = rec
        try
            Carina.evolve!(sim)
        finally
            Carina._nonlinear_status_test[] = rec.inner
        end
        FEC.close(sim.post_processor)
        return sim, rec.norms
    end

    energy(asm, U, p) = (FEC.assemble_scalar!(asm, FEC.energy, U, p);
                         sum(asm.scalar_quadrature_storage))
    function residual(asm, U, p)
        FEC.assemble_vector!(asm, FEC.residual, U, p)
        return copy(FEC.residual(asm))
    end
    function stiffness(asm, U, p)
        FEC.assemble_stiffness!(asm, FEC.stiffness, U, p)
        return copy(FEC.stiffness(asm))
    end
    perturb(U) = U .+ 2.0e-4 .* sin.(0.7 .* (1:length(U)))

    @testset "input: form, volumetric strain and their errors" begin
        mktempdir() do dir
            ph = first(values(build(dir, deck()).params_cpu.physics))
            @test ph isa Carina.ProjectedSolidMechanics
            @test Carina.volumetric_form(ph) isa Carina.GeneralForm
            @test Carina.volumetric_variable(ph) isa Carina.LogJ
        end
        mktempdir() do dir
            ph = first(values(build(dir, deck(keys = model_keys(strain = "J - 1"))).params_cpu.physics))
            @test Carina.volumetric_variable(ph) isa Carina.JMinusOne
        end
        mktempdir() do dir   # a split model selects the split form, with its own θ
            ph = first(values(build(dir, deck(material = j2(1.0e12))).params_cpu.physics))
            @test Carina.volumetric_form(ph) isa Carina.SplitForm
            @test Carina.volumetric_variable(ph) isa Carina.JMinusOne
        end
        mktempdir() do dir   # the material's θ written out is accepted
            ph = first(values(build(dir, deck(material = j2(1.0e12),
                                              keys = model_keys(strain = "J - 1"))).params_cpu.physics))
            @test Carina.volumetric_form(ph) isa Carina.SplitForm
        end
        mktempdir() do dir   # a split model forced to the general form
            ph = first(values(build(dir, deck(material = j2(1.0e12),
                                              keys = model_keys(form = "general"))).params_cpu.physics))
            @test Carina.volumetric_form(ph) isa Carina.GeneralForm
            @test Carina.volumetric_variable(ph) isa Carina.LogJ
        end
        mktempdir() do dir   # a split model with a different θ is refused
            err = try
                build(dir, deck(material = j2(1.0e12), keys = model_keys(strain = "log J"))); nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("block \"cube\"", err.msg)
            @test occursin("the volumetric strain log J differs from it", err.msg)
            @test occursin("θ = J - 1", err.msg)
        end
        mktempdir() do dir   # the split form for a model without the split is refused
            err = try
                build(dir, deck(keys = model_keys(form = "split"))); nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("has none", err.msg)
        end
        model(d) = Dict{String, Any}("model" => merge(Dict{String, Any}("volumetric projection" => "linear"), d))
        @test Carina._parse_volumetric_strain(model(Dict("volumetric strain" => "log J"))) === Carina.LogJ()
        @test Carina._parse_volumetric_strain(model(Dict("volumetric strain" => "J-1"))) === Carina.JMinusOne()
        @test Carina._parse_volumetric_strain(model(Dict{String, Any}())) === nothing
        err = try
            Carina._parse_volumetric_strain(model(Dict("volumetric strain" => "ln J"))); nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("\"log J\", \"J - 1\"", err.msg)
        @test Carina._parse_volumetric_form(model(Dict("volumetric form" => "General"))) === :general
        @test Carina._parse_volumetric_form(model(Dict{String, Any}())) === :automatic
        @test_throws ErrorException Carina._parse_volumetric_form(model(Dict("volumetric form" => "mixed")))
        # either key without a projection is an error
        @test_throws ErrorException Carina._parse_volumetric_strain(
            Dict{String, Any}("model" => Dict{String, Any}("volumetric strain" => "log J")))
        @test_throws ErrorException Carina._parse_volumetric_form(
            Dict{String, Any}("model" => Dict{String, Any}("volumetric form" => "general")))
    end

    @testset "neo-Hookean, θ = $strain, $projection projection: consistency" for
            strain in ("log J", "J - 1"), projection in ("constant", "linear")
        mktempdir() do dir
            sim, _ = run!(build(dir, deck(projection = projection,
                                          keys = model_keys(strain = strain))))
            ig  = sim.integrator
            asm = ig.asm
            p   = sim.params
            U   = copy(ig.U)
            @test maximum(abs, U) > 1e-3
            @test maximum(abs, residual(asm, U, p)) < 1e-6 * 1e9   # converged state

            # 1. residual = dW/dU, central differences on a sample of dofs
            U = perturb(U)
            R = residual(asm, U, p)
            h = 1.0e-6
            worst = 0.0
            for k in (1, 7, 40, 101, 233, length(U) - 3)
                Up = copy(U); Up[k] += h
                Um = copy(U); Um[k] -= h
                dW = (energy(asm, Up, p) - energy(asm, Um, p)) / (2h)
                worst = max(worst, abs(dW - R[k]) / max(abs(R[k]), 1e-3 * maximum(abs, R)))
            end
            @test worst < 1e-6

            # 2. tangent = dR/dU along two directions, and symmetric
            K = stiffness(asm, U, p)
            @test norm(K - K') / norm(K) < 1e-12
            worst = 0.0
            for seed in (0.31, 0.77)
                v  = [sin(seed * i) for i in 1:length(U)]
                dR = (residual(asm, U .+ h .* v, p) .- residual(asm, U .- h .* v, p)) ./ (2h)
                Kv = K * v
                worst = max(worst, norm(dR - Kv) / norm(Kv))
            end
            @test worst < 1e-6

            # 3. matrix-free action and diagonal kernels against K
            v = [cos(0.37 * i) for i in 1:length(U)]
            Kv = K * v
            y = similar(v)
            Carina._stiffness_matvec_qs!(y, v, asm, U, p)
            @test norm(y - Kv) / norm(Kv) < 1e-12
            FEC.assemble_diagonal!(asm, Carina.StiffnessDiagonal(), U, p)
            @test isapprox(FEC.diagonal(asm), diag(K); rtol = 1e-12)
            free = asm.dof.unknown_dofs
            pos  = Dict(d => k for (k, d) in enumerate(free))
            for col in 1:3
                FEC.assemble_diagonal!(asm, Carina.StiffnessBlockColumn{col}(), U, p)
                B = copy(FEC.diagonal(asm))
                worst = 0.0
                checked = 0
                for (k, d) in enumerate(free)
                    n = (d - 1) ÷ 3 + 1
                    kc = get(pos, 3 * (n - 1) + col, 0)
                    kc == 0 && continue
                    worst = max(worst, abs(B[k] - K[k, kc]) / maximum(abs, diag(K)))
                    checked += 1
                end
                @test checked > 100
                @test worst < 1e-12
            end
            FEC.assemble_mass!(asm, FEC.mass, U, p)
            Mm = FEC.mass(asm)
            c_M = 3.7
            FEC.assemble_diagonal!(asm, Carina.NewmarkDiagonal(c_M), U, p)
            @test isapprox(FEC.diagonal(asm), diag(K) .+ c_M .* diag(Mm); rtol = 1e-12)
            Carina._assemble_action!(asm, Carina.NewmarkAction(c_M), U, v, p)
            y = copy(asm.stiffness_action_storage[asm.dof.unknown_dofs])
            @test norm(y - (Kv .+ c_M .* (Mm * v))) / norm(Kv) < 1e-12

            # 4. Newton from a perturbed state converges quadratically
            Un = perturb(perturb(copy(ig.U)))
            r = Float64[]
            for _ in 1:8
                Rn = residual(asm, Un, p)
                push!(r, norm(Rn))
                r[end] < 1e-13 * r[1] && break
                Un .-= stiffness(asm, Un, p) \ Rn
            end
            @test r[end] < 1e-10 * r[1]
            @test length(r) <= 6
            # the exponent of the last reduction above the rounding level
            k = findlast(x -> x > 1e-11 * r[1], r)
            @test k !== nothing && k < length(r)
            @test log(r[k + 1] / r[1]) / log(r[k] / r[1]) > 1.8 ||
                  r[k + 1] < 1e-12 * r[1]
        end
    end

    @testset "linear patch test" begin
        exact(x, y, z) = (1.0e-3 * (x + 2y + 3z),
                          1.0e-3 * (4x - y + 0.5z),
                          1.0e-3 * (0.25x + 1.5y + 2z))
        funcs = ("1.0e-3 * (x + 2.0*y + 3.0*z)",
                 "1.0e-3 * (4.0*x - y + 0.5*z)",
                 "1.0e-3 * (0.25*x + 1.5*y + 2.0*z)")
        bcs = join(("    - side set: $s\n      component: $c\n      function: \"$f\"\n"
                    for (c, f) in zip(("x", "y", "z"), funcs)
                    for s in ("ssx-", "ssx+", "ssy-", "ssy+", "ssz-", "ssz+")), "")
        patch = replace(deck(),
            r"boundary conditions:\n  dirichlet:\n(    - .*\n|      .*\n)*" =>
            "boundary conditions:\n  dirichlet:\n" * bcs,
            "time step: 0.5" => "time step: 1.0")
        @test occursin("ssx+\n      component: z", patch)
        mktempdir() do dir
            sim, _ = run!(build(dir, patch))
            u = _field_matrix(sim)
            X = reshape(adapt(Array, sim.params_cpu.coords.data), 3, :)
            worst = 0.0
            for n in axes(X, 2)
                e = exact(X[1, n], X[2, n], X[3, n])
                worst = max(worst, maximum(abs(u[i, n] - e[i]) for i in 1:3))
            end
            @test worst < 1e-10
            @test maximum(abs, u) > 1e-4
        end
    end

    @testset "explicit integration" begin
        extra = "  time step: 1.0e-7\n  final time: 3.0e-7"
        text = replace(deck(integrator = "central difference"),
                       "  time step: 0.5\n" => "", "  final time: 1.0\n" => extra * "\n")
        mktempdir() do dir
            sim, _ = run!(build(dir, text))
            @test all(>(0.0), adapt(Array, sim.integrator.m_lumped))
            u = adapt(Array, sim.params.field.data)
            @test all(isfinite, u)
            @test maximum(abs, u) > 0
        end
    end

    # J2 through the general form with θ = J − 1 against the split form.
    # The plastic case is one load step from the virgin state.  After a
    # plastic step the points that yielded lie on the yield surface, and at
    # the start of the next step the J2 return map decides between its
    # elastic and its plastic branch on f_trial = 0 to rounding; the two forms
    # compute the trial state from F and from s F, which differ by rounding,
    # and may take different branches there.  The stress is continuous across
    # the yield surface but the tangent is not, so the Newton histories of
    # that step differ (by 1.5% at its first iteration on this mesh) while
    # both converge to the same state.
    rel(a, b) = norm(a - b) / norm(b)
    @testset "J2, general form against split form: $regime" for (regime, yield, dt) in
            (("elastic", 1.0e12, 0.5), ("plastic", 5.0e6, 1.0))
        mktempdir() do dir_s
            mktempdir() do dir_g
                sim_s, hist_s = run!(build(dir_s, deck(material = j2(yield), time_step = dt)))
                sim_g, hist_g = run!(build(dir_g, deck(material = j2(yield), time_step = dt,
                                       keys = model_keys(strain = "J - 1", form = "general"))))
                @test Carina.volumetric_form(first(values(sim_s.params_cpu.physics))) isa Carina.SplitForm
                @test Carina.volumetric_form(first(values(sim_g.params_cpu.physics))) isa Carina.GeneralForm
                eqps = maximum(sim_s.params.state_old.data[10:10:end])
                if regime == "plastic"
                    @test eqps > 1e-3
                else
                    @test eqps == 0
                end
                # Newton history, displacement and internal variables
                @test length(hist_g) == length(hist_s)
                @test maximum(abs.(hist_g .- hist_s)) / hist_s[1] < 1e-12
                @test rel(sim_g.integrator.U, sim_s.integrator.U) < 1e-12
                @test rel(sim_g.params.state_old.data, sim_s.params.state_old.data) < 1e-12
                # energy, residual and tangent at one perturbed state
                U = perturb(copy(sim_s.integrator.U))
                a_s, p_s = sim_s.integrator.asm, sim_s.params
                a_g, p_g = sim_g.integrator.asm, sim_g.params
                @test abs(energy(a_g, U, p_g) - energy(a_s, U, p_s)) / abs(energy(a_s, U, p_s)) < 1e-12
                @test rel(residual(a_g, U, p_g), residual(a_s, U, p_s)) < 1e-12
                @test rel(stiffness(a_g, U, p_g), stiffness(a_s, U, p_s)) < 1e-12
                @test rel(p_g.state_new.data, p_s.state_new.data) < 1e-12
            end
        end
    end

    backend = test_best_device()
    if backend isa Carina.KA.CPU
        @info "No GPU detected; the device run of the general form is skipped"
    else
        @testset "general form on $backend" begin
            text = deck(linear_solver = linear_solver_iterative)
            results = map((Carina.KA.CPU(), backend)) do dev
                mktempdir() do dir
                    sim, hist = run!(build(dir, text; backend = dev))
                    (copy(Array(sim.params.field.data)), hist)
                end
            end
            (u_cpu, h_cpu), (u_gpu, h_gpu) = results
            @test length(h_gpu) == length(h_cpu)
            @test maximum(abs.(h_gpu .- h_cpu)) / h_cpu[1] < 1e-8
            @test rel(u_gpu, u_cpu) < 1e-8
        end
    end
end
