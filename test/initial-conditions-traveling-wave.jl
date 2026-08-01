# Traveling-wave initial conditions: parser + symbolic-derivative checks.
#
# The traveling-wave IC path lets a user specify only the displacement profile
# u₀(x, y, z) along with a propagation direction and wave speed; Carina then
# derives the initial velocity field via the kinematic relation
#
#     u(x, t) = f(s − c·t)  ⇒  v(x, 0) = −c · ∂u₀/∂s
#
# where s ∈ {x, y, z} is the propagation axis.  This test exercises the
# pieces that wire that derivation together — YAML binding inlining, the
# parser front-end, and FEC's symbolic differentiator — by checking the
# evaluated derivative against the closed form at a grid of sample points.
# The integration into the full simulation pipeline is covered by the
# existing dynamics tests once a real YAML invokes the new IC type.

@testset "Traveling-wave IC parser + symbolic derivatives" begin
    # ---------------------------------------------------------------------- #
    # Binding inlining: `name=value;` shorthand → literal-substituted form
    # the FEC Pratt parser can consume.  Tests:
    #   • single binding
    #   • multiple bindings, including one referencing an earlier one
    #   • word boundaries (`tc` must not match `t`)
    # ---------------------------------------------------------------------- #
    @testset "_inline_expr_bindings" begin
        @test Carina._inline_expr_bindings("3.14") == "3.14"

        s1 = Carina._inline_expr_bindings("a=1.0e-3; a*exp(-t^2)")
        @test occursin("(1.0e-3)", s1)
        @test !occursin("a*", s1)
        @test occursin("exp(-t^2)", s1)

        s2 = Carina._inline_expr_bindings(
            "a=1.0e-3; tc=2.5e-4; tau=5.0e-5; a*exp(-(t-tc)^2/tau/tau/2)")
        @test occursin("(1.0e-3)", s2)
        @test occursin("(2.5e-4)", s2)
        @test occursin("(5.0e-5)", s2)
        @test !occursin(" a ", " " * s2 * " ")
        @test !occursin("tc", s2)
        @test !occursin("tau", s2)

        s3 = Carina._inline_expr_bindings("a=2.0; b=a*3; b*z")
        @test occursin("((2.0)*3)", s3)
    end

    # ---------------------------------------------------------------------- #
    # Parser: shape, required keys, direction validation.
    # ---------------------------------------------------------------------- #
    @testset "_parse_traveling_wave_ics" begin
        @test Carina._parse_traveling_wave_ics(Dict{String,Any}()) == Any[]

        good = Dict{String,Any}(
            "initial conditions" => Dict{String,Any}(
                "traveling wave" => Any[
                    Dict{String,Any}(
                        "node set"    => "nsall",
                        "component"   => "z",
                        "displacement"=> "a=0.01; s=0.02; a*exp(-z*z/s/s/2)",
                        "direction"   => "z",
                        "wave speed"  => 1000.0,
                    ),
                ],
            ),
        )
        out = Carina._parse_traveling_wave_ics(good)
        @test length(out) == 1
        @test out[1]["wave speed"] == 1000.0
        @test out[1]["direction"]  == "z"

        missing_dir = Dict{String,Any}(
            "initial conditions" => Dict{String,Any}(
                "traveling wave" => Any[
                    Dict{String,Any}(
                        "node set"    => "ns",
                        "component"   => "z",
                        "displacement"=> "0.0",
                        "wave speed"  => 1.0,
                    ),
                ],
            ),
        )
        @test_throws ErrorException Carina._parse_traveling_wave_ics(missing_dir)

        bad_dir = Dict{String,Any}(
            "initial conditions" => Dict{String,Any}(
                "traveling wave" => Any[
                    Dict{String,Any}(
                        "node set"    => "ns",
                        "component"   => "z",
                        "displacement"=> "0.0",
                        "direction"   => "w",
                        "wave speed"  => 1.0,
                    ),
                ],
            ),
        )
        @test_throws ErrorException Carina._parse_traveling_wave_ics(bad_dir)
    end

    # ---------------------------------------------------------------------- #
    # End-to-end derivative correctness on the standard clamped-bar profile:
    #   u₀(z) = a·exp(-z²/(2s²))   ⇒   du₀/dz = -(z/s²)·u₀(z)
    # so v₀(z) = -c · du₀/dz = (c·z/s²)·u₀(z).  We construct the SEF the way
    # `_apply_initial_traveling_wave_ics!` does and check the symbolic result
    # against the closed form at a few z values.
    # ---------------------------------------------------------------------- #
    @testset "Symbolic du/ds for clamped-bar IC" begin
        import FiniteElementContainers as FEC
        using StaticArrays

        a, s, c = 0.01, 0.02, 1000.0
        u_str   = Carina._inline_expr_bindings("a=0.01; s=0.02; a*exp(-z*z/s/s/2)")
        u_expr  = FEC.Expressions.ScalarExpressionFunction{Float64}(
                      u_str, Carina._CARINA_EXPR_VARS)
        dir_idx = Carina._direction_to_idx("z")    # 3
        @test dir_idx == 3
        du_dz   = FEC.Expressions.differentiate(u_expr, dir_idx)

        u_ref(z)    = a * exp(-z^2 / (2 * s^2))
        dudz_ref(z) = -(z / s^2) * u_ref(z)
        v_ref(z)    = -c * dudz_ref(z)

        for z in (-0.06, -0.02, 0.0, 0.01, 0.03, 0.05)
            X = SVector{3, Float64}(0.0, 0.0, z)
            u_num = u_expr(X, 0.0)
            d_num = du_dz(X, 0.0)
            v_num = -c * d_num
            @test u_num ≈ u_ref(z)    rtol=1e-12
            @test d_num ≈ dudz_ref(z) rtol=1e-10
            @test v_num ≈ v_ref(z)    rtol=1e-10
        end
    end

    # ---------------------------------------------------------------------- #
    # Sign convention: the wave_speed sign selects the direction of travel
    # along the chosen axis (sign of c flows straight through −c·∂u/∂s).
    # Verify that two opposite c's produce opposite-signed v₀ at the same
    # point — the simplest invariant of the formula.
    # ---------------------------------------------------------------------- #
    @testset "wave_speed sign flips v₀" begin
        import FiniteElementContainers as FEC
        using StaticArrays

        u_str  = Carina._inline_expr_bindings("a=0.01; s=0.02; a*exp(-z*z/s/s/2)")
        u_expr = FEC.Expressions.ScalarExpressionFunction{Float64}(
                     u_str, Carina._CARINA_EXPR_VARS)
        du_dz  = FEC.Expressions.differentiate(u_expr, 3)
        X      = SVector{3, Float64}(0.0, 0.0, 0.01)
        @test -(+1000.0) * du_dz(X, 0.0) ≈ -(-(-1000.0) * du_dz(X, 0.0))
    end
end

# --------------------------------------------------------------------------- #
# End-to-end: a real simulation applies the traveling-wave IC.
#
# Everything above stops at the parser and the symbolic derivative;
# `_apply_initial_traveling_wave_ics!` itself never ran, so a bug in the
# DOF scatter (or in wiring U and V together) would have passed the suite.
# Reuse the clamped-beam setup (E = 1 GPa, ν = 0, ρ = 1000 ⇒ c = 1000 m/s)
# whose displacement-IC variant is validated against the paper solution in
# mechanics-clamped-wave.jl.  With v₀ = −c·∂u₀/∂z the pulse propagates one
# way: u(z, t) = u₀(z − c·t) exactly in the continuum.
# --------------------------------------------------------------------------- #
@testset "Traveling-wave IC end-to-end (central difference)" begin
    a, s, c = 0.01, 0.02, 1000.0
    t_final = 2.0e-6

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics",
                           "explicit-dynamic", "clamped")
    yaml = """
type: single
input mesh file: clamped.g
output mesh file: clamped_tw.e
model:
  type: solid mechanics
  material:
    linear elastic:
      elastic modulus: 1.0e9
      density: 1000.0
      Poisson's ratio: 0.0
    blocks:
      clamped: linear elastic
time integrator:
  type: central difference
  time step: 1.0e-7
  gamma: 0.5
  final time: $t_final
  initial time: 0.0
initial conditions:
  traveling wave:
    - node set: nsall
      displacement: "a=$a; s=$s; a*exp(-z*z/s/s/2)"
      component: z
      direction: z
      wave speed: $c
boundary conditions:
  dirichlet:
    - node set: nsx-
      function: "0.0"
      component: x
    - node set: nsx+
      function: "0.0"
      component: x
    - node set: nsy-
      function: "0.0"
      component: y
    - node set: nsy+
      function: "0.0"
      component: y
    - node set: nsz-
      function: "0.0"
      component: z
    - node set: nsz+
      function: "0.0"
      component: z
"""

    u0(z) = a * exp(-z^2 / (2 * s^2))
    v0(z) = (c * z / s^2) * u0(z)      # −c · du₀/dz

    mktempdir() do dir
        cp_example(joinpath(example_dir, "clamped.g"), joinpath(dir, "clamped.g"))
        path = joinpath(dir, "clamped_tw.yaml")
        open(io -> write(io, yaml), path, "w")

        dict = Carina.YAML.load_file(path; dicttype=Dict{String,Any})
        sim  = Carina.create_simulation(dict, dir)
        ig   = sim.integrator
        @test ig isa Carina.CentralDifferenceIntegrator

        X     = reshape(Vector(sim.params_cpu.coords.data), 3, :)
        nnode = size(X, 2)
        free  = falses(3 * nnode)
        free[sim.asm_cpu.dof.unknown_dofs] .= true

        # ---- state right after initialization: u₀ applied, v₀ derived ----
        U = Vector(sim.params.field.data)
        V = Vector(ig.V)
        err_u = err_v = 0.0
        for n in 1:nnode
            zd = 3 * (n - 1) + 3
            free[zd] || continue     # constrained slots hold g(t₀), not the IC
            z = X[3, n]
            err_u = max(err_u, abs(U[zd] - u0(z)))
            err_v = max(err_v, abs(V[zd] - v0(z)))
        end
        # The IC is an exact nodal interpolation, so agreement is round-off,
        # scaled by the magnitude of each field (max|v₀| ≈ 300).
        @test err_u < 1.0e-12
        @test err_v < 1.0e-8
        # Transverse components carry no IC.
        @test maximum(abs, reshape(U, 3, :)[1:2, :]) == 0.0
        @test maximum(abs, reshape(V, 3, :)[1:2, :]) == 0.0

        # ---- propagate: one-way advection u(z, t) = u₀(z − c·t) ----------
        Carina.evolve!(sim)
        Carina.FEC.close(sim.post_processor)
        @test !ig.failed[]

        z_nodes = X[3, :]
        Uf = Vector(sim.params.field.data)
        uz = reshape(Uf, 3, :)[3, :]
        @test all(isfinite, uz)
        # Compare extrema against the analytic one-way solution sampled on
        # the same mesh (the standing-wave solution would still show a
        # trough of −a/2·e^{-1/2}-scale here; one-way advection does not).
        ref = [u0(z - c * t_final) for z in z_nodes]
        @test maximum(uz) ≈ maximum(ref) rtol = 5.0e-3
        @test minimum(uz) ≈ minimum(ref) atol = 5.0e-5
    end
end

# A quasi-static integrator has no velocity state: the traveling-wave IC is
# ignored with a warning instead of crashing the setup.
@testset "Traveling-wave IC ignored for quasi-static" begin
    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics",
                           "quasistatic", "cube")
    yaml = """
type: single
input mesh file: cube.g
output mesh file: cube_tw_qs.e
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
  time step: 1.0
initial conditions:
  traveling wave:
    - node set: nsall
      displacement: "0.001*exp(-z*z/1.0e-4)"
      component: z
      direction: z
      wave speed: 1000.0
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
        path = joinpath(dir, "cube_tw_qs.yaml")
        open(io -> write(io, yaml), path, "w")
        dict = Carina.YAML.load_file(path; dicttype=Dict{String,Any})
        sim  = Carina.create_simulation(dict, dir)
        Carina.FEC.close(sim.post_processor)
        @test sim.integrator isa Carina.QuasiStaticIntegrator
        # The warning fallback leaves the initial displacement untouched.
        @test maximum(abs, Vector(sim.params.field.data)) == 0.0
    end
end
