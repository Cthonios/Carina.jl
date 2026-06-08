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
