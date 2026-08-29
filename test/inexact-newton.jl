# Inexact Newton: Eisenstat-Walker forcing terms.
#
# Carina used to drive every Newton iteration's linear solve to the deck's
# `tolerance`, however far the *nonlinear* residual still was from converged.
# Measured on the J2 tension specimen, CG spent ~340 iterations on each of the
# first seven Newton iterations of a step while ‖R‖ crawled 1.57e3 → 1.76e2 —
# eight digits of linear accuracy thrown away by a line search that then
# backtracked to α = 0.25.
#
# The forcing term makes that tolerance per-iteration.  What these tests pin
# down is the safety property that makes it enable-able: a forcing term may
# only ever *loosen* a solve relative to the deck's tolerance, never tighten
# one, so the converged answer is unchanged and an A/B compares like with like.

@testset "Inexact Newton (Eisenstat-Walker)" begin

    # ----- residual target extraction ---------------------------------------
    # The over-solve guard needs the residual norm Newton is actually aiming
    # at, which lives in the status test tree -- not in the solver's tolerance
    # fields, which a deck's `termination:` block bypasses entirely.
    @testset "_residual_target" begin
        R0 = 1.0e3

        @test Carina._residual_target(Carina.AbsResidualTest(1e-8), R0) == 1e-8
        @test Carina._residual_target(Carina.RelResidualTest(1e-12), R0) == 1e-12 * R0
        # A relative test says nothing until there is an initial norm to scale.
        @test Carina._residual_target(Carina.RelResidualTest(1e-12), 0.0) == 0.0

        # OR: converging on either is enough, so the looser one is the target.
        or_tree = Carina.ComboOrTest(Carina.AbstractStatusTest[
            Carina.AbsResidualTest(1e-8),
            Carina.RelResidualTest(1e-12),
            Carina.MaxIterationsTest(16),
            Carina.FiniteValueTest(),
        ])
        @test Carina._residual_target(or_tree, R0) == 1e-8

        # AND: both must hold, so the tighter one is the target.  Tests that
        # say nothing about the residual must not drag it to zero.
        and_tree = Carina.ComboAndTest(Carina.AbstractStatusTest[
            Carina.AbsResidualTest(1e-8),
            Carina.RelResidualTest(1e-12),
            Carina.MinIterationsTest(2),
        ])
        @test Carina._residual_target(and_tree, R0) == 1e-12 * R0

        # A tree with no residual test at all yields 0.0, which disables the
        # guard rather than inventing a target for it.
        @test Carina._residual_target(
            Carina.ComboOrTest(Carina.AbstractStatusTest[
                Carina.MaxIterationsTest(16),
                Carina.AbsUpdateTest(1e-9),
            ]), R0) == 0.0

        # The two shipped trees differ, and the target follows each.
        # `_build_status_test` (what NewtonSolver uses) is an OR, so either
        # tolerance suffices; `default_nonlinear_status_test` ANDs them, so
        # both must hold and the tighter one governs.
        @test Carina._residual_target(
            Carina.default_nonlinear_status_test(abs_tol=1e-10, rel_tol=1e-14),
            R0) == 1e-14 * R0
    end

    # ----- forcing term arithmetic ------------------------------------------
    ew_default() = Carina.EisenstatWalker(1.0, 0.5 * (1 + sqrt(5)), 0.5, 0.5, 0.5)

    function mkls(rtol, forcing)
        n = 4
        ws = Carina.Krylov.CgWorkspace(n, n, Vector{Float64})
        Carina.KrylovLinearSolver(100, rtol, true, Carina.NoPreconditioner(),
                                  ws, zeros(n), forcing, rtol, NaN, NaN)
    end

    @testset "FixedForcing never moves the tolerance" begin
        ls = mkls(1e-8, Carina.FixedForcing())
        Carina._reset_forcing!(ls)
        for R in (7.10e3, 4.94e1, 1.09e-2, 1.89e-9)
            Carina._update_forcing!(ls, R, 1e-10)
            @test ls.rtol_eff == 1e-8
            # ...so the AMG detector sees the raw count, exactly as before.
            @test Carina._tracked_iters(ls, 137) == 137
        end
    end

    @testset "first iteration of a step uses eta_0" begin
        f  = Carina.EisenstatWalker(1.0, 1.618, 0.9, 0.3, 0.0)
        ls = mkls(1e-8, f)
        Carina._reset_forcing!(ls)
        Carina._update_forcing!(ls, 1.0e3, 0.0)
        @test ls.rtol_eff == 0.3

        # `_reset_forcing!` must clear the residual history, or the first
        # solve of step n+1 would form a ratio against step n's last residual
        # -- a number many orders smaller, which pins eta at rtol and silently
        # turns the forcing term off for the rest of the run.
        Carina._update_forcing!(ls, 1.0e3, 0.0)   # ratio 1.0 -> eta_max
        @test ls.rtol_eff == 0.9
        Carina._reset_forcing!(ls)
        @test ls.rtol_eff == 1e-8
        @test isnan(ls.eta_prev) && isnan(ls.norm_R_last)
        Carina._update_forcing!(ls, 1.0e3, 0.0)
        @test ls.rtol_eff == 0.3
    end

    @testset "eta is clamped to [rtol, eta_max]" begin
        # The lower clamp is the safety property: the forcing term can loosen
        # a solve but never tighten one past what the deck asked for, so the
        # final Newton iterations run at exactly the deck's tolerance.
        ls = mkls(1e-8, ew_default())
        Carina._reset_forcing!(ls)
        for R in (1e3, 1e-1, 1e-7, 1e-12, 1e-20, 0.0, 1e3, Inf)
            Carina._update_forcing!(ls, R, 1e-10)
            @test isfinite(ls.rtol_eff)
            @test 1e-8 <= ls.rtol_eff <= 0.5
        end
    end

    @testset "EW safeguard holds eta up when convergence stalls" begin
        # gamma * eta_prev^alpha > 0.1 means the previous solve's tolerance
        # does not yet justify a tighter one, whatever the residual ratio says.
        f  = Carina.EisenstatWalker(1.0, 2.0, 0.5, 0.5, 0.0)
        ls = mkls(1e-12, f)
        Carina._reset_forcing!(ls)
        Carina._update_forcing!(ls, 1.0e3, 0.0)         # eta_0
        @test ls.rtol_eff == 0.5
        # Ratio alone would give (1e-3)^2 = 1e-6; the safeguard floors it at
        # gamma * 0.5^2 = 0.25.
        Carina._update_forcing!(ls, 1.0, 0.0)
        @test ls.rtol_eff ≈ 0.25

        # Once eta_prev^alpha drops below 0.1 the safeguard lets go and the
        # ratio takes over.
        Carina._update_forcing!(ls, 1.0e-6, 0.0)        # 0.25^2 = 0.0625 < 0.1
        @test ls.rtol_eff ≈ 1.0e-12 atol=1e-13
    end

    @testset "over-solve guard keeps the last solve from being over-worked" begin
        # Newton is aiming at tau; resolving the linear system far below what
        # tau needs buys nothing.  safety * tau / ‖R‖ is the floor on eta.
        #
        # eta_0 = 0.3 with alpha = 2 puts gamma * eta_prev^alpha at 0.09, just
        # under the 0.1 threshold, so EW's own safeguard stays out of the way
        # and this measures the over-solve guard alone.
        f  = Carina.EisenstatWalker(1.0, 2.0, 0.5, 0.3, 0.5)
        ls = mkls(1e-12, f)
        Carina._reset_forcing!(ls)
        Carina._update_forcing!(ls, 1.0e3, 1e-10)
        @test ls.rtol_eff == 0.3
        Carina._update_forcing!(ls, 1.0e-6, 1e-10)      # ratio would give 1e-18
        @test ls.rtol_eff ≈ 0.5 * 1e-10 / 1.0e-6

        # safety = 0 disables the guard, and tau = 0 (a tree with no residual
        # test) has the same effect.
        for (safety, tau) in ((0.0, 1e-10), (0.5, 0.0))
            f2  = Carina.EisenstatWalker(1.0, 2.0, 0.5, 0.3, safety)
            ls2 = mkls(1e-12, f2)
            Carina._reset_forcing!(ls2)
            Carina._update_forcing!(ls2, 1.0e3, tau)
            Carina._update_forcing!(ls2, 1.0e-6, tau)
            @test ls2.rtol_eff == 1e-12
        end
    end

    @testset "_tracked_iters normalizes counts for the AMG detector" begin
        # The detector latches a baseline from the first CG count after a build
        # and rebuilds once a later count passes 3x it.  With a forcing term
        # running, those counts come from solves at different tolerances and
        # are not comparable as raw numbers.
        ls = mkls(1e-8, ew_default())
        Carina._reset_forcing!(ls)

        # A full-tolerance solve is the identity.  This is the property that
        # keeps every non-forcing run feeding the detector exactly what it fed
        # before -- see test/amg-staleness.jl for the behavior that rests on it.
        @test ls.rtol_eff == ls.rtol
        for n in (5, 40, 400)
            @test Carina._tracked_iters(ls, n) == n
        end

        # A solve loosened to eta = 0.5 asked for log10(1/0.5) = 0.301 digits
        # of reduction instead of 8, so its count stands in for ~26.6x as many.
        Carina._update_forcing!(ls, 1.0e3, 1e-10)
        @test ls.rtol_eff == 0.5
        @test Carina._tracked_iters(ls, 2) == round(Int, 2 * 8 / log10(2))

        # Rescaling is monotone in eta: the looser the solve, the larger the
        # equivalent count a given raw number stands for.
        counts = map((0.5, 0.1, 1e-4)) do eta
            ls.rtol_eff = eta
            Carina._tracked_iters(ls, 100)
        end
        @test issorted(counts, rev=true)

        # Degenerate tolerances have no meaningful rescaling; the raw count is
        # handed back rather than a NaN or a negative iteration count.
        ls.rtol_eff = 1.0
        @test Carina._tracked_iters(ls, 100) == 100
        ls_bad = mkls(1.0, ew_default())
        ls_bad.rtol_eff = 0.5
        @test Carina._tracked_iters(ls_bad, 100) == 100

        # Non-Krylov solvers have no forcing state at all.
        @test Carina._tracked_iters(Carina.DirectLinearSolver(), 42) == 42
        @test Carina._reset_forcing!(Carina.DirectLinearSolver()) === nothing
        @test Carina._update_forcing!(Carina.NoLinearSolver(), 1.0, 1e-8) === nothing
    end

    # ----- deck parsing ------------------------------------------------------
    @testset "deck parsing" begin
        base(extra...) = Dict{String,Any}(
            "type" => "cg",
            "tolerance" => 1.0e-8,
            "maximum iterations" => 500,
            extra...,
        )
        template = zeros(Float64, 8)
        parse_ls(d) = Carina._parse_linear_solver(d, template, Carina.KA.CPU(),
                                                  () -> Carina.NoPreconditioner())

        # No `forcing term:` block is the old behavior, and must stay so:
        # every deck written before this existed solves at a fixed tolerance.
        ls = parse_ls(base())
        @test ls.forcing isa Carina.FixedForcing
        @test ls.rtol_eff == 1.0e-8

        for name in ("eisenstat-walker", "Eisenstat Walker", "ew", "adaptive")
            ls = parse_ls(base("forcing term" => Dict{String,Any}("type" => name)))
            @test ls.forcing isa Carina.EisenstatWalker
        end
        for name in ("fixed", "none", "constant")
            ls = parse_ls(base("forcing term" => Dict{String,Any}("type" => name)))
            @test ls.forcing isa Carina.FixedForcing
        end

        # Omitting `type` inside the block means the caller wanted a forcing
        # term -- there is no other reason to write the block.
        ls = parse_ls(base("forcing term" => Dict{String,Any}()))
        @test ls.forcing isa Carina.EisenstatWalker

        # Defaults, and every knob honored.
        f = parse_ls(base("forcing term" => Dict{String,Any}())).forcing
        @test f.gamma == 1.0
        @test f.alpha ≈ 0.5 * (1 + sqrt(5))
        @test f.eta_max == 0.2
        @test f.eta_0 == f.eta_max      # `initial` defaults to `maximum`
        @test f.safety == 0.5

        f = parse_ls(base("forcing term" => Dict{String,Any}(
            "type" => "ew", "gamma" => 0.9, "exponent" => 2.0,
            "maximum" => 0.8, "initial" => 0.1, "safety factor" => 0.0))).forcing
        @test (f.gamma, f.alpha, f.eta_max, f.eta_0, f.safety) ==
              (0.9, 2.0, 0.8, 0.1, 0.0)

        # A typo in `type` must stop the run.  Falling through to FixedForcing
        # would cost exactly the speedup the block was added to get, with
        # nothing in the log to say the setting never took effect.
        @test_throws ErrorException parse_ls(
            base("forcing term" => Dict{String,Any}("type" => "eisenstadt")))
        @test_throws ErrorException parse_ls(
            base("forcing term" => "eisenstat-walker"))

        # Out-of-range parameters: each bound is one the method needs, not a
        # style preference.
        bad(k, v) = base("forcing term" => Dict{String,Any}("type" => "ew", k => v))
        for (k, v) in (("gamma", 0.0), ("gamma", 1.5),
                       ("exponent", 1.0), ("exponent", 2.5),
                       ("maximum", 0.0), ("maximum", 1.0), ("maximum", 1.5),
                       ("initial", 0.0), ("safety factor", -0.1))
            @test_throws ErrorException parse_ls(bad(k, v))
        end
        # `initial` may not exceed `maximum`.
        @test_throws ErrorException parse_ls(base("forcing term" =>
            Dict{String,Any}("type" => "ew", "maximum" => 0.2, "initial" => 0.5)))
    end

    # ----- key validation ----------------------------------------------------
    @testset "key validation" begin
        deck(ft) = Dict{String,Any}("solver" => Dict{String,Any}(
            "linear solver" => Dict{String,Any}("type" => "cg", "forcing term" => ft)))

        @test isempty(Carina.validate_input_keys(deck(Dict{String,Any}(
            "type" => "ew", "gamma" => 1.0, "exponent" => 1.6,
            "maximum" => 0.5, "initial" => 0.5, "safety factor" => 0.5))))

        msgs = Carina.validate_input_keys(deck(Dict{String,Any}(
            "type" => "ew", "gama" => 1.0)))
        @test length(msgs) == 1
        @test occursin("gama", msgs[1])

        # The Eisenstat-Walker knobs do nothing under `fixed`, so setting one
        # there is a mistake worth naming.
        msgs = Carina.validate_input_keys(deck(Dict{String,Any}(
            "type" => "fixed", "gamma" => 1.0)))
        @test length(msgs) == 1
        @test occursin("gamma", msgs[1])

        # `forcing term` itself is a known linear-solver key.
        @test isempty(Carina.validate_input_keys(deck(Dict{String,Any}())))
    end

    # ----- end to end ---------------------------------------------------------
    @testset "same answer, fewer CG iterations" begin
        # The whole claim in one test: turning the forcing term on changes how
        # hard each linear solve is worked and nothing else about the answer.
        example_dir = joinpath(@__DIR__, "..", "examples", "mechanics",
                               "quasistatic", "cube")

        deck(name, forcing) = """
type: single

input mesh file: cube.g
output mesh file: $(name).e

model:
  type: solid mechanics
  material:
    blocks:
      cube: neohookean
    neohookean:
      elastic modulus: 10.0e9
      Poisson's ratio: 0.25
      density: 1000.0

time integrator:
  type: quasi static
  initial time: 0.0
  final time: 1.0
  time step: 0.1

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
      function: "1.0e-3 * t"

solver:
  type: newton
  termination:
    fail when any:
      - maximum iterations: 20
    converge when any:
      - absolute residual: 1.0e-6
      - relative residual: 1.0e-10
  linear solver:
    type: cg
    tolerance: 1.0e-10
    maximum iterations: 500
    preconditioner:
      type: jacobi
$(forcing)"""

        # `CG: n iters` lines, summed.  The suite suppresses per-input .log
        # files, so turn them back on for this test only.
        function run_and_count(dir, name, forcing)
            path = joinpath(dir, name * ".yaml")
            text = deck(name, forcing)
            # The deck is assembled by interpolation, so check it parses to the
            # shape intended before trusting a run made from it.  An indentation
            # slip here puts `forcing term:` at the top level, where the key
            # validator warns and the forcing term silently never engages.
            parsed = Carina.YAML.load(text)
            @test isempty(Carina.validate_input_keys(parsed))
            open(io -> write(io, text), path, "w")
            old = Carina.CARINA_WRITE_LOG_FILE[]
            Carina.CARINA_WRITE_LOG_FILE[] = true
            sim = try
                Carina.run(path)
            finally
                Carina.CARINA_WRITE_LOG_FILE[] = old
            end
            log = read(joinpath(dir, name * ".log"), String)
            iters = sum(parse(Int, m.captures[1])
                        for m in eachmatch(r"CG: (\d+) iters", log))
            return sim, iters, log
        end

        mktempdir() do dir
            cp_example(joinpath(example_dir, "cube.g"), joinpath(dir, "cube.g"))

            fixed_sim, fixed_iters, fixed_log =
                run_and_count(dir, "fixed", "")
            # Not a triple-quoted literal: Julia strips the common leading
            # indentation from those, which would drop this block to the top
            # level of the deck instead of nesting it under `linear solver:`.
            ew_sim, ew_iters, ew_log = run_and_count(dir, "ew",
                "    forcing term:\n      type: eisenstat-walker")

            @test fixed_sim.integrator.nonlinear_solver.linear_solver.forcing isa
                  Carina.FixedForcing
            @test ew_sim.integrator.nonlinear_solver.linear_solver.forcing isa
                  Carina.EisenstatWalker

            # Non-vacuity: the run has to have actually loosened a solve.  A
            # forcing term that never engages would pass every assertion below
            # while doing nothing at all.
            @test occursin("inexact Newton", ew_log)
            @test !occursin("inexact Newton", fixed_log)

            # The mechanism does what it exists to do: less linear work for
            # the same answer.  Note this is not the same claim as "faster" --
            # on a cube this small and this nearly linear, CG converges in ~17
            # iterations either way and the extra Newton iterations EW buys the
            # savings with cost more than they save.  Whether the trade pays is
            # a property of the problem and of `maximum`; see
            # benchmark/evidence/inexact_newton.txt for where it does.
            @test ew_iters < fixed_iters

            # Same physics.  The tolerances here are the ones
            # mechanics-quasistatic-cube-chebyshev.jl asserts against the
            # analytical solution, so both runs are pinned to the same answer
            # rather than merely to each other.
            for sim in (fixed_sim, ew_sim)
                avg = average_components(sim)
                mx  = maximum_components(sim)
                @test avg[3] ≈  5.00e-4 rtol=1e-4
                @test avg[1] ≈ -1.25e-4 rtol=1e-2
                @test avg[2] ≈ -1.25e-4 rtol=1e-2
                @test mx[3]  ≈  1.00e-3 rtol=1e-6
            end

            # And to each other, far tighter than either is to the analytical
            # value: the final Newton iterations run at the deck's own
            # tolerance in both runs, so the converged states must agree to
            # very near it.
            @test average_components(ew_sim) ≈ average_components(fixed_sim) rtol=1e-8
            @test maximum_components(ew_sim) ≈ maximum_components(fixed_sim) rtol=1e-8
        end
    end
end
