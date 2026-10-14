@testset "Quasi-static tangent predictor" begin
    # The unit cube of mechanics-quasistatic-cube.jl, stretched in z by a
    # prescribed displacement in ten steps, solved with and without the
    # tangent predictor (`predictor: tangent`): one solve with the tangent of
    # the converged state before Newton starts a step.  The converged
    # solution is the same to the solver tolerance; the predictor changes the
    # first iterate only.  Both linear-solver paths of the predictor are
    # exercised: the direct solver (assembled tangent) and the iterative
    # solver with the Jacobi preconditioner (matrix-free tangent action).
    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics", "quasistatic", "cube")
    deck = read(joinpath(example_dir, "cube.yaml"), String)
    iterative = """
  linear solver:
    type: iterative
    tolerance: 1.0e-12
    maximum iterations: 500
    preconditioner:
      type: jacobi
"""
    function run_case(dir, predictor, direct)
        text = replace(deck, "  type: quasi static\n" =>
                             "  type: quasi static\n  predictor: $predictor\n")
        direct || (text = replace(text, "  linear solver:\n    type: direct\n" => iterative))
        path = joinpath(dir, "cube-$predictor-$(direct ? "direct" : "iterative").yaml")
        write(path, text)
        sim = Carina.run(path)
        return average_components(sim), maximum_components(sim)
    end
    mktempdir() do dir
        cp_example(joinpath(example_dir, "cube.g"), joinpath(dir, "cube.g"))
        avg0, mx0 = run_case(dir, "none", true)
        for direct in (true, false)
            avg, mx = run_case(dir, "tangent", direct)
            @test avg[3] ≈ avg0[3] rtol=1e-8
            @test avg[1] ≈ avg0[1] rtol=1e-6
            @test avg[2] ≈ avg0[2] rtol=1e-6
            @test mx[3]  ≈ mx0[3]  rtol=1e-12
        end
        # Bad input fails loudly.
        bad = replace(deck, "  type: quasi static\n" =>
                            "  type: quasi static\n  predictor: secant\n")
        write(joinpath(dir, "bad.yaml"), bad)
        @test_throws ErrorException Carina.run(joinpath(dir, "bad.yaml"))
    end
end
