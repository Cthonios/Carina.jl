using LinearAlgebra: norm, diag

@testset "Matrix-free operators" begin
    # The matrix-free path is what Carina runs on a GPU: instead of assembling a
    # sparse K, the Krylov solve applies K through `FEC.stiffness_action` and
    # preconditions with a diagonal extracted by a kernel.
    #
    # `_parse_linear_solver` picks the path with `assembled = backend isa KA.CPU`,
    # so a CPU run always takes the assembled branch and there is no input-file
    # knob to override it. That left every matrix-free operator untested on CI,
    # which has no GPU. `KrylovLinearSolver` is mutable, so these tests build the
    # simulation from YAML and flip `assembled` before stepping, exercising the
    # operators themselves on the CPU. What stays GPU-specific is only the array
    # backend, not the code under test here.

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics", "quasistatic", "cube")

    # Same problem as mechanics-quasistatic-cube.jl: uniaxial stretch of a cube,
    # u_z = 1.0e-3 prescribed on the +z face, so avg u_z is 5.0e-4 by linearity.
    qs_yaml(precond) = """
type: single

input mesh file: cube.g
output mesh file: cube_mf.e

model:
  type: solid mechanics
  material:
    blocks:
      cube: neohookean
    neohookean:
      elastic modulus: 1.0e10
      Poisson's ratio: 0.25
      density: 1000.0

time integrator:
  type: quasi static
  initial time: 0.0
  final time: 1.0
  time step: 0.5

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
      - maximum iterations: 16
    converge when any:
      - absolute residual: 1.0e-6
      - relative residual: 1.0e-10
  linear solver:
    type: iterative
    tolerance: 1.0e-10
    maximum iterations: 500
    preconditioner:
$(precond)
"""

    # Build a simulation from YAML text without running it, so `assembled` can be
    # flipped first. `matrix_free = false` gives the ordinary assembled run to
    # compare against.
    function build_sim(dir, yaml_text, name; matrix_free::Bool)
        cp_example(joinpath(example_dir, "cube.g"), joinpath(dir, "cube.g"))
        path = joinpath(dir, name)
        open(io -> write(io, yaml_text), path, "w")
        dict = Carina.YAML.load_file(path; dicttype=Dict{String,Any})
        sim  = Carina.create_simulation(dict, dir)
        if matrix_free
            sim.integrator.nonlinear_solver.linear_solver.assembled = false
        end
        return sim
    end

    function run_sim!(sim)
        Carina.evolve!(sim)
        Carina.FEC.close(sim.post_processor)
        return sim
    end

    @testset "quasi-static solve, Jacobi preconditioner" begin
        mktempdir() do dir
            sim = run_sim!(build_sim(dir, qs_yaml("      type: jacobi"),
                                     "mf_jacobi.yaml"; matrix_free=true))

            @test sim.integrator.nonlinear_solver.linear_solver.assembled == false

            avg = average_components(sim)
            mx  = maximum_components(sim)
            @test avg[3] ≈  5.00e-4 rtol=1e-4   # avg u_z (analytical)
            @test avg[1] ≈ -1.25e-4 rtol=1e-2   # avg u_x (Poisson)
            @test avg[2] ≈ -1.25e-4 rtol=1e-2   # avg u_y (Poisson)
            @test mx[3]  ≈  1.00e-3 rtol=1e-6   # max u_z = prescribed BC (exact)
        end
    end

    @testset "quasi-static solve, Chebyshev preconditioner" begin
        # Drives _update_chebyshev_precond_qs! and _chebyshev_precond_op, which
        # estimate lambda_max by a power method on the matrix-free operator
        # rather than reading it off an assembled matrix.
        mktempdir() do dir
            sim = run_sim!(build_sim(dir,
                                     qs_yaml("      type: chebyshev\n      degree: 5"),
                                     "mf_cheb.yaml"; matrix_free=true))

            precond = sim.integrator.nonlinear_solver.linear_solver.precond
            @test precond isa Carina.ChebyshevPreconditioner
            # The power-method estimate must have actually run and be positive;
            # K is SPD, so lambda_max > 0.
            @test precond.lambda_max[] > 0.0

            avg = average_components(sim)
            mx  = maximum_components(sim)
            @test avg[3] ≈  5.00e-4 rtol=1e-4
            @test mx[3]  ≈  1.00e-3 rtol=1e-6
        end
    end

    @testset "matrix-free and assembled agree" begin
        # The two paths solve the same discrete problem, so they must land on the
        # same displacement field to solver tolerance -- not merely on the same
        # analytical answer.
        u_mf = mktempdir() do dir
            copy(run_sim!(build_sim(dir, qs_yaml("      type: jacobi"),
                                    "mf.yaml"; matrix_free=true)).params.field.data)
        end
        u_assembled = mktempdir() do dir
            copy(run_sim!(build_sim(dir, qs_yaml("      type: jacobi"),
                                    "assembled.yaml"; matrix_free=false)).params.field.data)
        end
        @test u_mf ≈ u_assembled rtol=1e-8
    end

    @testset "stiffness_action reproduces the assembled stiffness" begin
        # The sharpest check available: FEC.stiffness_action is the kernel the
        # whole GPU path rests on, and it is never touched by an assembled run.
        # Applying it to an arbitrary vector must reproduce K*v exactly, since
        # both are the same bilinear form evaluated at the same state.
        mktempdir() do dir
            sim = run_sim!(build_sim(dir, qs_yaml("      type: jacobi"),
                                     "action.yaml"; matrix_free=true))

            ig  = sim.integrator
            asm = ig.asm
            p   = sim.params
            U   = ig.U

            Carina.FEC.assemble_stiffness!(asm, Carina.FEC.stiffness, U, p)
            K = Carina.FEC.stiffness(asm)
            @test size(K, 1) == length(U)

            # A vector with no particular symmetry, so component mix-ups cannot
            # cancel out.
            v = [sin(0.37 * i) for i in 1:length(U)]
            y = similar(v)
            Carina._stiffness_matvec_qs!(y, v, asm, U, p)

            Kv = K * v
            @test norm(y - Kv) / norm(Kv) < 1e-12

            # And the matrix-free Jacobi preconditioner must be the reciprocal of
            # that same matrix's diagonal.
            precond = Carina.JacobiPreconditioner(similar(v))
            Carina._update_jacobi_precond_qs!(precond, asm, U, p)
            @test precond.inv_diag ≈ 1.0 ./ abs.(diag(K)) rtol=1e-12
        end
    end

    @testset "reduced-precision smoother action" begin
        # `stiffness_action_fp32` is what the GPU AMG V-cycle smooths with
        # (`_use_fp32_smoother`).  It is device-agnostic, so its behavior is
        # testable on CPU where CI can reach it.  Three things must hold, and
        # each has a distinct silent-failure mode behind it.
        mktempdir() do dir
            sim = run_sim!(build_sim(dir, qs_yaml("      type: jacobi"),
                                     "action_fp32.yaml"; matrix_free=true))
            ig = sim.integrator; asm = ig.asm; p = sim.params; U = ig.U

            v = [sin(0.37 * i) for i in 1:length(U)]
            y64 = similar(v); y64b = similar(v); y32 = similar(v)
            Carina._stiffness_matvec_qs!(y64, v, asm, U, p)
            Carina._stiffness_matvec_qs!(y64b, v, asm, U, p)
            Carina._stiffness_matvec_qs_fp32!(y32, v, asm, U, p)

            # Nothing here may compare two assembled vectors for bit-identity.
            # The threaded element loop accumulates into shared nodes through
            # atomics, so the summation order differs between two calls of the
            # *same* kernel -- `y64` and `y64b` land ~1 ulp apart at more than
            # one thread.  That noise floor is the yardstick both assertions
            # below are measured against.
            noise = norm(y64b - y64) / norm(y64)

            # (1) It is a faithful approximation of the same operator.
            @test norm(y32 - y64) / norm(y64) < 1e-5

            # (2) It is not the *same* operator.  A silent fallback to Float64
            # would leave the run correct and the speedup at zero, with nothing
            # to notice.  `y32 != y64` cannot detect that: above one thread the
            # reduction noise alone makes it true.  A genuine Float32 action is
            # ~1e-7 relative, nine orders above the ~1e-16 floor, so require the
            # difference to be unambiguously larger than reduction noise.
            @test norm(y32 - y64) / norm(y64) > max(1e3 * noise, 1e-10)

            # (3) The model honors the requested precision.  Without this the
            # first two still pass while the arithmetic runs in Float64.
            fspace = Carina.FEC.function_space(asm.dof)
            Carina.FEC.foreach_block(fspace, p) do physics, ref_fe, b
                props_el = Carina.FEC.properties(p.properties, 1, b)
                @test Carina._fp32_action_is_effective(physics, props_el)
                return nothing
            end
        end

        # LinearElastic has its own small-strain `stiffness_action` that skips
        # the geometric push-forward.  The NS = 0 reduced-precision method would
        # shadow it and quietly change the physics, so it must dispatch to the
        # exact kernel -- the same arithmetic, not merely a close answer.
        mktempdir() do dir
            le_yaml = replace(qs_yaml("      type: jacobi"),
                              "cube: neohookean" => "cube: linear elastic",
                              "    neohookean:"  => "    linear elastic:")
            sim = run_sim!(build_sim(dir, le_yaml, "action_le.yaml";
                                     matrix_free=true))
            ig = sim.integrator; asm = ig.asm; p = sim.params; U = ig.U

            v = [cos(0.29 * i) for i in 1:length(U)]
            y64 = similar(v); y64b = similar(v); y32 = similar(v)
            Carina._stiffness_matvec_qs!(y64, v, asm, U, p)
            Carina._stiffness_matvec_qs!(y64b, v, asm, U, p)
            Carina._stiffness_matvec_qs_fp32!(y32, v, asm, U, p)

            # `y32 == y64` is the natural statement of "same kernel" and it is
            # NOT usable: atomic accumulation makes even `y64` vs `y64b` differ
            # by ~1 ulp above one thread.  Pin the fp32 twin to that same floor
            # instead -- it must be indistinguishable from re-running the exact
            # kernel, and nowhere near the ~1e-7 of a real Float32 action.
            noise = norm(y64b - y64) / norm(y64)
            @test norm(y32 - y64) / norm(y64) <= max(10 * noise, 1e-13)
        end
    end

    @testset "Newmark effective-stiffness operator" begin
        # The dynamic path applies K + c_M*M matrix-free through
        # _eff_stiffness_matvec! / _apply_eff_stiffness!, a different operator
        # from the quasi-static one and separately untested.
        dyn_dir = joinpath(@__DIR__, "..", "examples", "mechanics",
                           "implicit-dynamic", "cube")
        dyn_yaml = """
type: single

input mesh file: cube.g
output mesh file: cube_mf_dyn.e

model:
  type: solid mechanics
  material:
    blocks:
      cube: neohookean
    neohookean:
      elastic modulus: 1.0e10
      Poisson's ratio: 0.25
      density: 1000.0

time integrator:
  type: newmark
  initial time: 0.0
  final time: 0.02
  time step: 0.01
  beta: 0.25
  gamma: 0.5

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
      - maximum iterations: 16
    converge when any:
      - absolute residual: 1.0e-6
      - relative residual: 1.0e-10
  linear solver:
    type: iterative
    tolerance: 1.0e-10
    maximum iterations: 500
    preconditioner:
      type: jacobi
"""
        # The assembled dynamic path is the one already in use, and it works.
        mktempdir() do dir
            cp_example(joinpath(dyn_dir, "cube.g"), joinpath(dir, "cube.g"))
            path = joinpath(dir, "asm_newmark.yaml")
            open(io -> write(io, dyn_yaml), path, "w")
            dict = Carina.YAML.load_file(path; dicttype=Dict{String,Any})
            asm = Carina.create_simulation(dict, dir)
            run_sim!(asm)
            @test all(isfinite, asm.params.field.data)
            @test maximum(asm.params.field.data) > 0.0
        end

        # The effective operator is applied by `NewmarkAction`, which fuses
        # K·v and c_M·M·v into one element pass.  The reference is the same
        # operator as two full assemblies combined on the outside — the form
        # the solver used before the fusion.  Summation order differs between
        # the two (per-qp combine vs per-pass combine), so the comparison is
        # a tight ≈, not equality.
        mktempdir() do dir
            cp_example(joinpath(dyn_dir, "cube.g"), joinpath(dir, "cube.g"))
            path = joinpath(dir, "mf_newmark.yaml")
            open(io -> write(io, dyn_yaml), path, "w")
            dict = Carina.YAML.load_file(path; dicttype=Dict{String,Any})

            mf = Carina.create_simulation(dict, dir)
            ls = mf.integrator.nonlinear_solver.linear_solver
            ls.assembled = false
            ig = mf.integrator

            Uu = Carina._displacement(ig)
            v  = [sin(0.37 * i) for i in 1:length(Uu)]
            y  = similar(v)
            Carina._eff_stiffness_matvec!(y, v, ig.asm, Uu, ig.c_M, mf.params)
            @test all(isfinite, y)
            @test norm(y) > 0.0

            FEC = Carina.FEC
            FEC.assemble_matrix_free_action!(ig.asm, FEC.stiffness_action,
                                             Uu, v, mf.params)
            Kv = copy(FEC.hvp(ig.asm, v))
            FEC.assemble_matrix_free_action!(ig.asm, FEC.mass_action,
                                             Uu, v, mf.params)
            Mv = FEC.hvp(ig.asm, v)
            @test isapprox(y, Kv .+ ig.c_M .* Mv; rtol = 1e-12)

            run_sim!(mf)
            @test all(isfinite, mf.params.field.data)
            @test maximum(mf.params.field.data) > 0.0
        end
    end

    @testset "two-phase assembly matches the fused single-pass kernel" begin
        # `_assemble_action_two_phase!` computes the same operator as FEC's
        # one-thread-per-element kernel through per-(element,qp) staging plus
        # a node-parallel gather over the inverse adjacency.  Same per-qp
        # functor, different summation order — tight ≈, not equality.  Run
        # a step first so the geometric stiffness is nonzero; at U = 0 the
        # comparison could not see an error in that term.
        dyn_dir = joinpath(@__DIR__, "..", "examples", "mechanics",
                           "implicit-dynamic", "cube")
        tp_yaml = """
type: single
input mesh file: cube.g
output mesh file: cube_tp_dyn.e
model:
  type: solid mechanics
  material:
    blocks:
      cube: neohookean
    neohookean:
      elastic modulus: 1.0e10
      Poisson's ratio: 0.25
      density: 1000.0
time integrator:
  type: newmark
  initial time: 0.0
  final time: 0.02
  time step: 0.01
  beta: 0.25
  gamma: 0.5
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
      - maximum iterations: 16
    converge when any:
      - absolute residual: 1.0e-6
      - relative residual: 1.0e-10
  linear solver:
    type: iterative
    tolerance: 1.0e-10
    maximum iterations: 500
    preconditioner:
      type: jacobi
"""
        mktempdir() do dir
            cp_example(joinpath(dyn_dir, "cube.g"), joinpath(dir, "cube.g"))
            path = joinpath(dir, "tp_newmark.yaml")
            open(io -> write(io, tp_yaml), path, "w")
            dict = Carina.YAML.load_file(path; dicttype=Dict{String,Any})

            mf = Carina.create_simulation(dict, dir)
            mf.integrator.nonlinear_solver.linear_solver.assembled = false
            run_sim!(mf)
            ig = mf.integrator

            # The adjacency was built by _init_assembly_cache! at setup.
            @test Carina._two_phase_host[] !== nothing

            Uu = Carina._displacement(ig)
            v  = [sin(0.37 * i) for i in 1:length(Uu)]
            FEC = Carina.FEC

            for action in (Carina.NewmarkAction(ig.c_M), FEC.stiffness_action)
                FEC.assemble_matrix_free_action!(ig.asm, action, Uu, v,
                                                 mf.params)
                ref = copy(ig.asm.stiffness_action_storage.data)
                Carina._assemble_action_two_phase!(ig.asm, action, Uu, v,
                                                   mf.params)
                two = ig.asm.stiffness_action_storage.data
                @test isapprox(two, ref; rtol = 1e-12)
                @test norm(ref) > 0.0
            end
        end
    end

    @testset "diagonal-only kernels match the full-matrix diagonals" begin
        # `StiffnessDiagonal`/`NewmarkDiagonal` compute diag(K) and
        # diag(K + c_M·M) without forming the 24×24 element matrix.  The
        # reference is the full-matrix route through `FEC.stiffness` and
        # `FEC.mass`.  The comparison runs at the post-solve displacement so
        # the geometric part of the tangent is nonzero — at U = 0 the two
        # routes could agree while disagreeing on every geometric term.
        dyn_dir = joinpath(@__DIR__, "..", "examples", "mechanics",
                           "implicit-dynamic", "cube")
        diag_yaml = """
type: single

input mesh file: cube.g
output mesh file: cube_diag_dyn.e

model:
  type: solid mechanics
  material:
    blocks:
      cube: neohookean
    neohookean:
      elastic modulus: 1.0e10
      Poisson's ratio: 0.25
      density: 1000.0

time integrator:
  type: newmark
  initial time: 0.0
  final time: 0.02
  time step: 0.01
  beta: 0.25
  gamma: 0.5

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
      - maximum iterations: 16
    converge when any:
      - absolute residual: 1.0e-6
      - relative residual: 1.0e-10
  linear solver:
    type: iterative
    tolerance: 1.0e-10
    maximum iterations: 500
    preconditioner:
      type: jacobi
"""
        mktempdir() do dir
            cp_example(joinpath(dyn_dir, "cube.g"), joinpath(dir, "cube.g"))
            path = joinpath(dir, "diag_newmark.yaml")
            open(io -> write(io, diag_yaml), path, "w")
            dict = Carina.YAML.load_file(path; dicttype=Dict{String,Any})
            mf = Carina.create_simulation(dict, dir)
            mf.integrator.nonlinear_solver.linear_solver.assembled = false
            run_sim!(mf)

            ig = mf.integrator
            Uu = Carina._displacement(ig)
            @test maximum(abs, Uu) > 0.0
            FEC = Carina.FEC

            FEC.assemble_diagonal!(ig.asm, FEC.stiffness, Uu, mf.params)
            d_k = copy(FEC.diagonal(ig.asm))
            FEC.assemble_diagonal!(ig.asm, FEC.mass, Uu, mf.params)
            d_m = copy(FEC.diagonal(ig.asm))

            FEC.assemble_diagonal!(ig.asm, Carina.StiffnessDiagonal(), Uu, mf.params)
            @test isapprox(FEC.diagonal(ig.asm), d_k; rtol = 1e-13)

            FEC.assemble_diagonal!(ig.asm, Carina.NewmarkDiagonal(ig.c_M), Uu, mf.params)
            @test isapprox(FEC.diagonal(ig.asm), d_k .+ ig.c_M .* d_m; rtol = 1e-13)
        end

        # LinearElastic takes its tangent at ∇u = 0; the diagonal kernel must
        # mirror that specialization, not the finite-deformation one.
        mktempdir() do dir
            le_yaml = replace(qs_yaml("      type: jacobi"),
                              "cube: neohookean" => "cube: linear elastic",
                              "    neohookean:"  => "    linear elastic:")
            sim = run_sim!(build_sim(dir, le_yaml, "diag_le.yaml";
                                     matrix_free=true))
            ig = sim.integrator
            U = ig.U
            FEC = Carina.FEC

            FEC.assemble_diagonal!(ig.asm, FEC.stiffness, U, sim.params)
            d_k = copy(FEC.diagonal(ig.asm))
            FEC.assemble_diagonal!(ig.asm, Carina.StiffnessDiagonal(), U, sim.params)
            @test isapprox(FEC.diagonal(ig.asm), d_k; rtol = 1e-13)
        end
    end

    @testset "matrix-free Newmark matches assembled" begin
        # End-to-end check of the dynamic path that was previously unreachable:
        # the effective operator K + c_M*M applied matrix-free must drive the
        # solve to the same field the assembled K_eff does.
        dyn_dir = joinpath(@__DIR__, "..", "examples", "mechanics",
                           "implicit-dynamic", "cube")
        dyn_yaml_for(precond) = """
type: single

input mesh file: cube.g
output mesh file: cube_mf_dyn.e

model:
  type: solid mechanics
  material:
    blocks:
      cube: neohookean
    neohookean:
      elastic modulus: 1.0e10
      Poisson's ratio: 0.25
      density: 1000.0

time integrator:
  type: newmark
  initial time: 0.0
  final time: 0.02
  time step: 0.01
  beta: 0.25
  gamma: 0.5

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
      - maximum iterations: 16
    converge when any:
      - absolute residual: 1.0e-6
      - relative residual: 1.0e-10
  linear solver:
    type: iterative
    tolerance: 1.0e-10
    maximum iterations: 500
    preconditioner:
$(precond)
"""
        function dyn_run(precond, matrix_free)
            mktempdir() do dir
                cp_example(joinpath(dyn_dir, "cube.g"), joinpath(dir, "cube.g"))
                path = joinpath(dir, "dyn.yaml")
                open(io -> write(io, dyn_yaml_for(precond)), path, "w")
                dict = Carina.YAML.load_file(path; dicttype=Dict{String,Any})
                sim = Carina.create_simulation(dict, dir)
                matrix_free && (sim.integrator.nonlinear_solver.linear_solver.assembled = false)
                run_sim!(sim)
                return copy(sim.params.field.data)
            end
        end

        u_assembled = dyn_run("      type: jacobi", false)
        @test dyn_run("      type: jacobi",    true) ≈ u_assembled rtol=1e-8
        # Chebyshev drives the same operator through the power-method estimate.
        @test dyn_run("      type: chebyshev\n      degree: 5", true) ≈ u_assembled rtol=1e-8
    end

    @testset "analytic NeoHookean JVP matches the dual pass" begin
        # `_pk1_jvp` has a closed-form specialization for
        # Hyperelastic{NeoHookean} (register pressure: the dual pass carries a
        # partial through every intermediate, and the GPU kernel sits at the
        # 255-register cap because of it).  The specialization must be the
        # SAME derivative the generic dual pass computes, at finite strain,
        # and must preserve reduced precision.
        model = Carina.CM.Hyperelastic(Carina.CM.NeoHookean())
        props = [1000.0, 1.0e9, 4.0e8]   # density, kappa, mu
        struct_tag = Carina._PK1JVPTag
        dual_jvp = (∇u, ∇v) -> begin
            T = eltype(∇u)
            D = Carina.ForwardDiff.Dual{struct_tag, T, 1}
            ∇u_d = Carina.Tensor{2, 3, D, 9}(ntuple(
                i -> D(∇u.data[i],
                       Carina.ForwardDiff.Partials{1, T}((∇v.data[i],))),
                Val(9)))
            P_d = Carina.CM.pk1_stress(model, T.(props), nothing, nothing,
                                       zero(T), ∇u_d, zero(D))
            Carina.Tensor{2, 3, T, 9}(ntuple(
                i -> Carina.ForwardDiff.partials(P_d.data[i], 1), Val(9)))
        end
        # Deterministic dense strain states: sign-varying, no symmetry, and
        # scaled so J > 0 while the geometric term is far from zero (the trap
        # in the ablation campaign: at U = 0 that term vanishes identically).
        fill9 = (trial, s, scale) -> Carina.Tensor{2, 3, Float64, 9}(
            ntuple(i -> scale * sin(1.7 * i + 0.31 * trial + s), Val(9)))
        for trial in 1:100
            ∇u = fill9(trial, 0.0, 0.25)
            ∇v = fill9(trial, 2.1, 1.0)
            @assert Carina.Tensors.det(∇u + one(∇u)) > 0
            ref = dual_jvp(∇u, ∇v)
            ana = Carina._pk1_jvp(model, props, nothing, nothing, 0.0, ∇u, ∇v)
            @test maximum(abs, (ana - ref).data) <=
                  1e-13 * maximum(abs, ref.data)
        end
        f32 = (s) -> Carina.Tensor{2, 3, Float32, 9}(
            ntuple(i -> 0.1f0 * sin(2.3f0 * i + s), Val(9)))
        @test eltype(Carina._pk1_jvp(model, Float32.(props), nothing, nothing,
                                     0.0f0, f32(0.0f0), f32(1.5f0))) === Float32
    end
end
