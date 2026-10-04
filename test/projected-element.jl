# The projected volumetric formulation (model.volumetric projection): the
# volumetric strain θ(J) is replaced element by element by its L² projection
# onto polynomials of degree 0 or 1, and the kernels are assembled by element
# (src/projected_physics.jl).
#
# The checks are the ones a new set of kernels must pass before any physics
# is measured with them:
#   1. the residual is the derivative of the assembled energy;
#   2. the assembled tangent is the derivative of the residual;
#   3. the matrix-free action, the diagonal kernels and the Newmark diagonal
#      reproduce the assembled matrices;
#   4. an affine displacement field is reproduced exactly (a constant J is its
#      own projection, so the projected element must agree with the pointwise
#      one there);
#   5. a material without the split is accepted, in the general form
#      (test/projected-general.jl).
# All run on the TETRA15 cube with the J2 model in its elastic range, at a
# converged nonlinear state, so that the geometric and coupling terms of the
# tangent are nonzero.

using LinearAlgebra: norm, diag

@testset "Projected volumetric formulation" begin
    FEC = Carina.FEC
    mesh_dir(v) = joinpath(@__DIR__, "..", "examples", "meshes", "cube-$v")

    # Confined compression of the cube with a J2 material below yield.
    model_block(projection) = "model:\n  type: solid mechanics\n" *
        (projection === nothing ? "" : "  volumetric projection: $projection\n") *
        "  material:"
    deck(projection; mesh = "cube.g", integrator = "quasi static", extra = "") = """
type: single
input mesh file: $mesh
output mesh file: projected.e
$(model_block(projection))
    blocks:
      cube: j2 plasticity
    j2 plasticity:
      elastic modulus: 1.0e9
      Poisson's ratio: 0.3
      density: 1000.0
      yield stress: 1.0e12
      hardening modulus: 0.0
time integrator:
  type: $integrator
  initial time: 0.0
  final time: 1.0
  time step: 0.5
$extra
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
      function: "-2.0e-2 * t"
    - side set: ssx+
      component: x
      function: "1.0e-2 * t * z"
solver:
  type: newton
  linear solver:
    type: direct
  termination:
    fail when any:
      - maximum iterations: 32
    converge when any:
      - absolute residual: 1.0e-9
      - relative residual: 1.0e-12
"""

    function build(dir, yaml_text)
        cp_example(joinpath(mesh_dir("tet15"), "cube.g"), joinpath(dir, "cube.g"))
        path = joinpath(dir, "projected.yaml")
        open(io -> write(io, yaml_text), path, "w")
        dict = Carina.YAML.load_file(path; dicttype=Dict{String,Any})
        return Carina.create_simulation(dict, dir)
    end

    function run!(sim)
        Carina.evolve!(sim)
        FEC.close(sim.post_processor)
        return sim
    end

    energy(asm, U, p) = (FEC.assemble_scalar!(asm, FEC.energy, U, p);
                         sum(asm.scalar_quadrature_storage))
    function residual(asm, U, p)
        FEC.assemble_vector!(asm, FEC.residual, U, p)
        return copy(FEC.residual(asm))
    end

    @testset "the physics is the projected one" begin
        mktempdir() do dir
            sim = build(dir, deck("linear"))
            ph = first(values(sim.params_cpu.physics))
            @test ph isa Carina.ProjectedSolidMechanics
            @test Carina.projection_degree(ph) == 1
            @test FEC.assembly_granularity(ph) == FEC.ByElement()
        end
        mktempdir() do dir
            sim0 = build(dir, deck("constant"))
            @test Carina.projection_degree(first(values(sim0.params_cpu.physics))) == 0
        end
    end

    @testset "energy, residual and tangent are consistent" begin
        for projection in ("constant", "linear")
            mktempdir() do dir
                sim = run!(build(dir, deck(projection)))
                ig  = sim.integrator
                asm = ig.asm
                p   = sim.params
                U   = copy(ig.U)
                @test maximum(abs, U) > 1e-3

                # 1. residual = dW/dU, central differences on a sample of dofs
                R = residual(asm, U, p)
                @test maximum(abs, R) < 1e-6 * 1e9   # converged state
                # move off equilibrium so that the residual is not zero
                U .+= 2.0e-4 .* sin.(0.7 .* (1:length(U)))
                R = residual(asm, U, p)
                @test maximum(abs, R) > 0
                h = 1.0e-6
                worst = 0.0
                for k in (1, 7, 40, 101, 233, length(U) - 3)
                    Up = copy(U); Up[k] += h
                    Um = copy(U); Um[k] -= h
                    dW = (energy(asm, Up, p) - energy(asm, Um, p)) / (2h)
                    worst = max(worst, abs(dW - R[k]) / max(abs(R[k]), 1e-3 * maximum(abs, R)))
                end
                @test worst < 1e-6

                # 2. tangent = dR/dU along a few directions
                FEC.assemble_stiffness!(asm, FEC.stiffness, U, p)
                K = copy(FEC.stiffness(asm))
                @test K ≈ K' rtol = 1e-10        # the energy is a potential
                worst = 0.0
                for seed in (0.31, 0.77)
                    v  = [sin(seed * i) for i in 1:length(U)]
                    dR = (residual(asm, U .+ h .* v, p) .- residual(asm, U .- h .* v, p)) ./ (2h)
                    Kv = K * v
                    worst = max(worst, norm(dR - Kv) / norm(Kv))
                end
                @test worst < 1e-6

                # 3. matrix-free action and diagonal kernels
                v = [cos(0.37 * i) for i in 1:length(U)]
                Kv = K * v
                y = similar(v)
                Carina._stiffness_matvec_qs!(y, v, asm, U, p)
                @test norm(y - Kv) / norm(Kv) < 1e-12

                FEC.assemble_diagonal!(asm, Carina.StiffnessDiagonal(), U, p)
                @test isapprox(FEC.diagonal(asm), diag(K); rtol = 1e-12)

                # nodal 3×3 blocks, column by column.  K is indexed by free
                # dof; entry k of the kernel's vector is K[(n,i), (n,col)] for
                # the node n and component i of free dof k, and is compared
                # where the column dof is free as well.
                free = asm.dof.unknown_dofs
                pos  = Dict(d => k for (k, d) in enumerate(free))
                for col in 1:3
                    FEC.assemble_diagonal!(asm, Carina.StiffnessBlockColumn{col}(), U, p)
                    B = copy(FEC.diagonal(asm))
                    checked = 0
                    worst = 0.0
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
                d_m = diag(FEC.mass(asm))
                c_M = 3.7
                FEC.assemble_diagonal!(asm, Carina.NewmarkDiagonal(c_M), U, p)
                @test isapprox(FEC.diagonal(asm), diag(K) .+ c_M .* d_m; rtol = 1e-12)

                # the Newmark action against the assembled operator
                Carina._assemble_action!(asm, Carina.NewmarkAction(c_M), U, v, p)
                y = copy(asm.stiffness_action_storage[asm.dof.unknown_dofs])
                @test norm(y - (Kv .+ c_M .* (FEC.mass(asm) * v))) / norm(Kv) < 1e-12
            end
        end
    end

    @testset "linear patch test" begin
        # Prescribe an affine field on every face: J is constant per element,
        # equal to its projection, so the projected element reproduces the
        # field exactly, as the pointwise one does.
        exact(x, y, z) = (1.0e-3 * (x + 2y + 3z),
                          1.0e-3 * (4x - y + 0.5z),
                          1.0e-3 * (0.25x + 1.5y + 2z))
        funcs = ("1.0e-3 * (x + 2.0*y + 3.0*z)",
                 "1.0e-3 * (4.0*x - y + 0.5*z)",
                 "1.0e-3 * (0.25*x + 1.5*y + 2.0*z)")
        bcs = join(("    - side set: $s\n      component: $c\n      function: \"$f\"\n"
                    for (c, f) in zip(("x", "y", "z"), funcs)
                    for s in ("ssx-", "ssx+", "ssy-", "ssy+", "ssz-", "ssz+")), "")
        # one step: the boundary data do not depend on t
        patch = replace(deck("linear"),
            r"boundary conditions:\n  dirichlet:\n(    - .*\n|      .*\n)*" =>
            "boundary conditions:\n  dirichlet:\n" * bcs,
            "time step: 0.5" => "time step: 1.0")
        @test occursin("ssx+\n      component: z", patch)
        mktempdir() do dir
            sim = run!(build(dir, patch))
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

    @testset "explicit integration runs with the projected element" begin
        # The row-sum lumped mass is that of the element (positive on TETRA15).
        extra = "  time step: 1.0e-7\n  final time: 3.0e-7"
        text = replace(deck("linear"; integrator = "central difference"),
                       "  time step: 0.5\n" => "", "  final time: 1.0\n" => extra * "\n")
        mktempdir() do dir
            sim = run!(build(dir, text))
            m = adapt(Array, sim.integrator.m_lumped)
            @test all(>(0.0), m)
            @test all(isfinite, adapt(Array, sim.params.field.data))
        end
    end

    @testset "a material without the split is accepted in the general form" begin
        text = replace(deck("linear"), "cube: j2 plasticity" => "cube: neohookean",
                       "    j2 plasticity:" => "    neohookean:")
        mktempdir() do dir
            sim = run!(build(dir, text))
            ph = first(values(sim.params_cpu.physics))
            @test Carina.volumetric_form(ph) isa Carina.GeneralForm
            @test all(isfinite, adapt(Array, sim.params.field.data))
            @test maximum(abs, adapt(Array, sim.params.field.data)) > 1e-3
        end
        # and an unknown projection space
        @test_throws ErrorException Carina._parse_volumetric_projection(
            Dict{String,Any}("model" => Dict{String,Any}("volumetric projection" => "quadratic")))
        @test Carina._parse_volumetric_projection(Dict{String,Any}("model" => Dict{String,Any}())) === nothing
    end
end
