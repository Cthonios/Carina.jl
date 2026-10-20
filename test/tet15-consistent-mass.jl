# The consistent mass of the fifteen-node tetrahedron (TETRA15).
#
# The integrand N_a N_b of the consistent mass is of degree 8 on a
# straight-sided element.  The 14-point degree-5 rule of the element gives a
# scalar element mass of rank 14 for 15 functions, and the assembled mass then
# has a zero eigenvalue.  The projected element (src/projected_physics.jl)
# integrates the TETRA15 mass with the degree-9 conical Gauss-Jacobi rule
# (125 points) instead, which is exact there.  The checks:
#   1. the element mass is nonsingular and equals exact integration (a rule of
#      degree 15) to rounding, on the reference element and on a distorted
#      straight-sided element;
#   2. its total is ρV and its row sums are the lumped mass, which stays on
#      the element's rule;
#   3. its action and its diagonal are those of the matrix;
#   4. the assembled mass of the TETRA15 cube has no zero eigenvalue (and the
#      degree-5 rule leaves one on the same mesh);
#   5. the mass of the other elements is the one of their own rule, unchanged.

using LinearAlgebra: Symmetric, eigvals, cond, det, norm
using StaticArrays: SVector, SMatrix

@testset "TETRA15 consistent mass" begin
    FEC = Carina.FEC
    RFE = Carina.RFE
    el15 = RFE.Tet{RFE.EnrichedLagrange, 2}()
    ref15 = RFE.ReferenceFE(el15, RFE.GaussLegendre(5))
    @test RFE.num_cell_quadrature_points(ref15) == 14
    @test Carina._num_mass_points(ref15) == 125

    cm = Carina.CM.Hyperelastic(Carina.CM.NeoHookean())
    physics = Carina.ProjectedSolidMechanics(cm, 1)
    states = FEC.ElementState(nothing, nothing, 1, 1)
    ρ = 2.5
    props = SVector(ρ)

    # Reference node coordinates (3 × 15) and an affine image of them.
    Ξ = reduce(hcat, [Float64.(collect(x)) for x in RFE._nodes(el15)])
    A = [1.3 0.4 -0.2; 0.1 0.9 0.3; -0.3 0.2 1.6]
    b = [0.5, -1.0, 2.0]
    @test det(A) > 0

    flat(X) = SVector{length(X), Float64}(vec(X))
    element_mass(ref_fe, X) = FEC.mass(physics, ref_fe, flat(X), 0.0, 0.0,
                                       zero(flat(X)), zero(flat(X)), states, props)
    scalar_block(M) = Matrix(M[1:3:end, 1:3:end])

    # Exact integration: the scalar mass ρ |det A| ∫ N_a N_b dξ with the
    # degree-15 conical rule, and the same with the element's 14-point rule.
    function reference_mass(deg, detA)
        ξs, ws = RFE.cell_quadrature_points_and_weights(el15, RFE.GaussLegendre(deg))
        M = zeros(15, 15)
        for q in eachindex(ws)
            N = RFE.shape_function_value(el15, ξs[:, q])
            M .+= (ρ * detA * ws[q]) .* (N * N')
        end
        return M
    end

    @testset "element mass, $label" for (label, X, detA) in
            (("reference element", Ξ, 1.0), ("affine element", A * Ξ .+ b, det(A)))
        M = element_mass(ref15, X)
        Ms = scalar_block(M)
        # interleaved components, no coupling between them
        for i in 1:3, j in 1:3
            blk = Matrix(M[i:3:end, j:3:end])
            @test i == j ? blk == Ms : all(iszero, blk)
        end
        Mex = reference_mass(15, detA)
        err = maximum(abs.(Ms - Mex)) / maximum(abs.(Mex))
        @test err < 1e-14
        λ = eigvals(Symmetric(Ms))
        @test count(<(1e-12 * λ[end]), λ) == 0
        @test λ[1] > 0
        # on the reference element the condition number is that of the
        # nodal basis, 93.6
        label == "reference element" && @test 93.5 < cond(Ms) < 93.7
        @info "TETRA15 element mass, $label: max entry error $(err), " *
              "condition number $(cond(Ms)), smallest eigenvalue $(λ[1])"
        # the 14-point rule of the element: one zero eigenvalue
        λ5 = eigvals(Symmetric(reference_mass(5, detA)))
        @test count(<(1e-12 * λ5[end]), λ5) == 1

        # total mass ρV and row sums equal to the lumped mass
        V = detA / 6
        @test sum(Ms) ≈ ρ * V rtol = 1e-14
        m_l = sum(FEC.lumped_mass(Carina._inner(physics), FEC._cell_interpolants(ref15, q),
                                  flat(X), 0.0, 0.0, zero(flat(X)), zero(flat(X)),
                                  nothing, nothing, props)
                  for q in 1:RFE.num_cell_quadrature_points(ref15))
        @test maximum(abs.(vec(sum(M, dims = 2)) .- m_l)) < 1e-14 * maximum(m_l)

        # the action and the diagonal are those of the matrix
        v = SVector{45, Float64}(ntuple(i -> cos(0.37 * i), 45))
        Mv = Carina._mass_action_element(physics, ref15, flat(X), v, props)
        @test norm(Mv - M * v) < 1e-14 * norm(M * v)
        d = Carina._mass_diagonal_element(ref15, flat(X), ρ, zero(SVector{45, Float64}))
        @test maximum(abs.(d .- [M[k, k] for k in 1:45])) < 1e-15 * maximum(abs, M)
    end

    @testset "assembled mass of the TETRA15 cube" begin
        deck = """
type: single
input mesh file: cube.g
output mesh file: mass.e
model:
  type: solid mechanics
  volumetric projection: linear
  material:
    blocks:
      cube: neohookean
    neohookean:
      elastic modulus: 1.0e9
      Poisson's ratio: 0.3
      density: 1000.0
time integrator:
  type: quasi static
  initial time: 0.0
  final time: 1.0
  time step: 0.5
solver:
  type: newton
  linear solver:
    type: direct
  termination:
    fail when any:
      - maximum iterations: 4
    converge when any:
      - absolute residual: 1.0e-9
"""
        mesh = joinpath(@__DIR__, "..", "examples", "meshes", "cube-tet15", "cube.g")
        mktempdir() do dir
            cp_example(mesh, joinpath(dir, "cube.g"))
            open(io -> write(io, deck), joinpath(dir, "mass.yaml"), "w")
            dict = Carina.YAML.load_file(joinpath(dir, "mass.yaml"); dicttype = Dict{String, Any})
            sim = Carina.create_simulation(dict, dir)
            asm = sim.integrator.asm
            p = sim.params
            U = copy(sim.integrator.U)
            @test length(U) == 3 * 297      # no Dirichlet condition: every dof free
            FEC.assemble_mass!(asm, FEC.mass, U, p)
            M = Matrix(FEC.mass(asm))
            λ = eigvals(Symmetric(M))
            nzero = count(<(1e-12 * λ[end]), λ)
            @test nzero == 0
            @test λ[1] > 1e-3 * λ[end]
            @info "Assembled TETRA15 mass of the cube ($(size(M, 1)) dofs): zero eigenvalues " *
                  "$nzero, smallest $(λ[1]), largest $(λ[end])"

            # total mass and row sums
            m = FEC.UnstructuredMesh(joinpath(dir, "cube.g"))
            X = m.nodal_coords
            c = m.element_conns["cube"]
            V = sum(axes(c, 2)) do e
                x1 = X[:, c[1, e]]
                abs(det(hcat(X[:, c[2, e]] - x1, X[:, c[3, e]] - x1, X[:, c[4, e]] - x1))) / 6
            end
            @test sum(M) ≈ 3 * 1000.0 * V rtol = 1e-13
            FEC.assemble_lumped_mass!(asm, FEC.lumped_mass, U, p)
            m_l = FEC.lumped_mass(asm)
            @test maximum(abs.(vec(sum(M, dims = 2)) .- m_l)) < 1e-13 * maximum(m_l)

            # the same mesh under the element's 14-point rule: one zero
            # eigenvalue of the scalar mass
            n = size(X, 2)
            M5 = zeros(n, n)
            for e in axes(c, 2)
                Xe = SMatrix{3, 15, Float64}(reduce(hcat, [X[:, c[a, e]] for a in 1:15]))
                for q in 1:RFE.num_cell_quadrature_points(ref15)
                    cell = FEC.map_interpolants(FEC._cell_interpolants(ref15, q), flat(Xe))
                    M5[c[:, e], c[:, e]] .+= cell.JxW .* (cell.N * cell.N')
                end
            end
            λ5 = eigvals(Symmetric(M5))
            @test count(<(1e-12 * λ5[end]), λ5) == 1
        end
    end

    # Elements other than TETRA15 keep the mass of their own rule: the sum over
    # the element's quadrature points of ρ JxW N Nᵀ, bit for bit.
    # The first ten TETRA15 nodes are those of TETRA10, the first four its
    # vertices.
    hex8 = Float64[-1 1 1 -1 -1 1 1 -1; -1 -1 1 1 -1 -1 1 1; -1 -1 -1 -1 1 1 1 1]
    @testset "other elements: $(typeof(el)), $(typeof(rule))" for (el, rule, Ξe) in (
            (RFE.Tet{RFE.Lagrange, 1}(), RFE.GaussLegendre(2), Ξ[:, 1:4]),
            (RFE.Tet{RFE.Lagrange, 2}(), RFE.GaussLegendre(2), Ξ[:, 1:10]),
            (RFE.Tet{RFE.Lagrange, 2}(), RFE.GaussLegendre(5), Ξ[:, 1:10]),
            (RFE.Hex{RFE.Lagrange, 1}(), RFE.GaussLegendre(2), hex8))
        ref_fe = RFE.ReferenceFE(el, rule)
        NN = RFE.num_cell_dofs(el)
        @test Carina._num_mass_points(ref_fe) == RFE.num_cell_quadrature_points(ref_fe)
        X = A * Ξe .+ b
        M = element_mass(ref_fe, X)
        cell = FEC.map_interpolants(FEC._cell_interpolants(ref_fe, 1), flat(X))
        Ms = (cell.JxW * ρ) * (cell.N * cell.N')
        for q in 2:RFE.num_cell_quadrature_points(ref_fe)
            cell = FEC.map_interpolants(FEC._cell_interpolants(ref_fe, q), flat(X))
            Ms = Ms + (cell.JxW * ρ) * (cell.N * cell.N')
        end
        @test M == Carina._expand_mass(Ms)
        @test size(M) == (3 * NN, 3 * NN)
    end
end
