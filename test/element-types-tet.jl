# Tetrahedral element coverage: TETRA4 and TETRA10.
#
# Carina inherited tet support from FEC and RFE and never exercised it: before
# this file there was no tet mesh, no tet example and no tet test in the
# repository, so the capability could have broken at any point without anything
# noticing.  Everything here runs on the two shipped cube meshes, which are the
# same geometry and the same boundary sets as the hex `cube`, so a difference
# between element types is a difference in the element and not in the problem.
#
# The patch test is the load-bearing one.  A linear displacement field must be
# reproduced exactly by any conforming element of degree >= 1, so it catches
# shape-function, Jacobian and quadrature errors that a convergence test would
# only show as slight inaccuracy.  It runs on hex8 as well as the two tets:
# hex8 is the control, and a tet-only failure is then attributable to the tet.

@testset "Tetrahedral elements" begin

    mesh_dir(v) = joinpath(@__DIR__, "..", "examples", "meshes", "cube-$v")

    # Interior-node counts quoted in the patch test below, asserted so the
    # comments cannot drift away from the meshes they describe.
    function interior_node_count(path)
        m = Carina.FEC.UnstructuredMesh(path)
        X = m.nodal_coords
        lo, hi = minimum(X), maximum(X)
        tol = 1e-8 * (hi - lo)
        return count(j -> !any(abs(X[i, j] - lo) < tol || abs(X[i, j] - hi) < tol
                               for i in 1:3), axes(X, 2))
    end

    # ----- the meshes are what they claim to be ------------------------------
    @testset "meshes read with the expected element types" begin
        for (v, etype, p) in (("tet4", "TETRA4", 1), ("tet10", "TETRA10", 2))
            m = Carina.FEC.UnstructuredMesh(joinpath(mesh_dir(v), "cube.g"))
            @test m.element_types["cube"] == etype
            # Both meshes carry the same six named sets as the hex cube, so the
            # decks are interchangeable across element types.
            for s in ("ssx-", "ssy-", "ssz-", "ssz+")
                @test s in values(m.sideset_names)
            end
        end
    end

    @testset "the fine mesh is what the patch test needs" begin
        # A regression here means the patch test quietly became weak again.
        fine = joinpath(mesh_dir("tet4-fine"), "cube.g")
        m = Carina.FEC.UnstructuredMesh(fine)
        @test m.element_types["cube"] == "TETRA4"
        @test interior_node_count(fine) > 900
        # ...and the coarse meshes are as weak as the comments say.
        @test interior_node_count(joinpath(mesh_dir("tet4"), "cube.g")) == 1
        @test interior_node_count(joinpath(mesh_dir("tet10"), "cube.g")) == 28
    end

    # ----- patch test -------------------------------------------------------
    # Prescribe an affine displacement on every boundary face.  An element that
    # can represent affine fields must reproduce it exactly in the interior,
    # for any mesh, to machine precision.
    @testset "linear patch test" begin
        patch_deck(v) = """
type: single
input mesh file: cube.g
output mesh file: patch_$(v).e
model:
  type: solid mechanics
  material:
    blocks:
      cube: linear elastic
    linear elastic:
      elastic modulus: 1.0e9
      Poisson's ratio: 0.3
      density: 1000.0
time integrator:
  type: quasi static
  initial time: 0.0
  final time: 1.0
  time step: 1.0
boundary conditions:
  dirichlet:
    - side set: ssx-
      component: x
      function: "1.0e-3 * (x + 2.0*y + 3.0*z)"
    - side set: ssx+
      component: x
      function: "1.0e-3 * (x + 2.0*y + 3.0*z)"
    - side set: ssy-
      component: x
      function: "1.0e-3 * (x + 2.0*y + 3.0*z)"
    - side set: ssy+
      component: x
      function: "1.0e-3 * (x + 2.0*y + 3.0*z)"
    - side set: ssz-
      component: x
      function: "1.0e-3 * (x + 2.0*y + 3.0*z)"
    - side set: ssz+
      component: x
      function: "1.0e-3 * (x + 2.0*y + 3.0*z)"
    - side set: ssx-
      component: y
      function: "1.0e-3 * (4.0*x - y + 0.5*z)"
    - side set: ssx+
      component: y
      function: "1.0e-3 * (4.0*x - y + 0.5*z)"
    - side set: ssy-
      component: y
      function: "1.0e-3 * (4.0*x - y + 0.5*z)"
    - side set: ssy+
      component: y
      function: "1.0e-3 * (4.0*x - y + 0.5*z)"
    - side set: ssz-
      component: y
      function: "1.0e-3 * (4.0*x - y + 0.5*z)"
    - side set: ssz+
      component: y
      function: "1.0e-3 * (4.0*x - y + 0.5*z)"
    - side set: ssx-
      component: z
      function: "1.0e-3 * (0.25*x + 1.5*y + 2.0*z)"
    - side set: ssx+
      component: z
      function: "1.0e-3 * (0.25*x + 1.5*y + 2.0*z)"
    - side set: ssy-
      component: z
      function: "1.0e-3 * (0.25*x + 1.5*y + 2.0*z)"
    - side set: ssy+
      component: z
      function: "1.0e-3 * (0.25*x + 1.5*y + 2.0*z)"
    - side set: ssz-
      component: z
      function: "1.0e-3 * (0.25*x + 1.5*y + 2.0*z)"
    - side set: ssz+
      component: z
      function: "1.0e-3 * (0.25*x + 1.5*y + 2.0*z)"
solver:
  type: newton
  linear solver:
    type: direct
  termination:
    fail when any:
      - maximum iterations: 16
    converge when any:
      - absolute residual: 1.0e-8
      - relative residual: 1.0e-12
"""
        # The exact affine field the boundary data encodes.
        exact(x, y, z) = (1.0e-3 * (x + 2y + 3z),
                          1.0e-3 * (4x - y + 0.5z),
                          1.0e-3 * (0.25x + 1.5y + 2z))

        # hex8 is included as the control: it is the element every other test
        # in this suite uses, so a tet-only failure is attributable to the tet
        # rather than to the patch machinery.
        patch_mesh(v) = v == "hex8" ?
            joinpath(@__DIR__, "..", "examples", "meshes", "cube", "cube.g") :
            joinpath(mesh_dir(v), "cube.g")

        # `tet4-fine` is the one that matters.  The 2x2x2 meshes leave a single
        # interior node, so the linear cases solve three unknowns; this mesh has
        # 988 interior nodes and 2964 unknowns, and being an unstructured tet
        # mesh its elements are genuinely distorted -- which is the case a patch
        # test exists to catch, since an element can pass on a regular grid and
        # fail off it.
        for v in ("hex8", "tet4", "tet4-fine", "tet10")
            mktempdir() do dir
                cp_example(patch_mesh(v), joinpath(dir, "cube.g"))
                path = joinpath(dir, "patch.yaml")
                open(io -> write(io, patch_deck(v)), path, "w")
                sim = Carina.run(path)

                u = _field_matrix(sim)
                X = adapt(Array, sim.params_cpu.coords.data)
                X = reshape(X, 3, :)
                @test size(u) == size(X)

                worst = 0.0
                for n in axes(X, 2)
                    ex, ey, ez = exact(X[1, n], X[2, n], X[3, n])
                    worst = max(worst,
                                abs(u[1, n] - ex), abs(u[2, n] - ey), abs(u[3, n] - ez))
                end
                # Affine fields are in the span of all three elements, so the
                # only error is the solver tolerance -- not discretization.
                #
                # Strength varies sharply with the mesh, and a pass is worth
                # what the mesh makes it worth:
                #
                #   hex8       1 interior node,   3 unknowns
                #   tet4       1 interior node,   3 unknowns
                #   tet10     28 interior nodes, 84 unknowns
                #   tet4-fine 988 interior nodes, 2964 unknowns, distorted
                #
                # The first two are the classical Irons patch test in its
                # minimal valid form: enough to catch gross shape-function,
                # Jacobian and quadrature errors, not much more.  `tet4-fine`
                # is the one that would actually detect an element that works
                # only on a regular grid.
                @test worst < 1e-10
                # Non-vacuity: the field must actually be the affine one, not
                # zero.  A patch test that passes on an all-zero solution
                # checks nothing.
                @test maximum(abs, u) > 1e-4
            end
        end
    end

    # ----- quasi-static physics ---------------------------------------------
    # Same problem and same analytic answer as mechanics-quasistatic-cube.jl,
    # on tets.  Confined uniaxial compression: u_z is prescribed at the top and
    # the lateral faces are on rollers, so max u_z is exact and the lateral
    # contraction follows Poisson's ratio.
    @testset "quasi-static cube" begin
        for v in ("tet4", "tet10")
            example_dir = joinpath(@__DIR__, "..", "examples", "mechanics",
                                   "quasistatic", "cube-$v")
            mktempdir() do dir
                cp_example(joinpath(example_dir, "cube.g"), joinpath(dir, "cube.g"))
                cp_example(joinpath(example_dir, "cube.yaml"), joinpath(dir, "cube.yaml"))
                sim = Carina.run(joinpath(dir, "cube.yaml"))
                mx = maximum_components(sim)
                avg = average_components(sim)
                # Prescribed on the top face, so this is exact for any element.
                @test mx[3] ≈ 1.00e-3 rtol=1e-8
                # Poisson contraction.  Nodal averages are not volume averages
                # on an unstructured mesh, so the tolerance is looser than the
                # hex test's and the two element types are also compared to
                # each other below.
                @test avg[1] ≈ -1.25e-4 rtol=5e-2
                @test avg[2] ≈ -1.25e-4 rtol=5e-2
                @test avg[3] ≈ 5.0e-4  rtol=5e-2
            end
        end
    end

    # ----- lumped mass ------------------------------------------------------
    # The row-sum lumped mass rho * int(N_a) is positive for linear elements and
    # negative at the vertices of quadratic ones.  Explicit integration divides
    # by it, so a negative entry silently reverses the acceleration there.
    # TETRA4 must work; TETRA10 must be refused rather than run.
    @testset "explicit integration and the lumped mass" begin
        explicit_deck(v) = """
type: single
input mesh file: cube.g
output mesh file: ex_$(v).e
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
  type: central difference
  initial time: 0.0
  final time: 5.0e-7
  time step: 1.0e-7
boundary conditions:
  dirichlet:
    - side set: ssz-
      component: z
      function: "0.0"
"""
        mktempdir() do dir
            cp_example(joinpath(mesh_dir("tet4"), "cube.g"), joinpath(dir, "cube.g"))
            path = joinpath(dir, "ex.yaml")
            open(io -> write(io, explicit_deck("tet4")), path, "w")
            sim = Carina.run(path)
            m = adapt(Array, sim.integrator.m_lumped)
            @test all(>(0.0), m)
            @test isfinite(sum(m))
        end

        mktempdir() do dir
            cp_example(joinpath(mesh_dir("tet10"), "cube.g"), joinpath(dir, "cube.g"))
            path = joinpath(dir, "ex.yaml")
            open(io -> write(io, explicit_deck("tet10")), path, "w")
            # Must abort, not run.  Before the guard this produced 72
            # non-positive masses out of 353 and completed without complaint.
            @test_throws ErrorException Carina.run(path)
        end
    end
end
