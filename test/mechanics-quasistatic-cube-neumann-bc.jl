@testset "Mechanics Quasi-static Cube (Neumann BC)" begin
    # Unit cube [0,1]³, linear elastic (infinitesimal strain), E=1e9, ν=0.25.
    # BCs: u_x=0 on ssx-, u_y=0 on ssy-, u_z=0 on ssz-.
    # Neumann: traction t_z = +1e9*t on ssz+ (FEC sign convention: g = -traction).
    # Final time t=1.0 → applied traction 1e9 Pa.
    #
    # Matches Norma single-static-solid-neumann-bc (same E, traction, time stepping).
    #
    # Analytical solution (uniaxial stress, small strain):
    #   ε_z = t_z / E = 1e9 / 1e9 = 1.0    avg_uz = ε_z * 0.5 = 0.5
    #   ε_x = ε_y = -ν * ε_z = -0.25       avg_ux = avg_uy = -0.25 * 0.5 = -0.125

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics", "quasistatic", "cube-neumann-bc")
    mktempdir() do dir
        cp_example(joinpath(example_dir, "cube.g"),    joinpath(dir, "cube.g"))
        cp_example(joinpath(example_dir, "cube.yaml"), joinpath(dir, "cube.yaml"))
        sim = Carina.run(joinpath(dir, "cube.yaml"))
        avg = average_components(sim)

        @test avg[3] ≈  0.5   rtol=1e-6   # avg u_z (Norma: 0.500)
        @test avg[1] ≈ -0.125 rtol=1e-6   # avg u_x (Poisson; Norma: -0.125)
        @test avg[2] ≈ -0.125 rtol=1e-6   # avg u_y (Poisson; Norma: -0.125)
    end
end

# Resultant of a surface traction, per element type and direction.  The
# traction is assembled alone at the undeformed state; its resultant must be
# the traction times the face area, in the requested direction only.  Two
# defects hid here: the component of a traction was always taken as z (the
# comparison was against Symbols while the names are Strings), and the
# Jacobian of a triangular face was twice the face area, so tractions on
# tetrahedra were doubled.
@testset "Traction resultant on hexahedral and tetrahedral faces" begin
    FEC = Carina.FEC
    deck(mesh, side, comp) = """
type: single
input mesh file: $mesh
output mesh file: traction.e
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
  neumann:
    - side set: $side
      component: $comp
      function: "7.0 * t"
solver:
  type: newton
  linear solver:
    type: direct
  termination:
    fail when any:
      - maximum iterations: 16
    converge when any:
      - absolute residual: 1.0e-8
"""
    meshes = Dict("hex8"  => joinpath(@__DIR__, "..", "examples", "meshes", "cube", "cube.g"),
                  "tet4"  => joinpath(@__DIR__, "..", "examples", "meshes", "cube-tet4", "cube.g"),
                  "tet10" => joinpath(@__DIR__, "..", "examples", "meshes", "cube-tet10", "cube.g"))
    for (v, mesh) in meshes, (side, comp, idx) in (("ssz+", "z", 3), ("ssx+", "y", 2), ("ssy+", "x", 1))
        @testset "$v $side traction in $comp" begin
        mktempdir() do dir
            cp_example(mesh, joinpath(dir, "cube.g"))
            path = joinpath(dir, "traction.yaml")
            open(io -> write(io, deck("cube.g", side, comp)), path, "w")
            dict = Carina.YAML.load_file(path; dicttype=Dict{String,Any})
            sim = Carina.create_simulation(dict, dir)
            asm = sim.asm_cpu; p = sim.params_cpu
            p.times.time_current = 1.0
            FEC.update_bc_values!(p, asm)
            U0 = zeros(length(asm.dof.unknown_dofs))
            FEC.assemble_vector!(asm, FEC.residual, U0, p)
            FEC.assemble_vector_neumann_bc!(asm, U0, p)
            # the full residual, constrained dofs included: nodes of the loaded
            # face that lie on a constrained face still carry traction
            f = -reshape(Array(asm.residual_storage.data), 3, :)
            # the loaded face of the unit cube has area 1; the traction is 7
            for i in 1:3
                expected = i == idx ? 7.0 : 0.0
                @test isapprox(sum(f[i, :]), expected; atol = 1e-10)
            end
            FEC.close(sim.post_processor)
        end
        end
    end
end
