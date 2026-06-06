@testset "Mechanics Quasi-static Cube (Point Load)" begin
    # Same problem as the Neumann BC test but with the traction replaced by
    # equivalent point loads on node set nsz+.
    #
    # Single hex8 element [−0.5,0.5]³.  The z+ face has 4 nodes, each with
    # tributary area 0.25 (bilinear shape functions, uniform quad).
    #
    # Original traction: function = "1e9·t" on ssz+ (positive = tension in +z).
    # Equivalent point load per node: 0.25e9·t N (total force / 4 nodes).
    #
    # Analytical: ε_z = 1, avg u_z = 0.5, avg u_x = avg u_y = −0.125.

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics", "quasistatic", "cube-neumann-bc")
    mktempdir() do dir
        cp_example(joinpath(example_dir, "cube.g"), joinpath(dir, "cube.g"))

        toml = """
type = "single"

input_mesh_file = "cube.g"
output_mesh_file = "cube.e"

[model]
type = "solid_mechanics"

[model.material.blocks]
cube = "linear_elastic"

[model.material.linear_elastic]
elastic_modulus = 1.0e9
poissons_ratio = 0.25
density = 1000.0

[time_integrator]
type = "quasi_static"
initial_time = 0.0
final_time = 1.0
time_step = 0.1

[[boundary_conditions.dirichlet]]
side_set = "ssx-"
component = "x"
function = "0.0"

[[boundary_conditions.dirichlet]]
side_set = "ssy-"
component = "y"
function = "0.0"

[[boundary_conditions.dirichlet]]
side_set = "ssz-"
component = "z"
function = "0.0"

[[boundary_conditions.neumann]]
node_set = "nsz+"
component = "z"
function = "0.25e+09 * t"

[solver]
type = "newton"

[[solver.termination]]
type = "combo"
combo = "or"
tests = [
    { type = "absolute_residual", tolerance = 1.0e-8 },
    { type = "relative_residual", tolerance = 1.0e-14 },
]

[[solver.termination]]
type = "maximum_iterations"
value = 16

[solver.linear_solver]
type = "direct"
"""
        write(joinpath(dir, "cube_point_load.toml"), toml)
        sim = Carina.run(joinpath(dir, "cube_point_load.toml"))
        avg = average_components(sim)

        @test avg[3] ≈  0.5   rtol=1e-6   # avg u_z (tension)
        @test avg[1] ≈ -0.125 rtol=1e-6   # avg u_x (Poisson contraction)
        @test avg[2] ≈ -0.125 rtol=1e-6   # avg u_y (Poisson contraction)
    end
end
