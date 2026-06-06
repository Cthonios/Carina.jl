@testset "Mechanics Quasi-static Cube (Chebyshev Preconditioner)" begin
    # Same problem as mechanics-quasistatic-cube.jl but solved with
    # CG + Chebyshev polynomial preconditioner (degree 5) instead of direct.
    #
    # Verifies that the Chebyshev preconditioner produces the same physical
    # result to within discretization tolerance.

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics", "quasistatic", "cube")
    mktempdir() do dir
        cp_example(joinpath(example_dir, "cube.g"), joinpath(dir, "cube.g"))

        # Write a TOML config with iterative solver + Chebyshev preconditioner.
        toml_path = joinpath(dir, "cube_chebyshev.toml")
        open(toml_path, "w") do io
            write(io, """
type = "single"

input_mesh_file = "cube.g"
output_mesh_file = "cube_chebyshev.e"

[model]
type = "solid_mechanics"

[model.material.blocks]
cube = "neohookean"

[model.material.neohookean]
elastic_modulus = 10.0e9
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

[[boundary_conditions.dirichlet]]
side_set = "ssz+"
component = "z"
function = "1.0e-3 * t"

[solver]
type = "newton"

[[solver.termination]]
type = "combo"
combo = "or"
tests = [
    { type = "absolute_residual", tolerance = 1.0e-6 },
    { type = "relative_residual", tolerance = 1.0e-10 },
]

[[solver.termination]]
type = "maximum_iterations"
value = 16

[solver.linear_solver]
type = "iterative"
tolerance = 1.0e-10
maximum_iterations = 500

[solver.linear_solver.preconditioner]
type = "chebyshev"
degree = 5
""")
        end

        sim = Carina.run(toml_path)
        avg = average_components(sim)
        mx  = maximum_components(sim)

        @test avg[3] ≈  5.00e-4 rtol=1e-4   # avg u_z (analytical)
        @test avg[1] ≈ -1.25e-4 rtol=1e-2   # avg u_x (Poisson)
        @test avg[2] ≈ -1.25e-4 rtol=1e-2   # avg u_y (Poisson)
        @test mx[3]  ≈  1.00e-3 rtol=1e-6   # max u_z = prescribed BC (exact)
    end
end
