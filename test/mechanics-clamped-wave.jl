# Elastic wave propagation in a clamped beam.
#
# Reference: Mota, Tezaur, Phlipot, "The Schwarz alternating method for
#   transient solid dynamics", IJNME 123:5036-5071, 2022, Section 3.3.
#
# Beam l x l x L, l = 1 mm, L = 1 m.  z in [-L/2, L/2].
# Linear elastic E = 1 GPa, nu = 0, rho = 1000.  c = 1000 m/s, T = 1 ms.
# IC: Gaussian pulse u_z(z,0) = a*exp(-z^2/(2s^2)), a = 0.01, s = 0.02.
#
# Reference values are pinned against Norma.jl (test/single-{explicit,implicit}
# -dynamic-solid-clamped.jl) at the same mesh, integrators, and final time.
# Tolerances per quantity match Norma's verbatim.

# Pull per-component max/min from a finished simulation.  Displacement uses the
# FEC full-DOF field (BC values already merged); velocity and acceleration live
# in the integrator (free-DOF) and are scattered back to the full DOF layout
# before reshaping to (3, num_nodes).
function _clamped_full_field(sim, sym::Symbol)
    asm = sim.integrator.asm
    n_total = length(sim.params.field.data)
    if sym === :displacement
        full = Vector(sim.params.field.data)
    else
        src = sym === :velocity ? sim.integrator.V : sim.integrator.A
        full = zeros(n_total)
        full[Vector(asm.dof.unknown_dofs)] = Vector(src)
    end
    return reshape(full, 3, :)
end

_clamped_max(sim, sym) = vec(maximum(_clamped_full_field(sim, sym); dims = 2))
_clamped_min(sim, sym) = vec(minimum(_clamped_full_field(sim, sym); dims = 2))

@testset "Clamped Wave (Explicit)" begin
    # Central difference, dt = 1e-7, 100 steps to t = 1e-5 s.
    # Tolerances mirror Norma.jl test/single-explicit-dynamic-solid-clamped.jl.
    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics",
                           "explicit-dynamic", "clamped")
    mktempdir() do dir
        cp_example(joinpath(example_dir, "clamped.g"),    joinpath(dir, "clamped.g"))
        cp_example(joinpath(example_dir, "clamped.toml"), joinpath(dir, "clamped.toml"))
        sim = Carina.run(joinpath(dir, "clamped.toml"))

        max_disp = _clamped_max(sim, :displacement)
        min_disp = _clamped_min(sim, :displacement)
        max_velo = _clamped_max(sim, :velocity)
        min_velo = _clamped_min(sim, :velocity)
        max_acce = _clamped_max(sim, :acceleration)
        min_acce = _clamped_min(sim, :acceleration)

        @test max_disp[1] ≈ 0.0          atol = 1.0e-06
        @test max_disp[2] ≈ 0.0          atol = 1.0e-06
        @test max_disp[3] ≈ 0.00882497   rtol = 8.0e-05
        @test min_disp[1] ≈ 0.0          atol = 1.0e-06
        @test min_disp[2] ≈ 0.0          atol = 1.0e-06
        @test min_disp[3] ≈ 0.0          atol = 1.0e-06
        @test max_velo[1] ≈ 0.0          atol = 1.0e-06
        @test max_velo[2] ≈ 0.0          atol = 1.0e-06
        @test max_velo[3] ≈ 98.7781      rtol = 6.0e-04
        @test min_velo[1] ≈ 0.0          atol = 1.0e-06
        @test min_velo[2] ≈ 0.0          atol = 1.0e-06
        @test min_velo[3] ≈ -220.624     rtol = 5.0e-04
        @test max_acce[1] ≈ 0.0          atol = 1.0e-06
        @test max_acce[2] ≈ 0.0          atol = 1.0e-06
        @test max_acce[3] ≈ 7.95606e6    rtol = 6.0e-04
        @test min_acce[1] ≈ 0.0          atol = 1.0e-06
        @test min_acce[2] ≈ 0.0          atol = 1.0e-06
        @test min_acce[3] ≈ -1.65468e7   rtol = 2.0e-06
    end
end

@testset "Clamped Wave (Implicit)" begin
    # Newmark average acceleration, dt = 1e-6, 10 steps to t = 1e-5 s.
    # Tolerances mirror Norma.jl test/single-implicit-dynamic-solid-clamped.jl.
    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics",
                           "implicit-dynamic", "clamped")
    mktempdir() do dir
        cp_example(joinpath(example_dir, "clamped.g"),    joinpath(dir, "clamped.g"))
        cp_example(joinpath(example_dir, "clamped.toml"), joinpath(dir, "clamped.toml"))
        sim = Carina.run(joinpath(dir, "clamped.toml"))

        max_disp = _clamped_max(sim, :displacement)
        min_disp = _clamped_min(sim, :displacement)
        max_velo = _clamped_max(sim, :velocity)
        min_velo = _clamped_min(sim, :velocity)
        max_acce = _clamped_max(sim, :acceleration)
        min_acce = _clamped_min(sim, :acceleration)

        @test max_disp[1] ≈ 0.0          atol = 1.0e-06
        @test max_disp[2] ≈ 0.0          atol = 1.0e-06
        @test max_disp[3] ≈ 0.00882497   rtol = 8.0e-05
        @test min_disp[1] ≈ 0.0          atol = 1.0e-06
        @test min_disp[2] ≈ 0.0          atol = 1.0e-06
        @test min_disp[3] ≈ 0.0          atol = 1.0e-06
        @test max_velo[1] ≈ 0.0          atol = 1.0e-06
        @test max_velo[2] ≈ 0.0          atol = 1.0e-06
        @test max_velo[3] ≈ 98.7781      rtol = 5.0e-05
        @test min_velo[1] ≈ 0.0          atol = 1.0e-06
        @test min_velo[2] ≈ 0.0          atol = 1.0e-06
        @test min_velo[3] ≈ -220.624     rtol = 2.0e-04
        @test max_acce[1] ≈ 0.0          atol = 1.0e-06
        @test max_acce[2] ≈ 0.0          atol = 1.0e-06
        @test max_acce[3] ≈ 7.95606e6    rtol = 5.0e-04
        @test min_acce[1] ≈ 0.0          atol = 1.0e-06
        @test min_acce[2] ≈ 0.0          atol = 1.0e-06
        @test min_acce[3] ≈ -1.65468e7   rtol = 9.0e-04
    end
end
