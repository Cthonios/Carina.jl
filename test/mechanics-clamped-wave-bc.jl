# BC-driven Gaussian pulse in a clamped bar.  Mirrors Norma.jl tests
#   test/single-{explicit,implicit}-dynamic-solid-clamped-bc.jl.
#
# Right-traveling pulse injected by a time-dependent Dirichlet BC at z = -L/2
# instead of an initial condition.  Bar: 1 mm × 1 mm × 1 m, E = 1 GPa, ν = 0,
# ρ = 1000 ⇒ c = 1000 m/s.  Driver: g(t) = a*exp(-(t-tc)^2/(2 τ^2)),
# a = 1e-3, tc = 2.5e-4, τ = 5e-5.  Final time t_f = 7.5e-4 places the
# pulse peak (η = 0) at z = c·(t_f - tc) - L/2 = 0, strictly before reflection
# (first arrival at z = +0.5 is at t = tc + L/c = 1.25e-3).  At t_f the
# references at z ∈ {-0.05, 0, +0.05} sit on exact nodes (h = 1 mm) and
# correspond to η ∈ {+τ, 0, -τ}, sampling the displacement peak and the
# velocity peaks on either side of it.
#
# This exercises Carina's time-varying Dirichlet BC machinery for both
# implicit Newmark and explicit central-difference integrators.

# Build a full-DOF view of an integrator-side field (V or A) by scattering
# the free-DOF buffer into a length-3*num_nodes vector with zeros at
# constrained positions.  For homogeneous Dirichlet that's correct.  For
# the BC-driven test the prescribed value at the driver node is the
# *displacement* (carried by sim.params.field.data); V_BC and A_BC are
# not stored in Carina today and read as zero here.
function _clamped_bc_full(sim, sym::Symbol)
    asm = sim.integrator.asm
    n_total = length(sim.params.field.data)
    if sym === :displacement
        return Vector(sim.params.field.data)
    else
        src = sym === :velocity ? sim.integrator.V : sim.integrator.A
        full = zeros(n_total)
        full[Vector(asm.dof.unknown_dofs)] = Vector(src)
        return full
    end
end

# Locate the mesh-node whose z-coordinate is closest to z_target, asserting
# the snap is exact within 1 nm (the mesh has h = 1 mm).
function _clamped_bc_node_at(z_coords, z_target)
    idx = argmin(abs.(z_coords .- z_target))
    @assert abs(z_coords[idx] - z_target) < 1.0e-9 "no node at z = $z_target (closest $(z_coords[idx]))"
    return idx
end

@testset "Clamped Wave BC Pulse (Explicit)" begin
    # Central difference, dt = 5e-7 ⇒ τ/Δt = 100, CFL = 0.5.
    # Errors ≲ 0.03% after the pulse traverses 1 m.
    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics",
                           "explicit-dynamic", "clamped-bc")
    mktempdir() do dir
        cp_example(joinpath(example_dir, "clamped-bc.g"),    joinpath(dir, "clamped-bc.g"))
        cp_example(joinpath(example_dir, "clamped-bc.toml"), joinpath(dir, "clamped-bc.toml"))
        sim = Carina.run(joinpath(dir, "clamped-bc.toml"))

        a = 1.0e-3
        τ = 5.0e-5
        sqrt_e = sqrt(exp(1))
        u_amp  = a
        v_amp  = a / (τ * sqrt_e)
        a_amp  = a / τ^2
        u_eta1 = a / sqrt_e

        coords = reshape(Vector(sim.params_cpu.coords.data), 3, :)
        z_coords = coords[3, :]
        i_peak  = _clamped_bc_node_at(z_coords, 0.0)
        i_lead  = _clamped_bc_node_at(z_coords, 0.05)
        i_trail = _clamped_bc_node_at(z_coords, -0.05)

        U_full = _clamped_bc_full(sim, :displacement)
        V_full = _clamped_bc_full(sim, :velocity)
        A_full = _clamped_bc_full(sim, :acceleration)
        uz(i) = U_full[3 * i]
        vz(i) = V_full[3 * i]
        az(i) = A_full[3 * i]

        @test uz(i_peak)         ≈ u_amp    rtol = 1.0e-06
        @test abs(vz(i_peak))     < 1.0e-03 * v_amp
        @test az(i_peak)         ≈ -a_amp   rtol = 1.0e-03

        @test uz(i_lead)         ≈ u_eta1   rtol = 1.0e-03
        @test vz(i_lead)         ≈ +v_amp   rtol = 1.0e-03
        @test abs(az(i_lead))     < 1.0e-03 * a_amp

        @test uz(i_trail)        ≈ u_eta1   rtol = 1.0e-03
        @test vz(i_trail)        ≈ -v_amp   rtol = 1.0e-03
        @test abs(az(i_trail))    < 1.0e-03 * a_amp
    end
end

@testset "Clamped Wave BC Pulse (Implicit)" begin
    # Newmark γ=0.5, β=0.25, dt = 2e-6 ⇒ τ/Δt = 25.  ≲ 0.3% error in u, v, a
    # after the pulse traverses 1 m; tolerances below are 2-3× the observed.
    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics",
                           "implicit-dynamic", "clamped-bc")
    mktempdir() do dir
        cp_example(joinpath(example_dir, "clamped-bc.g"),    joinpath(dir, "clamped-bc.g"))
        cp_example(joinpath(example_dir, "clamped-bc.toml"), joinpath(dir, "clamped-bc.toml"))
        sim = Carina.run(joinpath(dir, "clamped-bc.toml"))

        a = 1.0e-3
        τ = 5.0e-5
        sqrt_e = sqrt(exp(1))
        u_amp  = a
        v_amp  = a / (τ * sqrt_e)
        a_amp  = a / τ^2
        u_eta1 = a / sqrt_e

        coords = reshape(Vector(sim.params_cpu.coords.data), 3, :)
        z_coords = coords[3, :]
        i_peak  = _clamped_bc_node_at(z_coords, 0.0)
        i_lead  = _clamped_bc_node_at(z_coords, 0.05)
        i_trail = _clamped_bc_node_at(z_coords, -0.05)

        U_full = _clamped_bc_full(sim, :displacement)
        V_full = _clamped_bc_full(sim, :velocity)
        A_full = _clamped_bc_full(sim, :acceleration)
        uz(i) = U_full[3 * i]
        vz(i) = V_full[3 * i]
        az(i) = A_full[3 * i]

        @test uz(i_peak)         ≈ u_amp    rtol = 1.0e-04
        @test abs(vz(i_peak))     < 0.01 * v_amp
        @test az(i_peak)         ≈ -a_amp   rtol = 2.0e-03

        @test uz(i_lead)         ≈ u_eta1   rtol = 5.0e-03
        @test vz(i_lead)         ≈ +v_amp   rtol = 5.0e-03
        @test abs(az(i_lead))     < 0.01 * a_amp

        @test uz(i_trail)        ≈ u_eta1   rtol = 5.0e-03
        @test vz(i_trail)        ≈ -v_amp   rtol = 5.0e-03
        @test abs(az(i_trail))    < 0.01 * a_amp
    end
end
