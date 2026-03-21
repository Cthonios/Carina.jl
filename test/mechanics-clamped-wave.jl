# Elastic wave propagation in a clamped beam.
#
# Reference: Mota, Tezaur, Phlipot, "The Schwarz alternating method for
#   transient solid dynamics", IJNME 123:5036-5071, 2022, Section 3.3.
#
# Beam l x l x L, l = 1 mm, L = 1 m.  z in [-L/2, L/2].
# Linear elastic E = 1 GPa, nu = 0, rho = 1000.  c = 1000 m/s, T = 1 ms.
# IC: Gaussian pulse u_z(z,0) = a*exp(-z^2/(2s^2)), a = 0.01, s = 0.02.
#
# Exact solution (eq. 28):
#   u_z(z,t) = f(z-ct) + f(z+ct) - f(z-c(T-t)) - f(z+c(T-t))
# where f(z) = (a/2)*exp(-z^2/(2s^2)).

const _clamped_a = 0.01
const _clamped_s = 0.02
const _clamped_c = 1000.0   # sqrt(E/rho)
const _clamped_L = 1.0
const _clamped_T = _clamped_L / _clamped_c   # 1e-3 s

function _clamped_f(z)
    return (_clamped_a / 2) * exp(-z^2 / (2 * _clamped_s^2))
end

function _clamped_exact_uz(z, t)
    c = _clamped_c; T = _clamped_T
    return _clamped_f(z - c*t) + _clamped_f(z + c*t) -
           _clamped_f(z - c*(T - t)) - _clamped_f(z + c*(T - t))
end

# Relative error in z-displacement averaged over all nodes at the final time.
function _clamped_z_disp_rel_error(sim)
    coords = reshape(sim.params_cpu.h1_coords.data, 3, :)
    disp   = reshape(sim.params.h1_field.data, 3, :)
    z_coords = coords[3, :]
    uz_comp  = disp[3, :]

    t_final = sim.controller.time
    uz_exact = [_clamped_exact_uz(z, t_final) for z in z_coords]

    denom = sum(abs, uz_exact)
    denom == 0.0 && return 0.0
    return sum(abs, uz_comp .- uz_exact) / denom
end

@testset "Clamped Wave (Explicit)" begin
    # Central difference, dt = 1e-7, 100 steps to t = 1e-5 s.
    # At t = 1e-5 the wave has barely moved (c*t = 0.01 m = 1% of L),
    # so the pulse is still near the center and far from boundaries.
    # Relative error should be < 1% (Table 1 in paper: ~0.35% for explicit CM).

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics",
                           "explicit-dynamic", "clamped")
    mktempdir() do dir
        cp_example(joinpath(example_dir, "clamped.g"),    joinpath(dir, "clamped.g"))
        cp_example(joinpath(example_dir, "clamped.yaml"), joinpath(dir, "clamped.yaml"))
        sim = Carina.run(joinpath(dir, "clamped.yaml"))

        rel_err = _clamped_z_disp_rel_error(sim)
        @test rel_err < 0.01   # < 1% relative error vs analytical solution
    end
end

@testset "Clamped Wave (Implicit)" begin
    # Newmark average acceleration, dt = 1e-6, 10 steps to t = 1e-5 s.
    # Same final time as explicit.  Implicit uses 10x larger time step.
    # Relative error should be < 1% (Table 1: ~0.28% for implicit).

    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics",
                           "implicit-dynamic", "clamped")
    mktempdir() do dir
        cp_example(joinpath(example_dir, "clamped.g"),    joinpath(dir, "clamped.g"))
        cp_example(joinpath(example_dir, "clamped.yaml"), joinpath(dir, "clamped.yaml"))
        sim = Carina.run(joinpath(dir, "clamped.yaml"))

        rel_err = _clamped_z_disp_rel_error(sim)
        @test rel_err < 0.01   # < 1% relative error vs analytical solution
    end
end
