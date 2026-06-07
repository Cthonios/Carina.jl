# Elastic wave propagation in a clamped beam.
#
# Reference: Mota, Tezaur, Phlipot, "The Schwarz alternating method for
#   transient solid dynamics", IJNME 123:5036-5071, 2022, Section 3.3.
#
# Beam l x l x L, l = 1 mm, L = 1 m.  z in [-L/2, L/2].
# Linear elastic E = 1 GPa, nu = 0, rho = 1000 ⇒ c = sqrt(E/rho) = 1000 m/s,
# T = L/c = 1 ms.
# IC: Gaussian pulse u_z(z, 0) = a*exp(-z^2/(2 s^2)), a = 0.01, s = 0.02;
# v_z(z, 0) = 0; clamped BCs (homogeneous) on all six faces.
#
# Analytical solution (eq. 28):
#   u_z(z, t) = f(z - c t) + f(z + c t)
#             - f(z - c(T - t)) - f(z + c(T - t))
# where f(ξ) = (a/2) exp(-ξ^2 / (2 s^2)).
# Time derivatives obtained by direct differentiation:
#   v_z = c [ -f'(z-ct) + f'(z+ct) - f'(z-c(T-t)) + f'(z+c(T-t)) ]
#   a_z = c^2 [ f''(z-ct) + f''(z+ct) - f''(z-c(T-t)) - f''(z+c(T-t)) ]
# with f'(ξ) = -(ξ/s²) f(ξ),  f''(ξ) = (ξ²/s⁴ - 1/s²) f(ξ).
#
# References were previously pinned to Norma.jl regression-baseline numbers.
# Those are simply this analytical solution sampled at the same mesh; we now
# evaluate the analytical solution at the FEM mesh nodes directly and
# compare component-wise with paper-predicted FEM discretization tolerance
# (~0.3% rel-err per Table 1 → assert with rtol = 5e-3).

const _clamped_a = 0.01
const _clamped_s = 0.02
const _clamped_c = 1000.0
const _clamped_L = 1.0
const _clamped_T = _clamped_L / _clamped_c   # 1e-3 s

@inline _clamped_f(ξ)    = (_clamped_a / 2) * exp(-ξ^2 / (2 * _clamped_s^2))
@inline _clamped_fp(ξ)   = -(ξ / _clamped_s^2) * _clamped_f(ξ)
@inline _clamped_fpp(ξ)  = (ξ^2 / _clamped_s^4 - 1 / _clamped_s^2) * _clamped_f(ξ)

@inline function _clamped_uz(z, t)
    c = _clamped_c; T = _clamped_T
    return _clamped_f(z - c*t) + _clamped_f(z + c*t) -
           _clamped_f(z - c*(T-t)) - _clamped_f(z + c*(T-t))
end

@inline function _clamped_vz(z, t)
    c = _clamped_c; T = _clamped_T
    return c * ( -_clamped_fp(z - c*t) + _clamped_fp(z + c*t) -
                  _clamped_fp(z - c*(T-t)) + _clamped_fp(z + c*(T-t)) )
end

@inline function _clamped_az(z, t)
    c = _clamped_c; T = _clamped_T
    return c^2 * ( _clamped_fpp(z - c*t) + _clamped_fpp(z + c*t) -
                    _clamped_fpp(z - c*(T-t)) - _clamped_fpp(z + c*(T-t)) )
end

# Pull per-component max/min from a finished simulation.  Displacement uses the
# FEC full-DOF field (BC values already merged); velocity and acceleration live
# in the integrator (free-DOF) and are scattered back to the full DOF layout
# before reshaping to (3, num_nodes).
function _clamped_full_field(sim, sym::Symbol)
    # Norma-shape integrator state: ig.U/V/A are full-DOF.  BC slots are
    # written each step by predict! → FEC.update_field_dirichlet_bcs!, so
    # they already carry the prescribed values.  For homogeneous clamped
    # BCs g(t)=g'(t)=g''(t)≡0, so this is numerically the same as the old
    # "scatter free-DOF, leave BC = 0" path, but correct in general.
    if sym === :displacement
        return Vector(sim.params.field.data)
    elseif sym === :velocity
        return Vector(sim.integrator.V)
    else
        return Vector(sim.integrator.A)
    end
end

_clamped_max(sim, sym) = vec(maximum(reshape(_clamped_full_field(sim, sym), 3, :); dims = 2))
_clamped_min(sim, sym) = vec(minimum(reshape(_clamped_full_field(sim, sym), 3, :); dims = 2))

# Build the analytical reference for the z-component max/min over the mesh
# at time t.  Analytical solution has u_x = u_y = 0 identically.
function _clamped_analytical_z_extrema(coords, t)
    z = view(coords, 3, :)
    uz = [_clamped_uz(zi, t) for zi in z]
    vz = [_clamped_vz(zi, t) for zi in z]
    az = [_clamped_az(zi, t) for zi in z]
    return (max_uz = maximum(uz), min_uz = minimum(uz),
            max_vz = maximum(vz), min_vz = minimum(vz),
            max_az = maximum(az), min_az = minimum(az))
end

function _clamped_run_and_assert(example_dir, rtol)
    mktempdir() do dir
        cp_example(joinpath(example_dir, "clamped.g"),    joinpath(dir, "clamped.g"))
        cp_example(joinpath(example_dir, "clamped.toml"), joinpath(dir, "clamped.toml"))
        sim = Carina.run(joinpath(dir, "clamped.toml"))

        coords = reshape(Vector(sim.params_cpu.coords.data), 3, :)
        t_final = sim.controller.time
        ref = _clamped_analytical_z_extrema(coords, t_final)

        max_disp = _clamped_max(sim, :displacement)
        min_disp = _clamped_min(sim, :displacement)
        max_velo = _clamped_max(sim, :velocity)
        min_velo = _clamped_min(sim, :velocity)
        max_acce = _clamped_max(sim, :acceleration)
        min_acce = _clamped_min(sim, :acceleration)

        # Transverse components: analytical = 0, FEM should be in
        # round-off noise (held by clamped BCs).
        @test max_disp[1] ≈ 0.0          atol = 1.0e-06
        @test max_disp[2] ≈ 0.0          atol = 1.0e-06
        @test min_disp[1] ≈ 0.0          atol = 1.0e-06
        @test min_disp[2] ≈ 0.0          atol = 1.0e-06
        @test max_velo[1] ≈ 0.0          atol = 1.0e-06
        @test max_velo[2] ≈ 0.0          atol = 1.0e-06
        @test min_velo[1] ≈ 0.0          atol = 1.0e-06
        @test min_velo[2] ≈ 0.0          atol = 1.0e-06
        @test max_acce[1] ≈ 0.0          atol = 1.0e-06
        @test max_acce[2] ≈ 0.0          atol = 1.0e-06
        @test min_acce[1] ≈ 0.0          atol = 1.0e-06
        @test min_acce[2] ≈ 0.0          atol = 1.0e-06

        # Z-component: analytical reference sampled on the FEM mesh.
        # rtol = 5e-3 (~2× the paper's reported ~0.3% explicit / 0.28% implicit).
        @test max_disp[3] ≈ ref.max_uz   rtol = rtol
        @test min_disp[3] ≈ ref.min_uz   atol = 1.0e-06  # both ≈ 0
        @test max_velo[3] ≈ ref.max_vz   rtol = rtol
        @test min_velo[3] ≈ ref.min_vz   rtol = rtol
        @test max_acce[3] ≈ ref.max_az   rtol = rtol
        @test min_acce[3] ≈ ref.min_az   rtol = rtol
    end
end

@testset "Clamped Wave (Explicit)" begin
    # Central difference, dt = 1e-7, 100 steps to t = 1e-5 s.
    # Paper Table 1 reports ~0.35% relative error for explicit CM at this mesh.
    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics",
                           "explicit-dynamic", "clamped")
    _clamped_run_and_assert(example_dir, 5.0e-3)
end

@testset "Clamped Wave (Implicit)" begin
    # Newmark γ=0.5, β=0.25, dt = 1e-6, 10 steps to t = 1e-5 s.
    # Paper Table 1 reports ~0.28% relative error for implicit at this mesh.
    example_dir = joinpath(@__DIR__, "..", "examples", "mechanics",
                           "implicit-dynamic", "clamped")
    _clamped_run_and_assert(example_dir, 5.0e-3)
end
