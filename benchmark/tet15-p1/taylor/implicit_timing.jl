# Time per element kernel of the implicit integrators for TET15-P1 on one
# device, split and general form, on the Taylor bar mesh of a level
# (meshes/taylor-h<h>-tet15.g), in a deformed state with plastic flow.
#
#   julia -t <N> --project=. benchmark/tet15-p1/taylor/implicit_timing.jl --h 0.75
#        [--device cpu|rocm|cuda] [--warm 400] [--reps 20] [--tag <text>]
#        [--always-inline]   (CUDA: inline every device-function call)
#
# A GPU run needs the vendor package loaded in the calling session, as for
# run.jl:  julia --project=bin -e 'using CUDA; append!(ARGS, [...]);
# include("benchmark/tet15-p1/taylor/implicit_timing.jl")' with
# JULIA_LOAD_PATH="@:$PWD:@stdlib".
#
# The state: --warm steps of the explicit run from the impact (400 steps =
# 4 us at h = 0.75 mm); the kernels are evaluated at the displacement of the
# last step with the internal variables of the step before as the old state,
# so that the points in plastic flow are past the yield surface and not on
# it.  The state is copied into a quasi-static problem on the same device
# (iterative solver, Jacobi preconditioner), whose assembler the kernels use.
#
# Measured for each form (J2, θ = J − 1, for which both forms give the same
# residual and tangent), each the median of --reps repetitions after one
# warm-up call, with the device synchronized before and after:
#   residual      the internal force
#   action        the matrix-free action of the tangent on a vector
#   diagonal      the diagonal of the tangent (Jacobi preconditioner)
#   matrix        the assembled tangent (CPU only; NaN on a GPU)
# and, for the general form,
#   recompute     the residual with the material called again in the third
#                 pass instead of storing its fourteen stresses (defined
#                 here, not in Carina; Carina stores them)
# One JSON record per form is appended to implicit-timing.jsonl in this
# directory; the table is printed.

using Carina
using Exodus
using Printf
using Dates
using StaticArrays: SVector
using LinearAlgebra: det, dot

const DIR = @__DIR__
const FEC = Carina.FEC
const RFE = Carina.RFE

function parse_args(args)
    opts = Dict("--h" => "0.75", "--device" => "cpu", "--warm" => "400", "--reps" => "20",
                "--tag" => "", "--always-inline" => "false")
    i = 1
    while i <= length(args)
        if args[i] == "--always-inline"
            opts["--always-inline"] = "true"; i += 1; continue
        end
        haskey(opts, args[i]) || error("unknown option $(args[i])")
        opts[args[i]] = args[i + 1]; i += 2
    end
    return opts
end

include(joinpath(DIR, "backend.jl"))

function deck(h, form, out, implicit)
    integrator = implicit ? """
time integrator:
  type: quasi static
  initial time: 0.0
  final time: 1.0
  time step: 1.0
solver:
  type: newton
  linear solver:
    type: iterative
    tolerance: 1.0e-10
    maximum iterations: 100
    preconditioner:
      type: jacobi
""" : """
time integrator:
  type: central difference
  initial time: 0.0
  final time: 8.0e-5
  time step: 1.0e-8
  cfl: 0.8
  stable time step interval: 10
"""
    return """
type: single
input mesh file: $(joinpath(DIR, "meshes", "taylor-h$(h)-tet15.g"))
output mesh file: $out
output interval: 1.0
model:
  type: solid mechanics
  volumetric projection: linear
  volumetric form: $form
  volumetric strain: J - 1
  material:
    blocks:
      bar: j2 plasticity
    j2 plasticity:
      elastic modulus: 117.0e9
      Poisson's ratio: 0.35
      density: 8930.0
      yield stress: 400.0e6
      hardening modulus: 100.0e6
$integrator
initial conditions:
  velocity:
    - node set: all
      component: z
      function: "-227.0"
boundary conditions:
  dirichlet:
    - node set: impact
      component: z
      function: "0.0"
"""
end

# The general residual with the material called again in the third pass.
@inline function recompute_residual(physics, ref_fe, x_el, t, dt, u_el, u_el_old, states, props_el)
    vv = Carina.volumetric_variable(physics)
    NQ = RFE.num_cell_quadrature_points(ref_fe)
    T  = eltype(x_el)
    NDOF = 3 * RFE.num_cell_dofs(ref_fe)
    θ̄, Minv, everted = Carina._project(physics, ref_fe, x_el, u_el)
    u_safe = everted ? zero(u_el) : u_el
    θ̄_safe = everted ? zero(θ̄) : θ̄
    b̄ = zero(θ̄)
    for q in 1:NQ
        _, p̃, J̃, _, χ = Carina._general_point(physics, ref_fe, x_el, dt, u_safe, states,
                                              props_el, θ̄_safe, q, Val(false))
        JxW = FEC.map_interpolants(FEC._cell_interpolants(ref_fe, q), x_el).JxW
        b̄ = b̄ + (JxW * p̃ / Carina._θ1(vv, J̃)) * χ
    end
    p̄ = Minv * b̄
    R = zero(SVector{NDOF, T})
    for q in 1:NQ
        P̃, p̃, J̃, s, χ = Carina._general_point(physics, ref_fe, x_el, dt, u_safe, states,
                                              props_el, θ̄_safe, q, Val(false))
        cell = FEC.map_interpolants(FEC._cell_interpolants(ref_fe, q), x_el)
        F = Carina._gradient_at(physics, cell, u_safe)
        F = F + one(F)
        J = det(F)
        P = s * P̃ + (dot(χ, p̄) * Carina._θ1(vv, J) * J - p̃ * J̃) * inv(F)'
        R = R + Carina._scatter_qp(cell.∇N_X, P, cell.JxW)
    end
    return everted ? T(NaN) * R : R
end

function median_time(f, sync, reps)
    f(); sync()
    ts = Float64[]
    for _ in 1:reps
        sync(); t = time_ns(); f(); sync()
        push!(ts, (time_ns() - t) / 1e9)
    end
    return sort(ts)[(length(ts) + 1) ÷ 2]
end

function load(dir, name, text, backend)
    path = joinpath(dir, name)
    open(io -> write(io, text), path, "w")
    dict = Carina.YAML.load_file(path; dicttype = Dict{String, Any})
    return Carina.create_simulation(dict, dir; backend = backend)
end

function time_form(h, form, backend, sync, warm, reps)
    dir = mktempdir()
    on_device = !(backend isa Carina.KA.CPU)
    sim = load(dir, "explicit.yaml", deck(h, form, joinpath(dir, "e.e"), false), backend)
    ig = sim.integrator; p = sim.params
    S_prev = similar(p.state_new.data)
    for k in 1:warm
        Carina._pre_step_hook!(ig, sim)
        k == warm && copyto!(S_prev, p.state_new.data)
        p.times.Δt = ig.time_step
        Carina._advance_one_step!(sim)
        on_device && k % Carina._DEVICE_GC_INTERVAL == 0 && GC.gc(false)
    end
    sync()
    simq = load(dir, "implicit.yaml", deck(h, form, joinpath(dir, "q.e"), true), backend)
    pq = simq.params; asm = simq.integrator.asm
    length(asm.dof.unknown_dofs) == length(ig.asm.dof.unknown_dofs) ||
        error("the explicit and the quasi-static problem have different unknowns")
    copyto!(pq.state_old.data, S_prev); copyto!(pq.state_new.data, S_prev)
    U = similar(ig.U, length(asm.dof.unknown_dofs))
    copyto!(U, view(ig.U, ig.asm.dof.unknown_dofs))
    V = similar(U); copyto!(V, sin.(0.37 .* (1:length(U))))
    ns = Carina.CM.num_state_variables(first(values(pq.physics)).constitutive_model)
    eqps = maximum(reshape(Array(pq.state_new.data), ns, :)[10, :])

    t_res = median_time(() -> FEC.assemble_vector!(asm, FEC.residual, U, pq), sync, reps)
    t_act = median_time(() -> Carina._assemble_action!(asm, FEC.stiffness_action, U, V, pq), sync, reps)
    t_dia = median_time(() -> FEC.assemble_diagonal!(asm, Carina.StiffnessDiagonal(), U, pq), sync, reps)
    t_mat = NaN
    if !on_device
        try
            t_mat = median_time(() -> FEC.assemble_stiffness!(asm, FEC.stiffness, U, pq), sync,
                                max(3, reps ÷ 4))
        catch e
            @warn "matrix assembly unavailable: $(sprint(showerror, e))"
        end
    end
    t_rec = NaN
    if form == "general"
        FEC.assemble_vector!(asm, FEC.residual, U, pq); R1 = Array(copy(FEC.residual(asm)))
        FEC.assemble_vector!(asm, recompute_residual, U, pq); R2 = Array(copy(FEC.residual(asm)))
        d = maximum(abs.(R1 .- R2)) / maximum(abs.(R1))
        d < 1e-10 || error("the recomputed residual differs from Carina's by $d")
        t_rec = median_time(() -> FEC.assemble_vector!(asm, recompute_residual, U, pq), sync, reps)
    end
    return (; dofs = length(asm.dof), free_dofs = length(U), max_eqps = eqps,
            residual_s = t_res, action_s = t_act, diagonal_s = t_dia, matrix_s = t_mat,
            recompute_s = t_rec)
end

function main(args)
    opts = parse_args(args)
    h = parse(Float64, opts["--h"])
    warm = parse(Int, opts["--warm"]); reps = parse(Int, opts["--reps"])
    backend = backend_of(opts["--device"]; always_inline = opts["--always-inline"] == "true")
    sync = backend isa Carina.KA.CPU ? (() -> nothing) : (() -> Carina.KA.synchronize(backend))
    e = ExodusDatabase(joinpath(DIR, "meshes", "taylor-h$(h)-tet15.g"), "r")
    nelem = sum(size(b.conn, 2) for b in read_sets(e, Block)); close(e)
    json(v::AbstractString) = "\"" * replace(v, "\"" => "\\\"") * "\""
    json(v::Integer) = string(v)
    json(v::AbstractFloat) = isnan(v) ? "null" : @sprintf("%.6e", v)
    @printf("%s  %s  %d threads  h = %g mm  %d elements\n", gethostname(), opts["--device"],
            Threads.nthreads(), h, nelem)
    @printf("  %-8s %10s %10s %10s %10s %10s   (ms)\n", "form", "residual", "action",
            "diagonal", "matrix", "recompute")
    for form in ("split", "general")
        # invokelatest: backend_of may have redefined KA.get_backend (--always-inline)
        # in a newer world than the one main() runs in.
        r = Base.invokelatest(time_form, h, form, backend, sync, warm, reps)
        rec = (; host = gethostname(), date = string(now()), tag = opts["--tag"],
               device = opts["--device"], backend = string(typeof(backend)),
               threads = Threads.nthreads(), julia = string(VERSION),
               carina = strip(read(`git -C $DIR rev-parse --short HEAD`, String)),
               always_inline = opts["--always-inline"],
               h, form, elements = nelem, warm_steps = warm, reps, r...)
        line = "{" * join(("\"$k\": " * json(v) for (k, v) in pairs(rec)), ", ") * "}"
        open(io -> println(io, line), joinpath(DIR, "implicit-timing.jsonl"), "a")
        @printf("  %-8s %10.2f %10.2f %10.2f %10.2f %10.2f   max eqps %.4f\n", form,
                1e3 * r.residual_s, 1e3 * r.action_s, 1e3 * r.diagonal_s, 1e3 * r.matrix_s,
                1e3 * r.recompute_s, r.max_eqps)
    end
end

main(ARGS)
