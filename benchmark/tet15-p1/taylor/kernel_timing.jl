# Time per element kernel and per explicit step of TET15-P1 on one device, on
# the Taylor bar mesh of a level (meshes/taylor-h<h>-tet15.g), in a deformed
# state with plastic flow (the state after `--warm` steps of the explicit
# integration from the impact, 400 steps = 4 us at h = 0.75 mm).
#
#   julia -t <N> --project=. benchmark/tet15-p1/taylor/kernel_timing.jl --h 0.75
#        [--device cpu|rocm|cuda] [--form split|general] [--warm 400] [--reps 20]
#        [--tag <text>]
#
# A GPU run needs the vendor package loaded in the calling session, as for
# run.jl:  julia --project=bin -e 'using AMDGPU; append!(ARGS, [...]);
# include("benchmark/tet15-p1/taylor/kernel_timing.jl")' with
# JULIA_LOAD_PATH="@:$PWD:@stdlib".
#
# Measured, each the median of --reps repetitions after one warm-up call, with
# the device synchronized before and after:
#   residual      assembly of the internal force (one residual evaluation)
#   step          one explicit step without the stable-step recomputation
#   element dt    the element estimate of the stable step (every 10 steps in a run)
#   global dt     the eigenvalue estimate of the stable step (every 200 steps),
#                 the median of 5, each after a garbage collection on a device
# and the time stepping of a full run to 80 us is estimated from them with the
# step count of the run on the same mesh.  One JSON record per invocation is
# appended to kernel-timing.jsonl in this directory; the table is printed.
#
# Reported with the record: host, device, threads, Julia and package versions,
# element and unknown counts, the warm-up state (max eqps).

using Carina
using Exodus
using Printf
using Dates

const DIR = @__DIR__
const FEC = Carina.FEC

function parse_args(args)
    opts = Dict("--h" => "0.75", "--device" => "cpu", "--form" => "split",
                "--warm" => "400", "--reps" => "20", "--tag" => "")
    i = 1
    while i <= length(args)
        haskey(opts, args[i]) || error("unknown option $(args[i])")
        opts[args[i]] = args[i + 1]; i += 2
    end
    return opts
end

include(joinpath(DIR, "backend.jl"))

function deck(h, form, out)
    return """
type: single
input mesh file: $(joinpath(DIR, "meshes", "taylor-h$(h)-tet15.g"))
output mesh file: $out
output interval: 1.0e-6
output:
  stress: false
  internal variables: false
model:
  type: solid mechanics
  volumetric projection: linear
  volumetric form: $form
  material:
    blocks:
      bar: j2 plasticity
    j2 plasticity:
      elastic modulus: 117.0e9
      Poisson's ratio: 0.35
      density: 8930.0
      yield stress: 400.0e6
      hardening modulus: 100.0e6
time integrator:
  type: central difference
  initial time: 0.0
  final time: 8.0e-5
  time step: 1.0e-8
  cfl: 0.8
  stable time step interval: 10
  stable time step method: global
  stable time step eigenvalue interval: 200
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

function median_time(f, sync, reps)
    f(); sync()
    ts = Float64[]
    for _ in 1:reps
        sync(); t = time_ns(); f(); sync()
        push!(ts, (time_ns() - t) / 1e9)
    end
    return sort(ts)[(length(ts) + 1) ÷ 2]
end

function main(args)
    opts = parse_args(args)
    h = parse(Float64, opts["--h"]); form = opts["--form"]
    warm = parse(Int, opts["--warm"]); reps = parse(Int, opts["--reps"])
    backend = backend_of(opts["--device"])
    sync = backend isa Carina.KA.CPU ? (() -> nothing) : (() -> Carina.KA.synchronize(backend))
    dir = mktempdir()
    path = joinpath(dir, "taylor.yaml")
    open(io -> write(io, deck(h, form, joinpath(dir, "taylor.e"))), path, "w")
    dict = Carina.YAML.load_file(path; dicttype = Dict{String, Any})
    sim = Carina.create_simulation(dict, dir; backend = backend)
    ig = sim.integrator; p = sim.params; asm = ig.asm
    step!() = (p.times.Δt = ig.time_step; Carina._advance_one_step!(sim))
    # The warm-up steps the integrator directly, so the collection valve of
    # the time loop (simulation.jl, _DEVICE_GC_INTERVAL) is applied here too:
    # on a device the temporaries of the steps are otherwise never released.
    on_device = !(backend isa Carina.KA.CPU)
    for k in 1:warm
        step!()
        on_device && k % Carina._DEVICE_GC_INTERVAL == 0 && GC.gc(false)
    end
    Carina._pre_step_hook!(ig, sim)          # allocates the stable-step storage
    sync()
    U = ig.stable_dt_U
    copyto!(U, view(ig.U, asm.dof.unknown_dofs))
    # The state field is flat: NS values per quadrature point; eqps is the
    # tenth state variable of the J2 model.
    ns = Carina.CM.num_state_variables(first(values(p.physics)).constitutive_model)
    eqps = maximum(reshape(Array(p.state_new.data), ns, :)[10, :])

    t_res  = median_time(() -> FEC.assemble_vector!(asm, FEC.residual, U, p), sync, reps)
    t_step = median_time(step!, sync, reps)
    t_el   = median_time(() -> Carina._compute_stable_dt(asm, p, 1.0, U; storage = ig.stable_dt_storage,
                                                           wave_speeds = ig.stable_dt_wave_speeds), sync, reps)
    # The eigenvalue estimate: the median of five, each preceded by a
    # collection on a device, which the time includes.
    # A single estimate timed right after the warm-up carried an overhead the
    # estimates inside a run do not show (on the H100, 12.8 ms per power
    # iteration against about 1.2 ms in the runs), and each estimate runs
    # about 20 iterations from the previous eigenvector, as in a run.
    t_gl = median_time(() -> begin
                           on_device && GC.gc(false)
                           Carina._global_stable_dt!(ig, p, 1.0)
                       end, sync, 5)
    # One step of a run: the step, the element estimate every 10 steps and the
    # global estimate every 200.
    t_run_step = t_step + t_el / 10 + t_gl / 200

    e = ExodusDatabase(joinpath(DIR, "meshes", "taylor-h$(h)-tet15.g"), "r")
    nelem = sum(size(b.conn, 2) for b in read_sets(e, Block)); close(e)
    rec = (; host = gethostname(), date = string(now()), tag = opts["--tag"],
           device = opts["--device"], backend = string(typeof(backend)),
           threads = Threads.nthreads(), julia = string(VERSION),
           carina = strip(read(`git -C $DIR rev-parse --short HEAD`, String)),
           h, form, elements = nelem, dofs = length(asm.dof), free_dofs = length(U),
           warm_steps = warm, max_eqps = eqps, reps,
           residual_s = t_res, step_s = t_step, element_dt_s = t_el, global_dt_s = t_gl,
           run_step_s = t_run_step)
    json(v::AbstractString) = "\"" * replace(v, "\"" => "\\\"") * "\""
    json(v::Integer) = string(v)
    json(v::AbstractFloat) = @sprintf("%.6e", v)
    line = "{" * join(("\"$k\": " * json(v) for (k, v) in pairs(rec)), ", ") * "}"
    open(io -> println(io, line), joinpath(DIR, "kernel-timing.jsonl"), "a")
    @printf("%s  %s  %d threads  h = %g mm  %s form  %d elements  %d unknowns  max eqps %.4f\n",
            rec.host, rec.device, rec.threads, h, form, nelem, rec.dofs, eqps)
    @printf("  residual        %8.2f ms\n", 1e3 * t_res)
    @printf("  explicit step   %8.2f ms\n", 1e3 * t_step)
    @printf("  element dt      %8.2f ms   (every 10 steps)\n", 1e3 * t_el)
    @printf("  global dt       %8.2f ms   (every 200 steps)\n", 1e3 * t_gl)
    @printf("  per run step    %8.2f ms   (%.3f us per element-step)\n", 1e3 * t_run_step,
            1e6 * t_run_step / nelem)
end

main(ARGS)
