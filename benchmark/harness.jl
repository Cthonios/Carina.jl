# Benchmark harness for the native-Julia GPU solver campaign (~/Carina-GPU.md).
#
# Runs one (case, variant, backend) combination per process — GPU state, JIT
# caches, and the assembly-flag globals make in-process repetition unreliable,
# and a fresh process is the only clean allocation/VRAM baseline.  Results are
# appended as JSON lines to benchmark/results/<tag>.jsonl.
#
# Usage:
#   julia --project=. benchmark/harness.jl <case> <variant> [tag] [device]
#
#   case:    torsion-newmark | torsion-qs | cube-newmark | cube-qs
#   variant: gpu-cg-jacobi | gpu-cg-chebyshev | gpu-cg-amg | gpu-lbfgs |
#            cpu-direct | cpu-cg-jacobi | cpu-cg-ic | cpu-cg-amg | cpu-lbfgs
#   device:  auto (default) | rocm | cuda -- only consulted for a gpu-* variant
#
# The harness ran ROCm-only until 2026-09-11, which is why every gpu-* record
# in results/ before then is a Radeon RX 7600.  Carina itself is vendor-free;
# the lock was here.
#
# Iteration counts are parsed from the Carina log file (the [SOLVE] lines),
# which keeps the harness decoupled from solver internals.

using Printf

# Minimal JSON emission for flat records (numbers, strings, bools, int arrays) —
# not worth a dependency.
_json(x::Union{Real, Bool}) = string(x)
_json(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
_json(x::AbstractVector) = "[" * join(map(_json, x), ",") * "]"
_json(nt::NamedTuple) =
    "{" * join(("\"$(k)\":" * _json(v) for (k, v) in pairs(nt)), ",") * "}"

# Case, mesh, and solver-variant definitions are shared with write_inputs.jl.
include(joinpath(@__DIR__, "cases.jl"))

# --------------------------------------------------------------------------- #
# Log parsing: Newton and CG iteration counts from the Carina log file
# --------------------------------------------------------------------------- #

function parse_log(logpath::String)
    newton_iters = Int[]     # per nonlinear solve: iteration count reached
    cg_iters     = Int[]     # per linear solve: CG iteration count
    step_walls   = Float64[] # per time step: wall seconds ([STOP] lines)
    t_solve_sum  = 0.0       # linear-solve seconds (logged when > 0.01s)
    t_eval_sum   = 0.0       # residual-evaluation seconds (same threshold)
    amg_build_s  = 0.0       # hierarchy (re)build seconds
    nbuilds      = 0         # hierarchy build count
    current_last = 0
    for line in eachline(logpath)
        m = match(r"Iter \[(\d+)\]", line)
        if m !== nothing
            it = parse(Int, m.captures[1])
            it == 0 && current_last > 0 &&
                (push!(newton_iters, current_last); current_last = 0)
            current_last = max(current_last, it)
        end
        mcg = match(r"CG: (\d+) iters", line)
        mcg !== nothing && push!(cg_iters, parse(Int, mcg.captures[1]))
        ms = match(r"t_solve = ([0-9.]+)s", line)
        ms !== nothing && (t_solve_sum += parse(Float64, ms.captures[1]))
        me = match(r"t_eval = ([0-9.]+)s", line)
        me !== nothing && (t_eval_sum += parse(Float64, me.captures[1]))
        mw = match(r"wall = ([0-9.]+)s", line)
        mw !== nothing && push!(step_walls, parse(Float64, mw.captures[1]))
        mb = match(r"hierarchy build #(\d+) \(([0-9.]+)s\)", line)
        if mb !== nothing
            amg_build_s += parse(Float64, mb.captures[2])
            nbuilds = max(nbuilds, parse(Int, mb.captures[1]))
        end
    end
    current_last > 0 && push!(newton_iters, current_last)
    return newton_iters, cg_iters, step_walls, t_solve_sum, t_eval_sum,
           amg_build_s, nbuilds
end

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

# --- top-level flow (include must not happen inside a function: world age) ---

length(ARGS) >= 2 || error("Usage: harness.jl <case> <variant> [tag]")
const case, variant = ARGS[1], ARGS[2]
const tag = length(ARGS) >= 3 ? ARGS[3] : "baseline"

const use_gpu = startswith(variant, "gpu-")
const device  = length(ARGS) >= 4 ? lowercase(ARGS[4]) : "auto"
device in ("auto", "rocm", "cuda") ||
    error("Unknown device \"$device\". Expected auto, rocm, or cuda.")

# Load only the vendor package that was asked for.  Each `using` is its own
# top-level statement: loading a package and then calling into it inside a
# single expression puts the call in the world age that preceded the load, and
# the resulting failure is indistinguishable from the package being absent.
# The load errors are reported rather than swallowed for the same reason -- a
# broken install must not present itself as "no GPU found".
if use_gpu && device in ("auto", "rocm")
    try
        @eval using AMDGPU
        include(joinpath(@__DIR__, "..", "bin", "rocm_workgroup_bound.jl"))
    catch err
        @warn "AMDGPU failed to load" exception = err
    end
end
if use_gpu && device in ("auto", "cuda")
    try
        @eval using CUDA
    catch err
        @warn "CUDA failed to load" exception = err
    end
end

# Which vendor actually resolved.  This is the single decision point for the
# backend and the VRAM query below.  An unusable request is an error rather
# than a silent fallback to the CPU, which would report a GPU variant's
# timings as if the device had run them.
const gpu_vendor = let v = :none
    if use_gpu
        if isdefined(Main, :AMDGPU) && AMDGPU.functional()
            v = :rocm
        elseif isdefined(Main, :CUDA) && CUDA.functional()
            v = :cuda
        else
            error("GPU variant requested but no functional " *
                  (device == "auto" ? "GPU" : device) * " was found.")
        end
    end
    v
end

include(joinpath(REPO, "src", "Carina.jl"))
const C = Carina

# Vendor-specific calls live in these two functions and nowhere else.  Both are
# ordinary function calls, resolved at run time, so the branch not taken need
# not have its package loaded -- a vendor *macro* here would be resolved when
# this file is lowered and would fail on any machine missing that package.
gpu_backend() = gpu_vendor === :rocm ? AMDGPU.ROCBackend() :
                gpu_vendor === :cuda ? CUDA.CUDABackend() : C.KA.CPU()

# Bytes currently held by the vendor allocator.  `AMDGPU.memory_stats().live`
# and `CUDA.used_memory()` are the same quantity: pool memory in use by this
# process, not total device occupancy.
gpu_live_bytes() = gpu_vendor === :rocm ? Int(AMDGPU.memory_stats().live) :
                   gpu_vendor === :cuda ? Int(CUDA.used_memory()) : 0

function main()
    meshname, meshpath, case_body = case_yaml(case)
    isfile(meshpath) || error(
        "$meshname missing — generate it with benchmark/meshgen.jl " *
        "(see benchmark/README.md)")
    yaml = "type: single\ninput mesh file: $meshname\n" *
           "output mesh file: bench_out.e\n" * case_body * variant_yaml(variant)

    resdir = joinpath(REPO, "benchmark", "results")
    mkpath(resdir)

    record = mktempdir() do dir
        # Example meshes are relative symlinks into examples/meshes; a plain cp
        # copies the dangling link, so follow it.
        cp(meshpath, joinpath(dir, meshname); follow_symlinks=true)
        path = joinpath(dir, "bench.yaml")
        write(path, yaml)

        backend = gpu_backend()

        C.CARINA_WRITE_LOG_FILE[] = true
        gc_before = Base.gc_num()
        t_total = @elapsed sim = C.run(path; backend=backend)
        gc_after = Base.gc_num()

        newton_iters, cg_iters, step_walls, t_solve_sum, t_eval_sum,
            amg_build_s, nbuilds = parse_log(joinpath(dir, "bench.log"))

        vram_live = gpu_live_bytes()
        n_dofs = length(sim.asm_cpu.dof.unknown_dofs)
        alloc_bytes = Base.GC_Diff(gc_after, gc_before).allocd

        (; case, variant, tag,
           n_dofs,
           t_total_s   = t_total,
           step_wall_s = step_walls,
           t_solve_s   = t_solve_sum,
           t_eval_s    = t_eval_sum,
           amg_build_s,
           nbuilds,
           newton_iters,
           cg_iters,
           cg_total    = sum(cg_iters; init=0),
           vram_live_bytes = vram_live,
           cpu_alloc_bytes = alloc_bytes,
           failed = sim.integrator.failed[],
           # Which vendor and machine produced the row.  Records written before
           # 2026-09-11 have neither and are all RX 7600 / ROCm.
           device = String(gpu_vendor),
           host = gethostname(),
           julia = string(VERSION),
           timestamp = string(round(Int, time())))
    end

    out = joinpath(resdir, "$(tag).jsonl")
    open(out, "a") do io
        println(io, _json(record))
    end
    @printf("[BENCH] %s %s: total %.2fs, %d Newton solves, %d CG iters, VRAM %.3f GB\n",
            case, variant, record.t_total_s, length(record.newton_iters),
            record.cg_total, record.vram_live_bytes / 1e9)
    println("[BENCH] appended to $out")
end

main()
