# Benchmark harness for the native-Julia GPU solver campaign (~/Carina-GPU.md).
#
# Runs one (case, variant, backend) combination per process — GPU state, JIT
# caches, and the assembly-flag globals make in-process repetition unreliable,
# and a fresh process is the only clean allocation/VRAM baseline.  Results are
# appended as JSON lines to benchmark/results/<tag>.jsonl.
#
# Usage:
#   julia --project=. benchmark/harness.jl <case> <variant> [tag]
#
#   case:    torsion-newmark | torsion-qs | cube-newmark | cube-qs
#   variant: gpu-cg-jacobi | gpu-cg-chebyshev | gpu-lbfgs |
#            cpu-direct | cpu-cg-jacobi | cpu-cg-ic | cpu-cg-amg | cpu-lbfgs
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
if use_gpu
    using AMDGPU
    AMDGPU.functional() || error("GPU variant requested but ROCm not functional.")
end

include(joinpath(REPO, "src", "Carina.jl"))
const C = Carina

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

        backend = use_gpu ? AMDGPU.ROCBackend() : C.KA.CPU()

        C.CARINA_WRITE_LOG_FILE[] = true
        gc_before = Base.gc_num()
        t_total = @elapsed sim = C.run(path; backend=backend)
        gc_after = Base.gc_num()

        newton_iters, cg_iters, step_walls, t_solve_sum, t_eval_sum,
            amg_build_s, nbuilds = parse_log(joinpath(dir, "bench.log"))

        vram_live = use_gpu ? Int(AMDGPU.memory_stats().live) : 0
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
