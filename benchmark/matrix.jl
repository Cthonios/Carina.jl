# The performance matrix: one record shape for every regime and every device.
#
#   julia --project=. benchmark/matrix.jl [--device auto|cpu|rocm|cuda]
#                                         [--part spine|ladder|size|all]
#                                         [--dry-run]
#
# Why this exists.  Carina had three measurement harnesses -- harness.jl for
# the implicit cases, explicit_sweep.jl for the explicit ladder, and
# crosscode/run_crosscode.py for the cross-code comparison -- each writing a
# different record shape, and the file name (the "tag") carried the only hint
# of what had been measured.  Tags were experiment names from optimization
# rounds (`jvp`, `fp32-csr`, `bisect`), not a device-by-regime matrix, and 80
# of the 127 records in results/ named neither the host nor the vendor.  It
# was not possible to answer "how do these GPUs compare across the three
# regimes" from the recorded data.
#
# This driver does not re-implement any measurement.  It runs the existing
# harnesses, one point per fresh process as they require, and normalizes what
# they emit into a single schema with mandatory provenance.  The measurement
# code stays where it is and stays tested.
#
# Output: benchmark/results/matrix/<host>-<gpu slug>.jsonl, appended.  Rows are
# self-describing, so the files can be concatenated and grouped by any field.

using Printf

const REPO    = dirname(@__DIR__)
const OUTDIR  = joinpath(REPO, "benchmark", "results", "matrix")
const TMPTAG  = "_matrix_scratch"
const SCHEMA  = "carina-matrix/1"

# --------------------------------------------------------------------------
# What the matrix contains
# --------------------------------------------------------------------------
# SPINE: one mesh (torsion.g, 530,523 DOF) through all three regimes, so a
# device's three numbers differ only by the time integrator and are comparable
# to each other as well as across devices.
const SPINE_GPU = [("qs", "torsion-qs", "gpu-cg-amg"),
                   ("qs", "torsion-qs", "gpu-cg-jacobi"),
                   ("newmark", "torsion-newmark", "gpu-cg-jacobi"),
                   ("newmark", "torsion-newmark", "gpu-cg-amg")]
const SPINE_CPU = [("qs", "torsion-qs", "cpu-cg-amg"),
                   ("newmark", "torsion-newmark", "cpu-cg-jacobi")]

# LADDER: explicit only, CFL held fixed so cost per step is the only thing
# that varies with refinement.  nsteps keeps the measured interval near ten
# seconds of CPU time at each size.
const LADDER = [(8, 8000), (12, 3000), (20, 800), (28, 300),
                (36, 160), (44, 100), (50, 80), (64, 40)]

# SIZE SCALING: the implicit analogue of the ladder.  A failing point is
# recorded as a row with ok = false rather than skipped, so a blank in the
# table is a measured limit and not a run nobody attempted.
#
# cube80 is the memory ceiling, and it is the HOST that runs out first, not the
# device: benchmark_report.md section 4 records AlgebraicMultigrid's setup
# being OOM-killed at 1.57M DOF because it preallocates for the worst case.  On
# a 60 GB desktop the kernel can pick this driver as the victim rather than the
# child, which is why records are appended per point above -- otherwise the
# whole sweep is lost at the last step.
const SIZE_GPU = [("qs", "cube64-qs", "gpu-cg-amg"),
                  ("qs", "cube80-qs", "gpu-cg-amg")]

# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
_git(args...) = try
        strip(read(`git -C $REPO $(collect(args))`, String))
    catch
        ""
    end

"""
Vendor and device label for `dev`, read through the launcher environment.

The vendor packages live in bin/, not in the main project this driver runs
under, so the query goes out to a short-lived `--project=bin` process rather
than importing them here.  The AMD name is parsed out of the device's own
printed form (`HIPDevice(id=1, name=..., gcn_arch=...)`): AMDGPU exposes it
there but not through an accessor that has been stable across versions.  A
label is provenance, not a measurement, so degrading to an empty string is
acceptable -- `device` still records the vendor.
"""
function gpu_label(dev::String)
    dev == "cpu" && return ""
    prog = dev == "cuda" ?
        "import CUDA; print(CUDA.name(CUDA.device()))" :
        """import AMDGPU
           s = string(AMDGPU.device())
           m = match(r"name=([^,)]+)", s)
           print(m === nothing ? s : m.captures[1])"""
    try
        out = read(`julia --project=$(joinpath(REPO, "bin")) -e $prog`, String)
        return strip(out)
    catch err
        @warn "could not read the device label; provenance falls back to the vendor name" dev exception = err
        return ""
    end
end

slug(s) = lowercase(replace(strip(s), r"[^A-Za-z0-9]+" => "-"))

"""
Bytes of device memory in use, from the vendor's own tool, or `nothing`.

This is recorded before every point so that "the device was clean when this
started" is a measured fact rather than an assumption.  It matters because
Julia's garbage collector triggers on host pressure and never sees VRAM, so a
finished simulation's device memory stays live until something fails to
allocate.  In a single process the remedy is explicit -- `sim = nothing;
GC.gc(true); AMDGPU.reclaim()` -- and this driver instead runs every point in
its own process and waits for it to exit, which returns the device to the
driver.  That is also why harness.jl mandates a fresh process per point.

The failure mode this guards against is subtle: leaked VRAM surfaces at
whichever later point needs the most memory, and the memory-hungry cases run
last here.  A `cube80` failure with a dirty device is a leak; the same failure
with a clean one is the hardware's limit.  Without this field the two are
indistinguishable after the fact.
"""
function device_vram_used(dev::String)
    dev == "cpu" && return nothing
    try
        if dev == "cuda"
            # stderr is discarded on both paths: rocm-smi warns about the
            # device being in a low-power state, and an inherited stderr lands
            # in the middle of this driver's own formatted output line.
            out = read(pipeline(`nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`,
                                stderr = devnull), String)
            return round(Int, parse(Float64, strip(first(split(strip(out), '\n')))) * 2^20)
        else
            # rocm-smi lists every card including the integrated one; card0 is
            # the discrete device the runs use.
            for l in split(read(pipeline(`rocm-smi --showmeminfo vram --csv`,
                                         stderr = devnull), String), '\n')
                startswith(l, "card0,") || continue
                f = split(strip(l), ',')
                length(f) >= 3 && return parse(Int, strip(f[3]))
            end
        end
    catch err
        @warn "could not read device memory use" dev exception = err
    end
    return nothing
end

# --------------------------------------------------------------------------
# Minimal JSON, matching the emitters in harness.jl and explicit_sweep.jl so
# the three families stay readable by the same tooling without a dependency.
# --------------------------------------------------------------------------
_json(x::Union{Real, Bool}) = string(x)
_json(::Nothing) = "null"
_json(x::AbstractString) = "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"") * "\""
_json(x::AbstractVector) = "[" * join(map(_json, x), ",") * "]"
_json(nt::NamedTuple) =
    "{" * join(("\"$(k)\":" * _json(v) for (k, v) in pairs(nt)), ",") * "}"

# Read the last record a harness appended, without a JSON dependency: the
# harnesses emit one flat object per line, so the fields this driver needs can
# be pulled out directly.
function last_record(path::String)
    isfile(path) || return nothing
    line = ""
    for l in eachline(path)
        isempty(strip(l)) || (line = l)
    end
    isempty(line) && return nothing
    return line
end

function field(line::String, key::String)
    m = match(Regex("\"" * key * "\":(\\[[^\\]]*\\]|\"[^\"]*\"|[^,}]+)"), line)
    m === nothing && return nothing
    v = strip(String(m.captures[1]))
    startswith(v, "\"") && return String(strip(v, '"'))
    v == "null" && return nothing
    v == "true" && return true
    v == "false" && return false
    if startswith(v, "[")
        inner = strip(v, ['[', ']'])
        isempty(inner) && return Float64[]
        return [parse(Float64, strip(x)) for x in split(inner, ",")]
    end
    return something(tryparse(Float64, v), v)
end

"Median of the steps after the first: the first absorbs warm-up in every regime."
function steady(walls)
    walls === nothing && return nothing
    w = walls isa AbstractVector ? Float64.(walls) : Float64[]
    length(w) <= 1 && return isempty(w) ? nothing : w[end]
    s = sort(w[2:end])
    n = length(s)
    return isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
end

# --------------------------------------------------------------------------
# Running one point
# --------------------------------------------------------------------------
function run_point(regime, case, variant, device, gpu, threads, dry)
    tmp = joinpath(REPO, "benchmark", "results", "$(TMPTAG).jsonl")
    rm(tmp; force = true)

    if regime == "explicit"
        N, nsteps = case, variant           # reused positionally for the ladder
        cmd = `julia --project=$REPO $(joinpath(REPO, "benchmark", "explicit_sweep.jl"))
               $N $device $TMPTAG $nsteps $threads`
        label = "explicit N=$N"
    else
        # harness.jl needs the vendor packages, which live in the launcher
        # environment; JULIA_LOAD_PATH is how its own README invokes it.
        cmd = addenv(`julia $(joinpath(REPO, "benchmark", "harness.jl"))
                      $case $variant $TMPTAG $(device == "cpu" ? "auto" : device)`,
                     "JULIA_LOAD_PATH" => "$REPO:$(joinpath(REPO, "bin")):@stdlib",
                     "JULIA_NUM_THREADS" => string(threads))
        label = "$case $variant"
    end

    @printf("  [%-8s] %-34s ", regime, label)
    if dry
        println("(dry run)")
        return nothing
    end
    flush(stdout)
    vram_before = device_vram_used(device)

    t0 = time()
    ok = try
        success(pipeline(cmd; stdout = devnull, stderr = devnull))
    catch
        false
    end
    line = last_record(tmp)
    rm(tmp; force = true)

    if !ok || line === nothing
        @printf("FAILED after %.0fs\n", time() - t0)
        return (; schema = SCHEMA, regime,
                  case = regime == "explicit" ? "explicit-torsion" : String(case),
                  variant = regime == "explicit" ? "central-difference" : String(variant),
                  n_dofs = nothing, n_dofs_kind = regime == "explicit" ? "total" : "free",
                  device, gpu,
                  host = gethostname(), commit = _git("rev-parse", "--short", "HEAD"),
                  julia = string(VERSION), threads,
                  t_total_s = nothing, steady_step_s = nothing, per_step_ms = nothing,
                  step_walls = Float64[], newton_iters = nothing, cg_total = nothing,
                  amg_build_s = nothing, vram_live_bytes = nothing,
                  vram_used_before_bytes = vram_before,
                  ok = false, timestamp = round(Int, time()))
    end

    walls = regime == "explicit" ? field(line, "interval_walls") : field(line, "step_wall_s")
    rec = (; schema = SCHEMA, regime,
             case    = regime == "explicit" ? "explicit-torsion" : String(case),
             variant = regime == "explicit" ? "central-difference" : String(variant),
             # explicit_sweep.jl names it n_dof, harness.jl n_dofs -- and they
             # count different things: the mesh total, and the free degrees of
             # freedom left after boundary-condition elimination.  On the
             # free-free explicit torsion bar the two coincide; on torsion-qs
             # they are 530,523 and 527,877.  Merging them under one name would
             # reproduce inside the matrix exactly the drift it exists to
             # remove, so the count is labelled rather than silently unified.
             n_dofs  = something(field(line, "n_dofs"), field(line, "n_dof"), 0),
             n_dofs_kind = regime == "explicit" ? "total" : "free",
             device, gpu,
             host    = gethostname(),
             commit  = _git("rev-parse", "--short", "HEAD"),
             julia   = string(VERSION),
             threads,
             t_total_s     = field(line, "t_total_s"),
             steady_step_s = regime == "explicit" ? nothing : steady(walls),
             per_step_ms   = field(line, "per_step_ms"),
             step_walls    = walls === nothing ? Float64[] : walls,
             newton_iters  = let v = field(line, "newton_iters")
                                 v isa AbstractVector ? length(v) : v
                             end,
             cg_total      = field(line, "cg_total"),
             amg_build_s   = field(line, "amg_build_s"),
             vram_live_bytes = field(line, "vram_live_bytes"),
             vram_used_before_bytes = vram_before,
             ok = true, timestamp = round(Int, time()))

    if regime == "explicit"
        @printf("%8.2f ms/step\n", something(rec.per_step_ms, NaN))
    else
        @printf("%8.1f s total, %6s CG\n", something(rec.t_total_s, NaN),
                string(something(rec.cg_total, "?")))
    end
    return rec
end

# --------------------------------------------------------------------------
function main()
    device  = "auto"
    part    = "all"
    dry     = false
    threads = min(Sys.CPU_THREADS, 48)
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--device";  device  = lowercase(ARGS[i + 1]); i += 2
        elseif a == "--part";    part    = lowercase(ARGS[i + 1]); i += 2
        elseif a == "--threads"; threads = parse(Int, ARGS[i + 1]); i += 2
        elseif a == "--dry-run"; dry = true; i += 1
        else
            error("Unknown argument \"$a\". Expected --device, --part, --threads or --dry-run.")
        end
    end
    device in ("auto", "cpu", "rocm", "cuda") ||
        error("Unknown device \"$device\". Expected auto, cpu, rocm or cuda.")
    part in ("spine", "ladder", "size", "all") ||
        error("Unknown part \"$part\". Expected spine, ladder, size or all.")

    if device == "auto"
        device = Sys.which("nvidia-smi") !== nothing ? "cuda" :
                 Sys.which("rocm-smi")  !== nothing ? "rocm"  : "cpu"
    end
    gpu = gpu_label(device)

    mkpath(OUTDIR)
    name = device == "cpu" ? "$(gethostname())-cpu" :
           "$(gethostname())-$(isempty(gpu) ? device : slug(gpu))"
    out = joinpath(OUTDIR, "$name.jsonl")

    println("Carina performance matrix")
    println("  host   : ", gethostname())
    println("  device : ", device, isempty(gpu) ? "" : "  ($gpu)")
    # Tracked modifications only.  An untracked stray -- a results file from an
    # ad-hoc run, an editor backup -- cannot change what the code does, and
    # flagging it would teach the reader to ignore the warning.
    println("  commit : ", _git("rev-parse", "--short", "HEAD"),
            isempty(_git("status", "--porcelain", "--untracked-files=no")) ? "" :
            "  (DIRTY -- tracked files modified, results not reproducible)")
    println("  julia  : ", VERSION, "   threads: ", threads)
    println("  part   : ", part)
    println("  out    : ", relpath(out, REPO))
    println()

    # Each record is appended the moment its point finishes.  Buffering the
    # run and writing at the end loses everything if the process dies, and
    # these runs are long, unattended, and end in a case that is expected to
    # exhaust host memory on some machines -- a full sweep was lost exactly
    # that way before this was changed.  Appending also makes a run in flight
    # inspectable from another shell.
    mkpath(dirname(out))
    emit(r) = (r === nothing || open(io -> println(io, _json(r)), out, "a"); r)

    records = Any[]
    if part in ("spine", "all")
        println("SPINE  torsion.g, 530k DOF, three regimes")
        for (regime, case, variant) in (device == "cpu" ? SPINE_CPU : SPINE_GPU)
            r = emit(run_point(regime, case, variant, device, gpu, threads, dry))
            r === nothing || push!(records, r)
        end
        r = emit(run_point("explicit", 20, 800, device, gpu, threads, dry))
        r === nothing || push!(records, r)
        println()
    end
    if part in ("ladder", "all")
        println("LADDER  explicit, fixed CFL, 39k to 16.2M DOF")
        for (N, nsteps) in LADDER
            r = emit(run_point("explicit", N, nsteps, device, gpu, threads, dry))
            r === nothing || push!(records, r)
        end
        println()
    end
    if part in ("size", "all") && device != "cpu"
        println("SIZE  implicit at 823k and 1.57M DOF")
        for (regime, case, variant) in SIZE_GPU
            r = emit(run_point(regime, case, variant, device, gpu, threads, dry))
            r === nothing || push!(records, r)
        end
        println()
    end

    dry && return
    nfail = count(r -> !r.ok, records)
    @printf("%d records appended to %s%s\n", length(records), relpath(out, REPO),
            nfail == 0 ? "" : "  ($nfail failed -- recorded, not skipped)")
end

main()
