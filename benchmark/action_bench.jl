# Follow-on to action_bench.jl.  The atomic ablation came back flat, so this
# separates the two remaining explanations for the action's cost:
#
#   stiffness_action -- gather + deformation gradient + constitutive tangent
#                       + B^T C B v, then scatter
#   mass_action      -- gather + N^T N v, then scatter
#
# Identical connectivity gather, identical 24-DOF scatter, identical element
# count and launch geometry.  The ONLY difference is arithmetic per element.
# If mass is much cheaper, the kernel is FP64-compute-bound (this part runs
# FP64 at 1/32 rate) and the bandwidth roofline in benchmark_report.md §4 is
# the wrong yardstick.  If the two are close, the cost is in the memory
# movement and the bandwidth framing stands.
#
# Usage:  julia --project=bin benchmark/action_bench.jl [nreps] [state] [device]
#
#   nreps   repetitions of each timed action (default 50)
#   state   `initial` (U = 0) or `deformed` (after one load step)
#   device  `auto` (default), `rocm`, or `cuda`
#
# The benchmark is vendor-neutral like the rest of the suite: everything below
# the backend resolution goes through KernelAbstractions.  It ran only on ROCm
# originally, which is why the FP64 reading in benchmark_report.md section 6 is
# a single-vendor result; the L4 gives it a second data point on a card whose
# FP64 rate differs from its FP32 rate by a different factor.

import AMDGPU
import CUDA
include(joinpath(@__DIR__, "..", "bin", "rocm_workgroup_bound.jl"))
using Carina
import Carina: FEC
import KernelAbstractions as KA
using Printf
using Statistics
using Random

const NREPS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 50

# `initial` (default) evaluates the action at U = 0.  `deformed` first solves one
# load step, so the action is taken at finite strain -- where the geometric part
# of ∂P/∂∇u is nonzero.  At U = 0 that part vanishes identically, so an
# `initial` checksum cannot distinguish two kernels that differ only there;
# use `deformed` for any correctness claim about the operator.
const STATE = length(ARGS) >= 2 ? ARGS[2] : "initial"
STATE in ("initial", "deformed") ||
    error("state must be \"initial\" or \"deformed\", got \"$STATE\"")

# Backend resolution, matching bin/carina.jl.  An unknown name is an error
# rather than a silent fall back to the CPU, which would report timings that
# look like a very slow GPU.
const DEVICE = length(ARGS) >= 3 ? lowercase(ARGS[3]) : "auto"
function _resolve(dev)
    if dev == "rocm"
        AMDGPU.functional() || error("device rocm: no functional AMD GPU found.")
        return AMDGPU.ROCBackend()
    elseif dev == "cuda"
        CUDA.functional() || error("device cuda: no functional NVIDIA GPU found.")
        return CUDA.CUDABackend()
    elseif dev == "auto"
        AMDGPU.functional() && return AMDGPU.ROCBackend()
        CUDA.functional() && return CUDA.CUDABackend()
        error("no functional GPU found; this benchmark is GPU-only.")
    end
    return error("Unknown device \"$dev\". Expected auto, rocm, or cuda.")
end
const GPU_BACKEND = _resolve(DEVICE)
@info "action_bench" nreps=NREPS state=STATE device=DEVICE backend=GPU_BACKEND

const REPO  = normpath(joinpath(@__DIR__, ".."))
const DECK  = joinpath(REPO, "benchmark", "inputs", "torsion-qs-gpu-cg-jacobi.yaml")

dict = Carina.YAML.load_file(DECK; dicttype = Dict{String, Any})
dict["input mesh file"]  = joinpath(REPO, "examples", "meshes", "torsion", "torsion.g")
dict["output mesh file"] = tempname() * ".e"
# One load step is enough to leave U = 0; the full 4-step ramp costs minutes.
STATE == "deformed" && (dict["time integrator"]["final time"] = 0.25)

sim = Carina.create_simulation(dict, mktempdir(); backend = GPU_BACKEND)
STATE == "deformed" && Carina.evolve!(sim)

ig, asm, p = sim.integrator, sim.integrator.asm, sim.params
U = ig.U; n = length(U)
fspace  = FEC.function_space(asm.dof)
nblocks = length(fspace.ref_fes)
# Element counts come from the connectivity, not the state field.  `Connectivity`
# keeps its block metadata on the host across `adapt`, whereas `StateVariableField`
# moves `nelems`/`nepes`/`offsets` to the device -- so `block_view(p.state_old, b)`
# scalar-indexes GPU arrays here and throws.
nelems  = sum(fspace.elem_conns.nelems[b] for b in 1:nblocks)
backend = KA.get_backend(asm.stiffness_action_storage.data)

v = similar(U)
copyto!(v, 1.0e-6 .* randn(MersenneTwister(20260806), n))

function timeit(action, label)
    for _ in 1:5
        action()
    end
    KA.synchronize(backend)
    ts = Float64[]
    for _ in 1:NREPS
        t0 = time_ns()
        action()
        KA.synchronize(backend)
        push!(ts, (time_ns() - t0) * 1e-9)
    end
    tmed = median(ts)
    chk  = sum(abs, Array(asm.stiffness_action_storage.data))
    # Full precision, not %.6e: `sum(abs, ·)` over 5.3e5 entries compresses a
    # localized discrepancy hard, so 7 significant digits leaves only a
    # factor-of-a-few margin when this is used to compare two kernels.
    @printf("%-18s med %8.3f ms   min %8.3f ms   %8.2f ns/elem   %6.2f GB/s   chk %.15e\n",
            label, tmed * 1e3, minimum(ts) * 1e3,
            tmed * 1e9 / nelems, 856.0 * nelems / tmed / 1e9, chk)
    return tmed
end

@printf("\n===== ACTION COMPARISON =====\n")
@printf("elements %d, free DOF %d, reps %d, state %s (|U|_max = %.3e)\n\n",
        nelems, n, NREPS, STATE, maximum(abs, Array(U)))

t_stiff = timeit(() -> FEC.assemble_matrix_free_action!(asm, FEC.stiffness_action, U, v, p),
                 "stiffness_action")
t_mass  = timeit(() -> FEC.assemble_matrix_free_action!(asm, FEC.mass_action, U, v, p),
                 "mass_action")

@printf("\nstiffness / mass  = %.2fx\n", t_stiff / t_mass)
@printf("arithmetic-attributable share of stiffness = %.1f%%\n",
        100 * (t_stiff - t_mass) / t_stiff)
@printf("=============================\n")
