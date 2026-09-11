# Splits one preconditioned-CG iteration of the GPU quasi-static AMG path into
# its parts, to decide where an FP32 conversion would actually pay.
#
# §6 of benchmark_report.md measured the matrix-free action at 9.50 ms, and §2
# measured the whole solve phase at 70.5 s over 951 CG iterations (74 ms per
# iteration).  What that leaves unresolved is the attribution of the other
# ~64 ms: it could be the assembled coarse-level SpMVs (bandwidth-bound, so
# FP32 gains ~2x on bytes) or more matrix-free actions (arithmetic-bound, so
# FP32 gains up to ~32x on arithmetic before reaching the memory floor).
#
# The fine level of the V-cycle is matrix-free -- `_amg_vcycle!` passes
# `fine_matvec!` into `_smooth!` for both the pre- and post-smooth and uses it
# once more for the fine residual -- so a V(nu,nu) cycle costs 2*nu+1 actions
# plus the assembled skeleton.  This measures that split directly:
#
#   action        -- one `_stiffness_matvec_qs!`, i.e. CG's own matvec
#   vcycle        -- one full `_amg_vcycle!`
#   vcycle-nofine -- the same V-cycle with the fine matvec stubbed to a zero
#                    fill.  Wrong correction, valid timing; this is the
#                    assembled skeleton (coarse SpMVs, R/P, Jacobi kernels,
#                    dense coarse solve) with every fine-level action removed.
#                    Same ablation methodology as benchmark/evidence/action_ablation.txt.
#
# Also times the two per-Newton costs that sit outside the CG loop: the
# diagonal refresh, and a forced hierarchy rebuild.
#
# Usage:  julia --project=bin benchmark/vcycle_bench.jl [nreps] [auto|rocm|cuda]
#
# Beyond the action/skeleton split, the skeleton itself is broken down:
#
#   smoother kernel -- one `_jacobi_omega_kernel!` on the fine level, the
#                      kernel a block-Jacobi smoother would replace
#   csr level l     -- one SpMV with that level's assembled operator
#   coarse solve    -- the dense pinv GEMV
#
# so that a change to the smoother can be judged against the share it holds,
# and a reduction in iteration count against the cost of the action it saves.

import AMDGPU
import CUDA
using Carina
import Carina: FEC
import KernelAbstractions as KA
using LinearAlgebra
using Printf
using Statistics
using Random

const NREPS  = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 30
const DEVICE = length(ARGS) >= 2 ? lowercase(ARGS[2]) : "auto"
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

const REPO = normpath(joinpath(@__DIR__, ".."))
const DECK = joinpath(REPO, "benchmark", "inputs", "torsion-qs-gpu-cg-amg.yaml")

dict = Carina.YAML.load_file(DECK; dicttype = Dict{String, Any})
dict["input mesh file"]  = joinpath(REPO, "examples", "meshes", "torsion", "torsion.g")
dict["output mesh file"] = tempname() * ".e"
# One load step of the four.  The hierarchy and the fine diagonal are then
# built at a real deformed configuration rather than at U = 0, which is the
# operating point the 951-iteration figure came from.
dict["time integrator"]["final time"] = 0.25

sim = Carina.create_simulation(dict, mktempdir(); backend = GPU_BACKEND)
Carina.evolve!(sim)

ig  = sim.integrator
asm = ig.asm
p   = sim.params
ls  = ig.nonlinear_solver.linear_solver
pc  = ls.precond

pc isa Carina.GPUAMGPreconditioner ||
    error("expected a GPUAMGPreconditioner, got $(typeof(pc))")
pc.hierarchy === nothing && error("AMG hierarchy was never built")

U = ig.U
n = length(U)
backend = KA.get_backend(pc.inv_diag)
h = pc.hierarchy

# CG's matvec, verbatim from `_setup_linear_ops(::QuasiStaticIntegrator, ...)`.
matvec! = (y, v) -> Carina._stiffness_matvec_qs!(y, v, asm, U, p)
# The reduced-precision action the V-cycle smooths with in production.
matvec32! = (y, v) -> Carina._stiffness_matvec_qs_fp32!(y, v, asm, U, p)
# Ablated fine matvec: right shape, no work.  Keeps every assembled level, the
# restriction/prolongation SpMVs and the Jacobi kernels intact.
nofine! = (y, v) -> (fill!(y, 0.0); y)

v = similar(U); copyto!(v, 1.0e-6 .* randn(MersenneTwister(20260808), n))
y = similar(U); fill!(y, 0.0)
r = similar(U); copyto!(r, v)
z = similar(U); fill!(z, 0.0)

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
    @printf("%-22s med %8.3f ms   min %8.3f ms   max %8.3f ms\n",
            label, tmed * 1e3, minimum(ts) * 1e3, maximum(ts) * 1e3)
    return tmed
end

nlev = length(h.levels)
@printf("\n===== V-CYCLE / ACTION SPLIT =====\n")
@printf("free DOF %d, reps %d, nu = %d, assembled levels %d, coarse dim %d\n",
        n, NREPS, h.nu, nlev, length(h.coarse_x))
@printf("fine-level actions per V-cycle = 2*nu + 1 = %d\n\n", 2 * h.nu + 1)

t_act   = timeit(() -> matvec!(y, v), "action (fp64)")
t_a32   = timeit(() -> matvec32!(y, v), "action (fp32)")
t_vcyc  = timeit(() -> Carina._amg_vcycle!(z, r, h, matvec!, backend), "vcycle (fp64 smooth)")
t_v32   = timeit(() -> Carina._amg_vcycle!(z, r, h, matvec32!, backend), "vcycle (fp32 smooth)")
t_skel  = timeit(() -> Carina._amg_vcycle!(z, r, h, nofine!, backend), "vcycle (no fine mv)")
t_diag  = timeit(() -> Carina._update_gpu_amg_precond_qs!(pc, asm, U, p), "precond diag refresh")

# ---- the skeleton, itemized ------------------------------------------------
@printf("\n----- skeleton, itemized (one application each) -----\n")
ω1 = 4.0 / (3.0 * h.lmax1)
t_jac = timeit(() -> Carina._jacobi_omega_kernel!(backend)(z, r, h.inv_d1, ω1; ndrange = n),
               "smoother kernel (fine)")
t_csr = Float64[]
for (l, lev) in enumerate(h.levels)
    push!(t_csr, timeit(() -> Carina._csr_mul!(lev.r, lev.A, lev.x, backend),
                        @sprintf("csr level %d SpMV (n=%d)", l, lev.A.nrows)))
end
nc = length(h.coarse_x)
t_crs = timeit(() -> Carina._dense_mul_kernel!(backend)(h.coarse_x, h.coarse_pinv, h.coarse_b, nc;
                                                        ndrange = nc),
               @sprintf("coarse pinv GEMV (n=%d)", nc))
# Per V-cycle counts: the fine smoother kernel runs 2*nu times; each assembled
# level runs (2*nu + 1) SpMVs of A plus one R and one P; the coarse solve once.
n_jac_per_cycle = 2 * h.nu
t_jac_cycle = n_jac_per_cycle * t_jac
t_csr_cycle = sum((2 * h.nu + 1) * t for t in t_csr)

# Forced rebuild, timed once -- it is lagged, not per-iteration.
pc.rebuild = true
t_build = @elapsed Carina._build_gpu_amg_hierarchy!(pc, 0.0, U)

t_iter64 = t_act + t_vcyc     # exact operator, exact smoother (pre-change)
t_iter32 = t_act + t_v32      # exact operator, reduced-precision smoother (now)
@printf("\n----- derived -----\n")
@printf("hierarchy rebuild (once, host+device)   %8.3f s\n", t_build)
@printf("actions implied by vcycle-skel gap      %8.2f  (= (vcyc - skel)/action)\n",
        (t_vcyc - t_skel) / t_act)
@printf("action speedup from fp32                %8.2fx\n", t_act / t_a32)
@printf("\nper CG iteration, fp64 smoother         %8.3f ms\n", t_iter64 * 1e3)
@printf("  of which matrix-free action           %8.3f ms  (%.1f%%)\n",
        (t_act + t_vcyc - t_skel) * 1e3, 100 * (t_act + t_vcyc - t_skel) / t_iter64)
@printf("  of which assembled skeleton           %8.3f ms  (%.1f%%)\n",
        t_skel * 1e3, 100 * t_skel / t_iter64)
@printf("    fine smoother kernels (%d/cycle)     %8.3f ms  (%.1f%% of iteration)\n",
        n_jac_per_cycle, t_jac_cycle * 1e3, 100 * t_jac_cycle / t_iter64)
@printf("    csr-level A SpMVs (%d/level/cycle)   %8.3f ms  (%.1f%% of iteration)\n",
        2 * h.nu + 1, t_csr_cycle * 1e3, 100 * t_csr_cycle / t_iter64)
@printf("    coarse solve (1/cycle)               %8.3f ms  (%.1f%% of iteration)\n",
        t_crs * 1e3, 100 * t_crs / t_iter64)
@printf("    R/P SpMVs, vector ops, launch gaps   %8.3f ms  (remainder of skeleton)\n",
        (t_skel - t_jac_cycle - t_csr_cycle - t_crs) * 1e3)
@printf("per CG iteration, fp32 smoother         %8.3f ms\n", t_iter32 * 1e3)
@printf("  speedup per iteration                 %8.2fx\n", t_iter64 / t_iter32)
@printf("  remaining assembled-skeleton share    %8.1f%%\n", 100 * t_skel / t_iter32)
@printf("\nreport §2 measured 70.5 s / 951 CG iters = 74.1 ms/iter (fp64 smoother)\n")
@printf("this predicts %.1f ms/iter fp64, %.1f ms/iter fp32; the balance is\n",
        t_iter64 * 1e3, t_iter32 * 1e3)
@printf("CG's own vector ops, the per-Newton diag refresh, and %d hierarchy builds.\n",
        pc.nbuilds)
@printf("==================================\n")
