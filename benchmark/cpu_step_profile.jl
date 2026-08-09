# Phase profile of one CPU implicit (Newmark) step, at a given thread count.
#
# benchmark/crosscode/README.md establishes that the CPU implicit path scales
# only 1.25x over 24 threads, and that the serial assembled SpMV is ~48% of the
# step -- the largest single serial block, but not enough on its own to explain
# a 1.25x ceiling.  Amdahl with 48% serial allows ~2.0x; something else is
# serial too.  This finds it.
#
# The step's own log already gives the coarse split.  For torsion-newmark at 24
# threads: 19.50 s wall, of which 3 Newton solves report t_solve = 4.13 + 4.44
# + 4.30 = 12.87 s and t_eval ~ 0, leaving ~6.6 s in `setup_jacobian!`.  What
# the log cannot separate is (a) how the 12.87 s of CG divides between the
# SpMV, the preconditioner and the vector operations, and (b) which of those
# phases actually thread.  Both are measured here by timing each phase directly
# at whatever thread count the process was started with.
#
# Run at two thread counts and compare to get the parallel fraction per phase:
#
#   julia --project=bin -t 1  benchmark/cpu_step_profile.jl
#   julia --project=bin -t 24 benchmark/cpu_step_profile.jl

using Carina
import Carina: FEC
import KernelAbstractions as KA
using LinearAlgebra
using SparseArrays
using Printf
using Statistics
using Random

const NREPS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
const REPO  = normpath(joinpath(@__DIR__, ".."))
const DECK  = joinpath(REPO, "benchmark", "inputs",
                       "torsion-newmark-cpu-cg-jacobi.yaml")
# CG iterations per step, from the run log ([SOLVE] CG: lines, 202+225+217).
const CG_PER_STEP = 644

dict = Carina.YAML.load_file(DECK; dicttype = Dict{String, Any})
dict["input mesh file"]  = joinpath(REPO, "examples", "meshes", "torsion", "torsion.g")
dict["output mesh file"] = tempname() * ".e"
dict["time integrator"]["final time"] = dict["time integrator"]["time step"]

sim = Carina.create_simulation(dict, mktempdir(); backend = KA.CPU())
ig  = sim.integrator; asm = ig.asm; p = sim.params
ls  = ig.nonlinear_solver.linear_solver

Carina.predict!(ig, p)
Carina.evaluate!(ig, p)
Carina.setup_jacobian!(ig, p)

Uu = Carina._displacement(ig); c_M = ig.c_M; n = length(Uu)
K = FEC.stiffness(asm)

v = similar(Uu); copyto!(v, randn(MersenneTwister(20260809), n))
w = similar(v); copyto!(w, randn(MersenneTwister(20260810), n))
y = similar(v); fill!(y, 0.0)

function timeit(f, label, results)
    for _ in 1:3; f(); end
    ts = Float64[]
    for _ in 1:NREPS
        t0 = time_ns(); f(); push!(ts, (time_ns() - t0) * 1e-9)
    end
    t = median(ts)
    push!(results, (label, t))
    @printf("  %-28s %9.2f ms\n", label, t * 1e3)
    return t
end

res = Tuple{String, Float64}[]
@printf("\n===== CPU STEP PHASE PROFILE =====\n")
@printf("free DOF %d, nnz %d, threads %d, reps %d\n\n",
        n, nnz(K), Threads.nthreads(), NREPS)

@printf("per Newton iteration (assembly side):\n")
t_K = timeit(() -> FEC.assemble_stiffness!(asm, FEC.stiffness, Uu, p),
             "assemble_stiffness!", res)
t_M = timeit(() -> FEC.assemble_mass!(asm, FEC.mass, Uu, p),
             "assemble_mass!", res)
t_add = timeit(() -> (@. asm.stiffness_storage += c_M * asm.mass_storage),
               "K_eff = K + c_M*M", res)
t_pre = timeit(() -> Carina._update_jacobi_precond_assembled!(ls.precond, K),
               "jacobi precond update", res)
t_res = timeit(() -> Carina.evaluate!(ig, p), "residual (evaluate!)", res)

@printf("\nper CG iteration:\n")
t_spmv = timeit(() -> mul!(y, K, v), "assembled SpMV", res)
# What CG does around the SpMV each iteration: apply the diagonal
# preconditioner, two dots for alpha/beta, and three axpy-shaped updates of
# x, r and p.  Timed as a group because individually they are too fast to
# resolve against the clock.
t_vec = timeit(function ()
                   @. y = ls.precond.inv_diag * v
                   s1 = dot(v, y); s2 = dot(w, y)
                   @. v = v + s1 * w
                   @. w = w - s2 * y
                   @. y = y + s1 * v
                   return s1 + s2
               end, "precond + 2 dot + 3 axpy", res)

@printf("\n--- extrapolated to one step (%d CG iters, 3 Newton) ---\n", CG_PER_STEP)
asm_step = 3 * (t_K + t_M + t_add + t_pre + t_res)
cg_spmv  = CG_PER_STEP * t_spmv
cg_vec   = CG_PER_STEP * t_vec
tot      = asm_step + cg_spmv + cg_vec
@printf("  %-28s %8.2f s  %5.1f%%\n", "CG SpMV", cg_spmv, 100 * cg_spmv / tot)
@printf("  %-28s %8.2f s  %5.1f%%\n", "CG precond + vector ops", cg_vec, 100 * cg_vec / tot)
@printf("  %-28s %8.2f s  %5.1f%%\n", "assembly + residual (x3)", asm_step, 100 * asm_step / tot)
@printf("  %-28s %8.2f s\n", "TOTAL (predicted step)", tot)
@printf("  measured step wall at 24 threads: 19.50 s (run log)\n")
@printf("==================================\n")
