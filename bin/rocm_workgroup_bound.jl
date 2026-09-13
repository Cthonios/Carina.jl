# Tell LLVM the workgroup size a ROCm kernel is launched with.
#
# AMDGPU.jl compiles every kernel without a flat-work-group-size bound, so the
# backend sizes its register budget for the worst case it must support, a
# 1024-thread workgroup.  On RDNA3 that means at most 128 VGPRs per lane so
# that eight waves fit per SIMD, and the element kernels -- which carry far
# more live state than that -- spill the rest to scratch.  Under Julia 1.13
# (LLVM 20) the AMDGPU attributor also pins `amdgpu-waves-per-eu = 8,16` from
# that default, and the same kernels spill 25% more than under LLVM 18
# (benchmark/evidence/julia113_rocm_regression.txt).
#
# Carina launches every element loop at 256 threads (FEC's block-size
# preferences), and a KernelAbstractions kernel carries that size in its
# signature, so the exact bound is known at compile time.  This hook reads it
# off the kernel's `CompilerMetadata` type and attaches
# `amdgpu-flat-work-group-size = 1,<size>` before optimization, where LLVM's
# own attributor derives the waves-per-EU range from it.  The RX 7600 then
# allocates 256 VGPRs at occupancy 4, spills fall by more than half, and the
# stiffness action runs 9.6 -> 8.5 ms under Julia 1.13 -- the Julia 1.12
# level -- with identical checksums.  Kernels with a dynamic workgroup size
# are left alone.
#
# This is a method added to GPUCompiler's `optimize!` for AMDGPU's job type,
# and it belongs in AMDGPU.jl itself (CUDA.jl passes `maxthreads` through to
# its compiler target for the same reason).  Until then it lives here in the
# launcher environment, next to the vendor packages it patches, and every
# ROCm entry point includes it right after `import AMDGPU`.

import AMDGPU
import AMDGPU: GPUCompiler, LLVM
import AMDGPU.GPUCompiler: CompilerJob, GCNCompilerTarget
import KernelAbstractions as KA

const _ROCmJob = CompilerJob{GCNCompilerTarget, AMDGPU.Compiler.HIPCompilerParams}

"""
Static workgroup size of the KernelAbstractions kernel `job` compiles, read from
the `NDRange{N, StaticBlocks, StaticWorkitems, ...}` inside its
`CompilerMetadata` argument; `nothing` if the kernel is not a KA kernel or its
workgroup size is dynamic.
"""
function _ka_static_workgroup(job::CompilerJob)
    for T in job.source.specTypes.parameters
        T isa DataType && T <: KA.CompilerMetadata || continue
        for R in T.parameters
            R isa DataType && R <: KA.NDIteration.NDRange || continue
            W = R.parameters[3]
            W <: KA.NDIteration.StaticSize || return nothing
            return prod(W.parameters[1])
        end
    end
    return nothing
end

function GPUCompiler.optimize!(job::_ROCmJob, mod::LLVM.Module,
                               relocs::GPUCompiler.Relocations; opt_level = 2)
    if job.config.kernel
        wg = _ka_static_workgroup(job)
        if wg !== nothing
            for f in LLVM.functions(mod)
                LLVM.callconv(f) == LLVM.API.LLVMAMDGPUKERNELCallConv || continue
                push!(LLVM.function_attributes(f),
                      LLVM.StringAttribute("amdgpu-flat-work-group-size", "1,$wg"))
            end
        end
    end
    return invoke(GPUCompiler.optimize!,
                  Tuple{CompilerJob, LLVM.Module, GPUCompiler.Relocations},
                  job, mod, relocs; opt_level)
end
