# Compile Carina's CUDA kernels with every device-function call inlined.
#
# CUDA.jl compiles a kernel with `always_inline = false` unless its backend
# asks otherwise, and FiniteElementContainers launches its assembly kernels
# with the backend it derives from its arrays, `KA.get_backend(::CuArray) =
# CUDABackend()`.  The element loop and the material functions then remain
# separate device functions, and ptxas chooses one register budget for the
# whole call graph: 32 registers per thread for the general-form diagonal of
# TET15-P1 on an H100 and on an A100 in some builds, with local frames of 50
# KB per thread.  With every call inlined the same kernels compile alike on
# the V100, A100, L4 and H100 (128 registers, a 13.7 KB frame for the general
# diagonal), and on the Taylor bar at h = 0.75 mm the general diagonal is 26
# to 44% faster, the split action 41 to 71% and the split diagonal 28 to 48%
# (4.5% slower on the L4); benchmark/tet15-p1/taylor/README.md, "The
# general-form diagonal restructured, and inlining".  AMDGPU.jl compiles
# with `always_inline = true` already.
#
# The method below is more specific than CUDA.jl's (device-memory arrays,
# which Carina allocates), so it adds to it rather than overwriting it.  It
# must be defined before the first kernel is compiled and in a world the
# caller sees: include this file at the top level, as bin/carina.jl does, or
# call the simulation through `Base.invokelatest` afterwards.  It belongs in
# FiniteElementContainers, whose assemblers should launch with the
# simulation's backend; until then it lives here, next to the vendor package
# it configures, like rocm_workgroup_bound.jl.
import CUDA
import KernelAbstractions as KA

const CARINA_CUDA_BACKEND = CUDA.CUDABackend(; always_inline = true)

KA.get_backend(::CUDA.CuArray{<:Any, <:Any, CUDA.DeviceMemory}) = CARINA_CUDA_BACKEND
