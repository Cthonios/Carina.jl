# The backend of a device name for the Taylor scripts (run.jl,
# kernel_timing.jl), resolved as bin/carina.jl does.

# The backend of a device name, as bin/carina.jl resolves it: the vendor
# package must have been loaded by the caller (the Carina library does not
# depend on it), a ROCm run attaches the workgroup-size bound of
# bin/rocm_workgroup_bound.jl and a CUDA run the inlining of
# bin/cuda_always_inline.jl before any kernel is compiled.  Both are included
# here, inside a call, so the caller runs the simulation through
# `Base.invokelatest` to see the methods they define.
#
# always_inline = false (CUDA only) keeps CUDA.jl's default, every
# device-function call compiled as a separate function; it applies only if
# bin/cuda_always_inline.jl has not been included in the process.
function backend_of(device; always_inline::Bool = true)
    device == "cpu" && return Carina.KA.CPU()
    pkg = device == "rocm" ? :AMDGPU : device == "cuda" ? :CUDA :
          error("unknown device $device; expected cpu, rocm or cuda")
    isdefined(Main, pkg) || error("--device $device needs `using $pkg` in the calling session")
    mod = getfield(Main, pkg)
    mod.functional() || error("--device $device: no functional GPU found")
    if device == "rocm"
        include(joinpath(pkgdir(Carina), "bin", "rocm_workgroup_bound.jl"))
        return mod.ROCBackend()
    end
    always_inline || return mod.CUDABackend()
    include(joinpath(pkgdir(Carina), "bin", "cuda_always_inline.jl"))
    return Base.invokelatest(() -> Main.CARINA_CUDA_BACKEND)
end
