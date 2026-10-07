# The backend of a device name for the Taylor scripts (run.jl,
# kernel_timing.jl), resolved as bin/carina.jl does.

# The backend of a device name, as bin/carina.jl resolves it: the vendor
# package must have been loaded by the caller (the Carina library does not
# depend on it), and a ROCm run attaches the workgroup-size bound of
# bin/rocm_workgroup_bound.jl before any kernel is compiled.
#
# always_inline = true (CUDA only) asks CUDA.jl to inline every device-function
# call of the kernels, as CUDABackend(; always_inline = true) does.
function backend_of(device; always_inline::Bool = false)
    device == "cpu" && return Carina.KA.CPU()
    pkg = device == "rocm" ? :AMDGPU : device == "cuda" ? :CUDA :
          error("unknown device $device; expected cpu, rocm or cuda")
    isdefined(Main, pkg) || error("--device $device needs `using $pkg` in the calling session")
    mod = getfield(Main, pkg)
    mod.functional() || error("--device $device: no functional GPU found")
    always_inline && device != "cuda" && error("--always-inline applies to --device cuda only")
    if device == "rocm"
        include(joinpath(pkgdir(Carina), "bin", "rocm_workgroup_bound.jl"))
        return mod.ROCBackend()
    end
    if always_inline
        # FiniteElementContainers launches its kernels with the backend it
        # derives from its arrays, KA.get_backend(::CuArray) = CUDABackend(),
        # whose always_inline is false; the backend passed to the simulation
        # does not reach those launches.  For this process only, the backend
        # of a CuArray is redefined to the inlining one.
        backend = mod.CUDABackend(; always_inline = true)
        @eval Carina.KA.get_backend(::$(mod.CuArray)) = $backend
        return backend
    end
    return mod.CUDABackend()
end
