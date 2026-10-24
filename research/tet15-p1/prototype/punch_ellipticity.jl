# Strong ellipticity at the quadrature points and positive definiteness of the
# assembled tangent at the converged states of a TET15-P1 punch run
# (benchmark/tet15-p1/punch/run.jl --dump-states).  At every state written,
# and at every quadrature point, the least value over unit a, k of
# (a ⊗ k) : C : (a ⊗ k), divided by μ, for three pointwise tangents C:
#   A          the isochoric material tangent,
#   A + H      with the pressure term p̄ ∂g/∂F, the pointwise part of the
#              TET15-P1 tangent (its volumetric stiffness acts through the
#              projection only),
#   A + H + V  with V = κ g ⊗ g, the tangent of the material itself;
# and whether the assembled tangent of the free DOFs has a Cholesky factor.
# Each state holds the internal variables at the start of the step that
# ended at the output time, so the tangent is the algorithmic tangent of that
# step at its converged displacement.
#   julia -t 32 --project=. research/tet15-p1/prototype/punch_ellipticity.jl <run dir>
using Carina, LinearAlgebra, SparseArrays, StaticArrays, Tensors, Printf
const FEC = Carina.FEC; const CM = Carina.CM; const RFE = Carina.RFE
const MU = 1.0e9    # shear modulus of the punch material

function read_state(f)
    open(f) do io
        map(1:3) do _
            nd = read(io, Int64); dims = Tuple(read(io, Int64) for _ in 1:nd)
            a = Vector{Float64}(undef, prod(dims)); read!(io, a); reshape(a, dims)
        end
    end
end

# min over unit a, k of (a ⊗ k) : C : (a ⊗ k), C the 9×9 tangent in the
# column-major vec order (F[i,j] ↦ i + 3(j-1)): a direction grid in k, then a
# local refinement from the best grid direction.
@inline function acoustic_min(C::SMatrix{9, 9, Float64}, k::SVector{3, Float64})
    Q = SMatrix{3, 3, Float64}(ntuple(Val(9)) do n
        i, m = (n - 1) % 3 + 1, (n - 1) ÷ 3 + 1
        s = 0.0
        for j in 1:3, l in 1:3
            s += C[i + 3(j - 1), m + 3(l - 1)] * k[j] * k[l]
        end
        s
    end)
    return eigmin3((Q + Q') / 2)
end

# least eigenvalue of a symmetric 3×3 matrix by cyclic Jacobi rotations.
# eigmin of a Symmetric SMatrix calls LAPACK through eigvals(A, 1:1) and
# allocates; the closed-form eigvals of StaticArrays loses accuracy near
# repeated eigenvalues.
@inline function eigmin3(S::SMatrix{3, 3, Float64})
    A = MMatrix(S)
    for _ in 1:50
        off = A[1, 2]^2 + A[1, 3]^2 + A[2, 3]^2
        off <= eps()^2 * (A[1, 1]^2 + A[2, 2]^2 + A[3, 3]^2) && break
        for (p, q, r) in ((1, 2, 3), (1, 3, 2), (2, 3, 1))
            apq = A[p, q]
            apq == 0 && continue
            θ = (A[q, q] - A[p, p]) / (2apq)
            t = (θ >= 0 ? 1 : -1) / (abs(θ) + sqrt(θ^2 + 1))
            c = 1 / sqrt(t^2 + 1); s = t * c
            arp, arq = A[r, p], A[r, q]
            A[r, p] = A[p, r] = c * arp - s * arq
            A[r, q] = A[q, r] = s * arp + c * arq
            A[p, p] -= t * apq; A[q, q] += t * apq
            A[p, q] = A[q, p] = 0.0
        end
    end
    return min(A[1, 1], A[2, 2], A[3, 3])
end

function ellipticity(C::SMatrix{9, 9, Float64}; nθ = 19)
    best = Inf; kbest = SVector(0.0, 0.0, 1.0)
    for θ in range(0, π / 2; length = nθ), φ in range(0, 2π; length = 4nθ - 3)
        k = SVector(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))   # k and -k agree
        e = acoustic_min(C, k)
        e < best && (best = e; kbest = k)
    end
    step = π / (2nθ)
    dirs = (SVector(1.0, 0, 0), SVector(0, 1.0, 0), SVector(0, 0, 1.0),
            SVector(-1.0, 0, 0), SVector(0, -1.0, 0), SVector(0, 0, -1.0))
    for _ in 1:40
        improved = false
        for dk in dirs
            k = normalize(kbest + step * dk)
            e = acoustic_min(C, k)
            e < best && (best = e; kbest = k; improved = true)
        end
        improved || (step /= 2)
    end
    return best
end

function main(dir)
    dir = abspath(dir)
    deck = Carina.YAML.load_file(joinpath(dir, "punch.yaml"); dicttype = Dict{String, Any})
    deck["output mesh file"] = joinpath(dir, "states", "ellipticity.e")
    sim = Base.invokelatest(Carina.create_simulation, deck, joinpath(dir, "states"))
    p = sim.params; asm = sim.integrator.asm
    fspace = FEC.function_space(asm.dof); conns = fspace.elem_conns
    dt = FEC.time_step(p.times)
    physics = first(values(p.physics)); ref_fe = first(values(fspace.ref_fes))
    physics isa Carina.ProjectedSolidMechanics || error("not a projected element")
    model = physics.constitutive_model
    PD = Val(Carina.projection_degree(physics))
    ne = conns.nelems[1]; coff = conns.offsets[1]
    nq = RFE.num_cell_quadrature_points(ref_fe)
    free = asm.dof.unknown_dofs
    files = filter(f -> occursin(r"^state_\d+\.bin$", f), readdir(joinpath(dir, "states")))
    sort!(files; by = f -> parse(Int, match(r"\d+", f).match))
    println("# $(basename(dir)): $ne elements, $nq quadrature points each")
    println("# k  t     |R| (N) chol  | A: qp, el, min/μ | A+H: qp, el, min/μ | A+H+V: qp, el, min/μ | max |p̄|/σy")
    for f in files
        k = parse(Int, match(r"\d+", f).match)
        field, sold, snew = read_state(joinpath(dir, "states", f))
        copyto!(p.field.data, field); copyto!(p.state_old.data, sold); copyto!(p.state_new.data, snew)
        E = fill(Inf, 3, nq, ne); pmax = zeros(ne)
        quick = get(ENV, "ELL_QUICK", "0") == "1"   # residual and Cholesky only
        Threads.@threads for e in (quick ? (1:0) : (1:ne))
            conn = FEC.connectivity(ref_fe, conns.data, e, coff)
            x_el = FEC._element_level_fields_flat(p.coords, ref_fe, conn)
            u_el = FEC._element_level_fields_flat(p.field, ref_fe, conn)
            props_el = FEC.properties(p.properties, e, 1)
            states = FEC.ElementState(p.state_old, p.state_new, e, 1)
            θ̄, _, _ = Carina._project(physics, ref_fe, x_el, u_el)
            κ = CM.bulk_modulus(model, props_el)
            for q in 1:nq
                interps = FEC._cell_interpolants(ref_fe, q)
                cell = FEC.map_interpolants(interps, x_el)
                ∇u = Carina._gradient_at(physics, cell, u_el)
                so, sn = FEC.state_variables(states, q)
                A = SMatrix{9, 9, Float64}(FEC.extract_stiffness(FEC.ThreeDimensional(),
                        CM.isochoric_material_tangent(model, props_el, so, sn, dt, ∇u, 0.0)))
                vol = Carina._volumetric(model, ∇u)
                p̄ = κ * dot(Carina._projection_basis(PD, interps.ξ), θ̄)
                pmax[e] = max(pmax[e], abs(p̄))
                H = SMatrix{9, 9, Float64}(p̄ * Carina._dg_matrix(vol))
                g = SVector{9, Float64}(Carina._g(vol).data)
                E[1, q, e] = ellipticity(A) / MU
                E[2, q, e] = ellipticity(A + H) / MU
                E[3, q, e] = ellipticity(A + H + κ * g * g') / MU
            end
        end
        # the Dirichlet values at the output time: assembly rewrites the
        # constrained DOFs of the field from them
        t = 0.05 * k
        p.times.time_current = t
        FEC.update_bc_values!(p, asm)
        Uu = p.field.data[free]
        FEC.assemble_vector!(asm, FEC.residual, Uu, p)
        res = norm(FEC.residual(asm))
        FEC.assemble_stiffness!(asm, FEC.stiffness, Uu, p)
        K = FEC.stiffness(asm)
        pd = issuccess(cholesky(Symmetric(K, :L); check = false))
        @printf("%3d %.3f |R| %.2e %5s", k, t, res, pd ? "yes" : "NO")
        for c in 1:3
            Ec = @view E[c, :, :]
            @printf("  | %6d %5d %+.3e", count(<(0), Ec), count(e -> any(<(0), @view Ec[:, e]), 1:ne), minimum(Ec))
        end
        @printf("  | %.2f\n", maximum(pmax) / 1e6)
        flush(stdout)
    end
end
main(ARGS[1])
