# Loss of positive definiteness of the punch tangent (TET15-P1, case B, h = a/4):
# split the energy v'Kv of the negative mode, element by element, into the
# isochoric material part ∇v : A_iso : ∇v, the pressure part p̄ ∇v : ∂g/∂F : ∇v
# and the projected volumetric part κ (D'v)' M⁻¹ (D'v), and evaluate the strong
# ellipticity of the pointwise tangents at the quadrature points.
#   julia -t 32 --project=. research/tet15-p1/prototype/pd_split.jl <dump dir> <deck>
ENV["TM_NOMAIN"] = "1"
include("tangent_modes.jl")
using Carina, LinearAlgebra, Tensors, StaticArrays
const FEC = Carina.FEC; const CM = Carina.CM
const MU = 1.0e9    # shear modulus of the punch material

function read_state(f)
    open(f) do io
        map(1:3) do _
            nd = read(io, Int64); dims = Tuple(read(io, Int64) for _ in 1:nd)
            a = Vector{Float64}(undef, prod(dims)); read!(io, a); reshape(a, dims)
        end
    end
end

# The most negative eigenpair of the free-DOF tangent.
function lowest_mode(K)
    Ks = sparse(Symmetric((K + K') / 2)); dmax = maximum(abs, diag(Ks))
    ok(s) = issuccess(cholesky(Symmetric(Ks + s * dmax * I); check = false))
    lo, hi = 0.0, 1e-8
    while !ok(hi); lo = hi; hi *= 10; end
    for _ in 1:30; mid = sqrt(max(lo, hi * 1e-6) * hi); ok(mid) ? (hi = mid) : (lo = mid); end
    lam, V, _ = nearest_modes(Ks, -hi * dmax; k = 3)
    i = argmin(lam)
    return lam[i], V[:, i], Ks, dmax
end

# min over unit a, k of (a ⊗ k) : A : (a ⊗ k), A the 9×9 tangent in the
# column-major vec order (F[i,j] ↦ i + 3(j-1)).
function ellipticity(A; nθ = 91)
    best = Inf; kbest = zeros(3)
    acoustic(k) = [sum(A[i + 3(j - 1), m + 3(l - 1)] * k[j] * k[l] for j in 1:3, l in 1:3) for i in 1:3, m in 1:3]
    for θ in range(0, π; length = nθ), φ in range(0, 2π; length = 2nθ - 1)
        k = [sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ)]
        e = eigmin(Symmetric((acoustic(k) + acoustic(k)') / 2))
        e < best && (best = e; kbest = k)
    end
    # local refinement from the best grid direction
    step = π / nθ
    for _ in 1:60
        improved = false
        for dk in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1])
            k = normalize(kbest + step * dk)
            e = eigmin(Symmetric((acoustic(k) + acoustic(k)') / 2))
            e < best && (best = e; kbest = k; improved = true)
        end
        improved || (step /= 2)
    end
    return best, kbest
end

function main(dir, deckfile, name)
    K, free, _ = read_dump(joinpath(dir, name * ".bin"))
    field, sold, snew = read_state(joinpath(dir, name * "_state.bin"))
    lam, v, Ks, dmax = lowest_mode(K)
    @printf("lam_min = %.5e  (lam / dmax = %.4e, dmax = %.3e)\n", lam, lam / dmax, dmax)

    deck = Carina.YAML.load_file(deckfile; dicttype = Dict{String, Any})
    deck["output mesh file"] = joinpath(dir, "pd_split.e")
    sim = Base.invokelatest(Carina.create_simulation, deck, dir)
    p = sim.params; asm = sim.integrator.asm
    copyto!(p.field.data, field); copyto!(p.state_old.data, sold); copyto!(p.state_new.data, snew)
    nd = length(p.field.data)
    vfull = zeros(nd); vfull[free] = v
    vf = similar(p.field); copyto!(vf.data, vfull)

    fspace = FEC.function_space(asm.dof); conns = fspace.elem_conns
    dt = FEC.time_step(p.times)
    physics = first(values(p.physics)); ref_fe = first(values(fspace.ref_fes))
    model = physics.constitutive_model
    PD = Val(Carina.projection_degree(physics))
    ne = conns.nelems[1]; coff = conns.offsets[1]
    Eiso = zeros(ne); Egeo = zeros(ne); Evol = zeros(ne); Ktot = zeros(ne)
    qmin = fill(Inf, ne); qmin_iso = fill(Inf, ne); pmax = zeros(ne)
    for e in 1:ne
        conn = FEC.connectivity(ref_fe, conns.data, e, coff)
        x_el = FEC._element_level_fields_flat(p.coords, ref_fe, conn)
        u_el = FEC._element_level_fields_flat(p.field, ref_fe, conn)
        v_el = FEC._element_level_fields_flat(vf, ref_fe, conn)
        props_el = FEC.properties(p.properties, e, 1)
        states = FEC.ElementState(p.state_old, p.state_new, e, 1)
        θ̄, Minv, _ = Carina._project(physics, ref_fe, x_el, u_el)
        κ = CM.bulk_modulus(model, props_el)
        Dv = zero(θ̄)
        for q in 1:RFE.num_cell_quadrature_points(ref_fe)
            interps = FEC._cell_interpolants(ref_fe, q)
            cell = FEC.map_interpolants(interps, x_el)
            ∇u = Carina._gradient_at(physics, cell, u_el); ∇v = Carina._gradient_at(physics, cell, v_el)
            so, sn = FEC.state_variables(states, q)
            A = FEC.extract_stiffness(FEC.ThreeDimensional(),
                CM.isochoric_material_tangent(model, props_el, so, sn, dt, ∇u, 0.0))
            vol = Carina._volumetric(model, ∇u); χ = Carina._projection_basis(PD, interps.ξ)
            pmax[e] = max(pmax[e], abs(κ * dot(χ, θ̄)))
            H = (κ * dot(χ, θ̄)) * Carina._dg_matrix(vol)
            w = SVector{9}(∇v.data)
            Eiso[e] += cell.JxW * (w' * A * w); Egeo[e] += cell.JxW * (w' * H * w)
            Dv += (cell.JxW * (Carina._g(vol) ⊡ ∇v)) * χ
            # pointwise tangent of the split form without the projected term,
            # and the isochoric material tangent alone (coarse direction grid)
            qmin[e] = min(qmin[e], ellipticity(Matrix(A + H); nθ = 13)[1] / MU)
            qmin_iso[e] = min(qmin_iso[e], ellipticity(Matrix(A); nθ = 13)[1] / MU)
        end
        Evol[e] = κ * dot(Dv, Minv * Dv)
        Kel = FEC.stiffness(physics, ref_fe, x_el, 0.0, dt, u_el, u_el, states, props_el)
        ve = SVector{length(v_el)}(v_el)
        Ktot[e] = ve' * Kel * ve
    end
    vKv = dot(v, Ks * v)
    @printf("v'Kv assembled %.5e  sum of element v'K_e v %.5e  (iso %.5e  pressure %.5e  vol %.5e)\n",
            vKv, sum(Ktot), sum(Eiso), sum(Egeo), sum(Evol))
    ord = sortperm(abs.(Ktot); rev = true)
    println("elements by |v'K_e v|: e, total, iso, pressure, vol, min ellipticity (A+H)/μ, (A)/μ, max |p̄|/σy")
    for e in ord[1:12]
        @printf("  %5d  %+.4e  %+.4e  %+.4e  %+.4e  %+.4e  %+.4e  %.2f\n", e, Ktot[e], Eiso[e], Egeo[e], Evol[e], qmin[e], qmin_iso[e], pmax[e] / 1e6)
    end
    @printf("mesh: min ellipticity (A+H)/μ %.4e at element %d; elements with negative value %d of %d\n",
            minimum(qmin), argmin(qmin), count(<(0), qmin), ne)
    @printf("mesh: min ellipticity A/μ %.4e at element %d; elements with negative value %d\n",
            minimum(qmin_iso), argmin(qmin_iso), count(<(0), qmin_iso))
    ps = sort(pmax; rev = true)
    @printf("max |p̄|/σy over the mesh %.2f at element %d; 2nd-5th %.2f %.2f %.2f %.2f\n", ps[1] / 1e6, argmax(pmax), (ps[2:5] ./ 1e6)...)
    open(io -> write(io, Float64.(vfull)), joinpath(dir, "negative_mode_full.bin"), "w")
end
main(ARGS[1], ARGS[2], get(ARGS, 3, "first_indef"))
