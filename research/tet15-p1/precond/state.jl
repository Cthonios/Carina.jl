# Stage 1 of the preconditioner study: run a Carina deck on the CPU to a
# converged state, assemble the tangent of the TET15-P1 element there at the
# deck's bulk modulus κ and at κ' = μ, assemble the element projection
# operators D and M of the volumetric term (research/tet15-p1/note.tex and
# src/projected_physics.jl: the tangent is K_u + κ D M⁻¹ Dᵀ up to the
# initial-stress term of the projected pressure), and the right-hand side of
# the Newton step that follows (the residual with the boundary values
# advanced by one step).  Everything is written to one file for
# experiments.jl.
#
#   julia --project research/tet15-p1/precond/state.jl deck.yaml out.jls
#
# The deck must use the CPU, one element block, the projected element with
# the linear projection and a material with the volumetric-isochoric split
# (J2 or neo-Hookean), and block-constant properties.

using Carina, SparseArrays, LinearAlgebra, Serialization, ForwardDiff, StaticArrays
const FEC = Carina.FEC
const RFE = Carina.RFE

deck, out = ARGS[1], ARGS[2]

sim = Carina.run(deck)
p   = sim.params
asm = sim.asm_cpu
ig  = sim.integrator
p === sim.params_cpu || error("The study runs on the CPU only.")
length(p.physics) == 1 || error("One element block expected; got $(length(p.physics)).")

dof   = asm.dof
udofs = collect(dof.unknown_dofs)
U     = copy(Array(p.field.data))          # full displacement, Dirichlet slots at t_n
Uu    = U[udofs]
nu    = length(udofs)
ndof  = length(U)
X     = copy(vec(Array(p.coords.data)))    # full coordinates, interleaved

# Properties of the block: [ρ, λ, μ, σ_y, H] for J2; κ = λ + 2μ/3.
pf = p.properties
pf.isblockconstant[1] == FEC.PROPS_CONST ||
    error("Block-constant properties expected.")
pv = FEC.properties(pf, 1, 1)
λ  = pv[2]; μ = pv[3]
κ  = λ + 2μ / 3
κp = μ                                     # the bulk modulus of the preconditioner's operator
λp = κp - 2μ / 3
iλ = pv.start + 1                          # linear index of λ in pf.data

function tangent!(asm, Uu, p)
    FEC._update_for_assembly!(p, asm.dof, Uu)
    FEC.assemble_stiffness!(asm, FEC.stiffness, Uu, p)
    K = sparse(FEC.stiffness(asm))
    return sparse((K + K') / 2)
end

@info "Tangent at κ = $κ"
K  = tangent!(asm, Uu, p)
@info "Tangent at κ' = $κp"
pf.data[iλ] = λp
Kp = tangent!(asm, Uu, p)
pf.data[iλ] = λ
FEC._update_for_assembly!(p, asm.dof, Uu)

# Element projection operators.  For element e with nodal displacements u_el
# (45 entries, node-major: 3(a−1)+i), θ̄ = M⁻¹ b(u_el); D_el = (M ∂θ̄/∂u_el)ᵀ,
# 45 × 4, so that the volumetric tangent is κ D_el M⁻¹ D_elᵀ.
fspace  = FEC.function_space(dof)
conns   = fspace.elem_conns
physics = values(p.physics)[1]
ref_fe  = fspace.ref_fes[1]
nel     = conns.nelems[1]
coff    = conns.offsets[1]
nnpe    = RFE.num_cell_dofs(ref_fe)
NC      = Carina._num_projection_functions(Val(Carina.projection_degree(physics)))
NC == 4 || error("The linear projection (4 functions per element) is expected; got $NC.")
np      = NC * nel

fidx = zeros(Int, ndof); fidx[udofs] .= 1:nu      # global dof → free index (0 if constrained)

rows_D = Int[]; cols_D = Int[]; vals_D = Float64[]
rows_M = Int[]; cols_M = Int[]; vals_M = Float64[]; vals_Mi = Float64[]
@info "Projection operators of $nel elements"
for e in 1:nel
    conn = FEC.connectivity(ref_fe, conns.data, e, coff)
    x_el = FEC._element_level_fields_flat(p.coords, ref_fe, conn)
    u_el = FEC._element_level_fields_flat(p.field, ref_fe, conn)
    θ̄, Minv, everted = Carina._project(physics, ref_fe, x_el, u_el)
    everted && error("Element $e is everted.")
    M_el  = inv(Minv)
    Jθ    = ForwardDiff.jacobian(u -> Carina._project(physics, ref_fe, x_el, u)[1], u_el)  # 4 × 45
    D_el  = (M_el * Jθ)'                                                                   # 45 × 4
    for r in 1:3nnpe
        a = (r - 1) ÷ 3 + 1; i = (r - 1) % 3 + 1
        g = 3 * (conn[a] - 1) + i
        f = fidx[g]
        f == 0 && continue
        for c in 1:NC
            push!(rows_D, f); push!(cols_D, NC * (e - 1) + c); push!(vals_D, D_el[r, c])
        end
    end
    for c1 in 1:NC, c2 in 1:NC
        push!(rows_M, NC * (e - 1) + c1); push!(cols_M, NC * (e - 1) + c2)
        push!(vals_M, M_el[c1, c2]); push!(vals_Mi, Minv[c1, c2])
    end
end
D    = sparse(rows_D, cols_D, vals_D, nu, np)
M    = sparse(rows_M, cols_M, vals_M, np, np)
Minv = sparse(rows_M, cols_M, vals_Mi, np, np)

# Consistency: K(κ) − K(κ') = (κ − κ') D M⁻¹ Dᵀ up to the initial-stress term
# of the projected pressure, which is of relative size θ̄ ~ p/κ.
G    = D * Minv * D'
Gsym = sparse((G + G') / 2)
dev  = norm(K - Kp - (κ - κp) * Gsym) / norm(K - Kp)
@info "Relative deviation of K(κ) − K(κ') from (κ − κ') D M⁻¹ Dᵀ: $dev"
dev < 1e-6 || @warn "The projection operators do not reproduce the volumetric tangent to 1e-6."

# Right-hand side of the next Newton step: the residual with the boundary
# values advanced by one step (R_eff = −R, as Carina's solvers use it).
FEC.update_time!(p)
FEC.update_bc_values!(p, asm)
Carina.evaluate!(ig, p) || error("The residual at the advanced boundary values is not finite.")
b = copy(Array(Carina.residual(ig)))

serialize(out, (; K, Kp, D, M, Minv, b, udofs, X, U, κ, κp, μ, λ, nel, nu, np, ndof, deck))
@info "Written $out: $nu displacement unknowns, $np projection unknowns, $(nnz(K)) nonzeros in K"
