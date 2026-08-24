# Two-phase (lgrtk-style) matrix-free action assembly.
#
# FEC's kernel gives each GPU thread one ELEMENT: gather 4x24 nodal values,
# loop the quadrature points serially, scatter 24 results with atomic adds.
# The per-thread working set (element fields + the 24-wide accumulator held
# across the qp loop) is what limits occupancy, and one thread per element
# barely fills an A100 for this mesh.
#
# The two-phase form splits that along the lgrtk pattern:
#   phase 1  one thread per (element, qp): compute the qp action and write
#            its 24 values to a disjoint slot of a staging array - no
#            atomics, no accumulator, 8x the threads, 1/8th the state;
#   phase 2  one thread per node: sum the staging slots of the (element,
#            local-node) pairs incident on the node, found through an
#            inverse adjacency (CSR) built once on the CPU - disjoint
#            writes, no atomics.
#
# The action functor API is untouched: `func_action` was per-qp all along;
# only the summation and scatter strategy changes.  Staging for the torsion
# mesh is 24 x 8 x nelem doubles (~250 MB, ~0.3 ms of extra traffic on an
# A100) - the bet was that occupancy buys more than the traffic costs.
#
# VERDICT (2026-08-24, torsion 530k DOF, NewmarkAction, deformed state):
# the bet loses on every card - A100 3.04 -> 3.80 ms (0.80x), RX 7600
# 9.89 -> 12.57 ms (0.79x), V100 6.59 -> 10.05 ms (0.66x) - and the loss
# ordering tracks effective cache capacity (V100's 6 MB L2 absorbs least
# of the 8x-duplicated element gathers).  The fused kernel is memory-
# system-bound on the gather side, not occupancy-bound, and the earlier
# atomic ablation was already flat, so two-phase fixes neither cost while
# adding staging traffic.  Kept off by default (`_two_phase_enabled`) as
# a measured negative result; per-iteration gains from here require a
# data-layout change or fewer iterations (GPU AMG), not a thread-mapping
# change.

# The qualified forms KA.@kernel / KA.@index are NOT rewritten by the
# @kernel macro on the CPU backend (same trap as KA.@Const in the device-CG
# round) — the imports must be unqualified.
import KernelAbstractions: @kernel, @index

# ---------------------------------------------------------------------------
# Inverse adjacency: node -> (element, local node) pairs, per block
# ---------------------------------------------------------------------------

struct _InverseAdjacency{V <: AbstractVector{Int32}}
    offsets::V   # n_nodes + 1 CSR offsets into `pairs`
    pairs  ::V   # packed codes (e - 1) * nnpe + ln, 1-based
end

# Built on the CPU assembler at setup (needs host connectivity); adapted to
# the device lazily on first use.
function _build_inverse_adjacency(asm_cpu)
    fspace = FEC.function_space(asm_cpu.dof)
    conns_all = fspace.elem_conns
    n_nodes = size(asm_cpu.stiffness_action_storage, 2)

    blocks = _InverseAdjacency{Vector{Int32}}[]
    for (b, ref_fe) in enumerate(values(fspace.ref_fes))
        nelems  = conns_all.nelems[b]
        coffset = conns_all.offsets[b]
        nnpe    = RFE.num_cell_dofs(ref_fe)

        counts = zeros(Int32, n_nodes)
        for e in 1:nelems
            conn = FEC.connectivity(ref_fe, conns_all.data, e, coffset)
            for ln in 1:nnpe
                counts[conn[ln]] += Int32(1)
            end
        end

        offsets = Vector{Int32}(undef, n_nodes + 1)
        offsets[1] = Int32(1)
        for i in 1:n_nodes
            offsets[i + 1] = offsets[i] + counts[i]
        end

        pos   = copy(offsets)
        pairs = Vector{Int32}(undef, Int(offsets[end]) - 1)
        for e in 1:nelems
            conn = FEC.connectivity(ref_fe, conns_all.data, e, coffset)
            for ln in 1:nnpe
                node = conn[ln]
                pairs[pos[node]] = Int32((e - 1) * nnpe + ln)
                pos[node] += Int32(1)
            end
        end
        push!(blocks, _InverseAdjacency(offsets, pairs))
    end
    return blocks
end

# ---------------------------------------------------------------------------
# Module-level caches (reset per run by _init_assembly_cache!)
# ---------------------------------------------------------------------------

# Host CSR per block, built at setup from the CPU assembler.
const _two_phase_host = Ref{Any}(nothing)
# Device-side workspaces per block: (adjacency, staging), built lazily.
const _two_phase_ws = Ref{Any}(nothing)
# Master switch; flipped per-backend policy at simulation setup or by
# benchmark scripts.
const _two_phase_enabled = Ref{Bool}(false)

_use_two_phase(backend) =
    _two_phase_enabled[] && _two_phase_host[] !== nothing &&
    !(backend isa KA.CPU)

function _two_phase_device_ws(storage, fspace, backend)
    ws = _two_phase_ws[]
    ws === nothing || return ws
    host = _two_phase_host[]
    host === nothing && error(
        "two-phase assembly requested but the inverse adjacency was never " *
        "built; _init_assembly_cache! must run first")
    conns_all = fspace.elem_conns
    upload(v) = (d = KA.allocate(backend, eltype(v), length(v)); copyto!(d, v); d)
    ws = map(enumerate(values(fspace.ref_fes))) do (b, ref_fe)
        adj = _InverseAdjacency(upload(host[b].offsets), upload(host[b].pairs))
        nelems  = conns_all.nelems[b]
        nq      = RFE.num_cell_quadrature_points(ref_fe)
        ndof_el = RFE.num_cell_dofs(ref_fe) * size(storage, 1)
        staging = KA.allocate(backend, eltype(storage.data), ndof_el, nq, nelems)
        (adj = adj, staging = staging)
    end
    _two_phase_ws[] = ws
    return ws
end

# ---------------------------------------------------------------------------
# Phase 1: per-(element, qp) action into disjoint staging
# ---------------------------------------------------------------------------

@kernel function _two_phase_qp_kernel!(
    staging, conns, coffset::Int, func_action::F, physics, ref_fe,
    X, t, Δt, U, U_old, V, state_old, state_new, props, b::Int, ::Val{NQ},
) where {F, NQ}
    i = @index(Global, Linear)
    e = (i - 1) ÷ NQ + 1
    q = (i - 1) % NQ + 1
    conn = FEC.connectivity(ref_fe, conns, e, coffset)
    x_el, u_el, u_el_old, v_el =
        FEC.element_level_fields(ref_fe, conn, X, U, U_old, V)
    props_el = FEC.properties(props, e, b)
    interps = FEC._cell_interpolants(ref_fe, q)
    state_old_q = FEC.state_variables(state_old, q, e, b)
    state_new_q = FEC.state_variables(state_new, q, e, b)
    Kv_q = func_action(physics, interps, x_el, t, Δt, u_el, u_el_old, v_el,
                       state_old_q, state_new_q, props_el)
    @inbounds for j in 1:length(Kv_q)
        staging[j, q, e] = Kv_q[j]
    end
end

# ---------------------------------------------------------------------------
# Phase 2: per-node gather through the inverse adjacency
# ---------------------------------------------------------------------------

@kernel function _two_phase_gather_kernel!(
    out_data, staging, offsets, pairs, ::Val{NNPE}, ::Val{NQ}, ::Val{ND},
) where {NNPE, NQ, ND}
    node = @index(Global, Linear)
    @inbounds begin
        lo = Int(offsets[node])
        hi = Int(offsets[node + 1]) - 1
        if hi >= lo
            acc = zero(MVector{ND, eltype(out_data)})
            for k in lo:hi
                code = Int(pairs[k])
                e    = (code - 1) ÷ NNPE + 1
                ln   = (code - 1) % NNPE + 1
                base = ND * (ln - 1)
                for q in 1:NQ
                    for d in 1:ND
                        acc[d] += staging[base + d, q, e]
                    end
                end
            end
            gbase = ND * (node - 1)
            for d in 1:ND
                out_data[gbase + d] += acc[d]
            end
        end
    end
end

# ---------------------------------------------------------------------------
# Entry point: drop-in for FEC.assemble_matrix_free_action! on the GPU
# ---------------------------------------------------------------------------

function _assemble_action_two_phase!(asm, func_action::F, Uu, Vu, p) where {F <: Function}
    storage = asm.stiffness_action_storage
    fill!(storage, zero(eltype(storage)))
    dof = asm.dof
    fspace = FEC.function_space(dof)
    X  = FEC.coordinates(p)
    t  = FEC.current_time(p)
    Δt = FEC.time_step(p)
    U     = p.field
    U_old = p.field_old
    V     = p.hvp_scratch_field
    FEC._update_for_assembly!(p, dof, Uu, Vu)
    conns_all = fspace.elem_conns
    backend = KA.get_backend(storage.data)
    ws = _two_phase_device_ws(storage, fspace, backend)
    nd = size(storage, 1)
    n_nodes = size(storage, 2)
    FEC.foreach_block(fspace, p) do physics, ref_fe, b
        nelems  = conns_all.nelems[b]
        coffset = conns_all.offsets[b]
        nq      = RFE.num_cell_quadrature_points(ref_fe)
        nnpe    = RFE.num_cell_dofs(ref_fe)
        _two_phase_qp_kernel!(backend)(
            ws[b].staging, conns_all.data, coffset, func_action, physics,
            ref_fe, X, t, Δt, U, U_old, V,
            p.state_old, p.state_new, p.properties, b, Val(nq);
            ndrange = nelems * nq,
        )
        _two_phase_gather_kernel!(backend)(
            storage.data, ws[b].staging, ws[b].adj.offsets, ws[b].adj.pairs,
            Val(nnpe), Val(nq), Val(nd);
            ndrange = n_nodes,
        )
    end
    return nothing
end

# The switch the solver hot paths call: two-phase where enabled and possible,
# FEC's fused single-pass kernel otherwise.
function _assemble_action!(asm, func_action::F, Uu, Vu, p) where {F <: Function}
    backend = KA.get_backend(asm.stiffness_action_storage.data)
    if _use_two_phase(backend)
        _assemble_action_two_phase!(asm, func_action, Uu, Vu, p)
    else
        FEC.assemble_matrix_free_action!(asm, func_action, Uu, Vu, p)
    end
    return nothing
end
