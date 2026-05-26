# ForwardDiff Dual dispatch for the OOP `klein_map` — routes through the
# same big-K JVP pipeline as the Enzyme forward rule (docs § Big-K JVP).
# Strip Duals → value layer, build the plan once on the values, then run
# N partial-direction JVP solves and pack the resulting tangents into
# Dual outputs.

function klein_map(
        A::StridedMatrix{<:Dual{Tag, V, N}},
        B::StridedMatrix{<:Dual{Tag, V, N}};
        threshold = 1.0e-6,
    ) where {Tag, V <: Union{Float32, Float64}, N}
    Aval = map(value, A)
    Bval = map(value, B)
    primal = MatrixEquationsAD.klein_map(Aval, Bval; threshold)
    # Plan = LU of K (docs § Big-K JVP Step 2). Shared across all N partials.
    plan = _klein_bigk_plan(Aval, Bval, primal.g_x, primal.h_x)

    # Per-partial JVP: solve K · v = −vec(dA·Ψ·h_x + dB·Ψ).
    derivs = ntuple(Val(N)) do i
        Base.@_inline_meta
        _klein_bigk_jvp(plan, map(x -> partials(x, i), A), map(x -> partials(x, i), B))
    end
    dg = ntuple(i -> derivs[i].g_x, Val(N))
    dh = ntuple(i -> derivs[i].h_x, Val(N))

    # Re-pack value + N partials into Duals at each output index.
    return (;
        g_x = map(CartesianIndices(primal.g_x)) do idx
            Base.@_inline_meta
            Dual{Tag}(primal.g_x[idx], Partials(ntuple(i -> dg[i][idx], Val(N))))
        end,
        h_x = map(CartesianIndices(primal.h_x)) do idx
            Base.@_inline_meta
            Dual{Tag}(primal.h_x[idx], Partials(ntuple(i -> dh[i][idx], Val(N))))
        end,
    )
end

# ForwardDiff Dual dispatch for the in-place `klein_map!` — routes through
# the reduced-Sylvester JVP pipeline (docs § Reduced-Sylvester JVP). Same
# pattern as the OOP overload above, plus an in-place write of Duals into
# the caller's g_x / h_x buffers.

function klein_map!(
        g_x::StridedMatrix{<:Dual{Tag, V, N}},
        h_x::StridedMatrix{<:Dual{Tag, V, N}},
        A::StridedMatrix{<:Dual{Tag, V, N}},
        B::StridedMatrix{<:Dual{Tag, V, N}};
        threshold = 1.0e-6,
    ) where {Tag, V <: Union{Float32, Float64}, N}
    Aval = map(value, A)
    Bval = map(value, B)
    g_val = Matrix{V}(undef, size(g_x))
    h_val = Matrix{V}(undef, size(h_x))
    MatrixEquationsAD.klein_map!(g_val, h_val, Aval, Bval; threshold)
    # Plan: LU of n × n C₀ + Schurs of J_y and h_x.
    plan = _klein_structured_plan(Aval, Bval, g_val, h_val)

    # Per-partial reduced-Sylvester JVP.
    derivs = ntuple(Val(N)) do i
        Base.@_inline_meta
        _klein_structured_jvp(
            plan, map(x -> partials(x, i), A), map(x -> partials(x, i), B),
        )
    end
    dg = ntuple(i -> derivs[i].g_x, Val(N))
    dh = ntuple(i -> derivs[i].h_x, Val(N))

    # Pack Duals into the caller's output buffers.
    copyto!(
        g_x,
        map(CartesianIndices(g_val)) do idx
            Base.@_inline_meta
            Dual{Tag}(g_val[idx], Partials(ntuple(i -> dg[i][idx], Val(N))))
        end,
    )
    copyto!(
        h_x,
        map(CartesianIndices(h_val)) do idx
            Base.@_inline_meta
            Dual{Tag}(h_val[idx], Partials(ntuple(i -> dh[i][idx], Val(N))))
        end,
    )
    return (; g_x, h_x)
end
