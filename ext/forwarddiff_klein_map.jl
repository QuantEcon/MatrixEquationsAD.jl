function klein_map(
        A::StridedMatrix{<:Dual{Tag, V, N}},
        B::StridedMatrix{<:Dual{Tag, V, N}};
        threshold = 1.0e-6,
    ) where {Tag, V <: Union{Float32, Float64}, N}
    Aval = map(value, A)
    Bval = map(value, B)
    primal = MatrixEquationsAD.klein_map(Aval, Bval; threshold)
    plan = _klein_bigk_plan(Aval, Bval, primal.g_x, primal.h_x)

    derivs = ntuple(Val(N)) do i
        Base.@_inline_meta
        _klein_bigk_jvp(plan, map(x -> partials(x, i), A), map(x -> partials(x, i), B))
    end
    dg = ntuple(i -> derivs[i].g_x, Val(N))
    dh = ntuple(i -> derivs[i].h_x, Val(N))

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
    plan = _klein_structured_plan(Aval, Bval, g_val, h_val)

    derivs = ntuple(Val(N)) do i
        Base.@_inline_meta
        _klein_structured_jvp(
            plan, map(x -> partials(x, i), A), map(x -> partials(x, i), B),
        )
    end
    dg = ntuple(i -> derivs[i].g_x, Val(N))
    dh = ntuple(i -> derivs[i].h_x, Val(N))

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
