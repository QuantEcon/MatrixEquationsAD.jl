function _klein_partials_matrix(M, i)
    return map(x -> partials(x, i), M)
end

function _klein_dual_matrix(::Type{Tag}, values, tangents::NTuple{N}) where {Tag, N}
    return map(CartesianIndices(values)) do idx
        Base.@_inline_meta
        Dual{Tag}(
            values[idx],
            Partials(ntuple(i -> tangents[i][idx], Val(N))),
        )
    end
end

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
        _klein_bigk_jvp(plan, _klein_partials_matrix(A, i), _klein_partials_matrix(B, i))
    end
    dg = ntuple(i -> derivs[i].g_x, Val(N))
    dh = ntuple(i -> derivs[i].h_x, Val(N))

    return (;
        g_x = _klein_dual_matrix(Tag, primal.g_x, dg),
        h_x = _klein_dual_matrix(Tag, primal.h_x, dh),
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
            plan, _klein_partials_matrix(A, i), _klein_partials_matrix(B, i),
        )
    end
    dg = ntuple(i -> derivs[i].g_x, Val(N))
    dh = ntuple(i -> derivs[i].h_x, Val(N))

    copyto!(g_x, _klein_dual_matrix(Tag, g_val, dg))
    copyto!(h_x, _klein_dual_matrix(Tag, h_val, dh))
    return (; g_x, h_x)
end
