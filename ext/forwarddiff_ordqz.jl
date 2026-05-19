function _ordqz!(
        S::StridedMatrix{<:Dual{T, V, N}}, Targ::StridedMatrix{<:Dual{T, V, N}},
        Q::StridedMatrix{<:Dual{T, V, N}}, Z::StridedMatrix{<:Dual{T, V, N}},
        A::StridedMatrix{<:Dual{T, V, N}}, B::StridedMatrix{<:Dual{T, V, N}},
        ordering::Symbol, threshold
    ) where {T, V <: Union{Float32, Float64}, N}
    Aval = map(value, A)
    Bval = map(value, B)
    Sval = Matrix{V}(undef, size(S))
    Tval = Matrix{V}(undef, size(Targ))
    Qval = Matrix{V}(undef, size(Q))
    Zval = Matrix{V}(undef, size(Z))

    sdim = _ordqz!(Sval, Tval, Qval, Zval, Aval, Bval, ordering, threshold)

    tangents = ntuple(Val(N)) do i
        Base.@_inline_meta
        dS = Matrix{V}(undef, size(S))
        dT = Matrix{V}(undef, size(Targ))
        dQ = Matrix{V}(undef, size(Q))
        dZ = Matrix{V}(undef, size(Z))
        ordqz_tangent!(
            dS, dT, dQ, dZ, Sval, Tval, Qval, Zval,
            map(x -> partials(x, i), A),
            map(x -> partials(x, i), B),
        )
        return dS, dT, dQ, dZ
    end

    for idx in eachindex(S)
        S[idx] = Dual{T}(
            Sval[idx],
            Partials(ntuple(k -> tangents[k][1][idx], Val(N))),
        )
        Targ[idx] = Dual{T}(
            Tval[idx],
            Partials(ntuple(k -> tangents[k][2][idx], Val(N))),
        )
        Q[idx] = Dual{T}(
            Qval[idx],
            Partials(ntuple(k -> tangents[k][3][idx], Val(N))),
        )
        Z[idx] = Dual{T}(
            Zval[idx],
            Partials(ntuple(k -> tangents[k][4][idx], Val(N))),
        )
    end

    return sdim
end
