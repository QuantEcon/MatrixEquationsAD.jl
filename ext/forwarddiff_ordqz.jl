function MatrixEquationsAD.ordqz!(
        S::FDDualMatrix{T, V, N}, Targ::FDDualMatrix{T, V, N},
        Q::FDDualMatrix{T, V, N}, Z::FDDualMatrix{T, V, N},
        A::FDDualMatrix{T, V, N}, B::FDDualMatrix{T, V, N}, select
    ) where {T, V <: Union{Float32, Float64}, N}
    Aval = map(ForwardDiff.value, A)
    Bval = map(ForwardDiff.value, B)
    Sval = Matrix{V}(undef, size(S))
    Tval = Matrix{V}(undef, size(Targ))
    Qval = Matrix{V}(undef, size(Q))
    Zval = Matrix{V}(undef, size(Z))

    sdim = MatrixEquationsAD.ordqz!(Sval, Tval, Qval, Zval, Aval, Bval, select)

    tangents = ntuple(Val(N)) do i
        Base.@_inline_meta
        dS = Matrix{V}(undef, size(S))
        dT = Matrix{V}(undef, size(Targ))
        dQ = Matrix{V}(undef, size(Q))
        dZ = Matrix{V}(undef, size(Z))
        ordqz_tangent!(
            dS, dT, dQ, dZ, Sval, Tval, Qval, Zval,
            map(x -> ForwardDiff.partials(x, i), A),
            map(x -> ForwardDiff.partials(x, i), B),
        )
        return dS, dT, dQ, dZ
    end

    for idx in eachindex(S)
        S[idx] = ForwardDiff.Dual{T}(
            Sval[idx],
            ForwardDiff.Partials(ntuple(k -> tangents[k][1][idx], Val(N))),
        )
        Targ[idx] = ForwardDiff.Dual{T}(
            Tval[idx],
            ForwardDiff.Partials(ntuple(k -> tangents[k][2][idx], Val(N))),
        )
        Q[idx] = ForwardDiff.Dual{T}(
            Qval[idx],
            ForwardDiff.Partials(ntuple(k -> tangents[k][3][idx], Val(N))),
        )
        Z[idx] = ForwardDiff.Dual{T}(
            Zval[idx],
            ForwardDiff.Partials(ntuple(k -> tangents[k][4][idx], Val(N))),
        )
    end

    return sdim
end
