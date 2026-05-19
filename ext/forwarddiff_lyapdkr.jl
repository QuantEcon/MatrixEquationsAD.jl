function lyapdkr(
        A::StridedMatrix{<:Dual{T, V, N}},
        C::StridedMatrix{<:Dual{T, V, N}};
        tol_diag::Real = Inf, check_psd::Bool = false
    ) where {T, V <: Union{Float32, Float64}, N}
    Aval = map(value, A)
    Cval = map(value, C)
    cache = lyapdkrfactor(Aval)
    X = lyapdkrsolve(cache, Cval)
    _lyapdkr_check!(X, tol_diag, check_psd)

    dXs = ntuple(Val(N)) do i
        Base.@_inline_meta
        rhs = map(x -> partials(x, i), C)
        dA = map(x -> partials(x, i), A)
        rhs .+= dA * X * Aval'
        rhs .+= Aval * X * dA'
        lyapdkrsolve(cache, rhs)
    end

    return map(CartesianIndices(X)) do idx
        Base.@_inline_meta
        Dual{T}(
            X[idx],
            Partials(ntuple(k -> dXs[k][idx], Val(N))),
        )
    end
end
