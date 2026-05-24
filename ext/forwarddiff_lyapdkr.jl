function lyapdkr(
        A::StridedMatrix{<:Dual{T, V, N}},
        C::StridedMatrix{<:Dual{T, V, N}};
        tol_diag::Real = Inf, check_psd::Bool = false,
    ) where {T, V <: Union{Float32, Float64}, N}
    Aval = map(value, A)
    Cval = map(value, C)
    n = size(Aval, 1)
    M = Matrix{V}(undef, n * n, n * n)
    _build_lyapdkr_matrix!(M, Aval, n)
    F = lu!(M)
    X = copy(Cval)
    ldiv!(F, vec(X))
    _symmetrize_square!(X, n)
    _lyapdkr_check!(X, tol_diag, check_psd)

    dXs = ntuple(Val(N)) do i
        Base.@_inline_meta
        dX = map(x -> partials(x, i), C)
        dA = map(x -> partials(x, i), A)
        dX .+= dA * X * Aval'
        dX .+= Aval * X * dA'
        ldiv!(F, vec(dX))
        _symmetrize_square!(dX, n)
        dX
    end

    return map(CartesianIndices(X)) do idx
        Base.@_inline_meta
        Dual{T}(
            X[idx],
            Partials(ntuple(k -> dXs[k][idx], Val(N))),
        )
    end
end
