@concrete struct LyapDSchurCache
    T
    Z
end

function lyapdfactor(A::StridedMatrix{T}) where {T <: Union{Float32, Float64}}
    F = LinearAlgebra.issymmetric(A) ? schur(Symmetric(A)) : schur(A)
    return LyapDSchurCache(F.T, F.Z)
end

function lyapdsolve(cache::LyapDSchurCache, C::StridedMatrix{T}) where {T}
    if LinearAlgebra.issymmetric(C)
        rhs = utqu(C, cache.Z)
        lyapds!(cache.T, rhs)
        utqu!(rhs, cache.Z')
    else
        rhs = cache.Z' * C * cache.Z
        sylvds!(-cache.T, cache.T, rhs; adjB = true)
        rhs = cache.Z * rhs * cache.Z'
    end
    return rhs
end

function lyapd(
        A::StridedMatrix{<:Dual{T, V, N}},
        C::StridedMatrix{<:Dual{T, V, N}}
    ) where {T, V <: Union{Float32, Float64}, N}
    Aval = map(value, A)
    Cval = map(value, C)
    cache = lyapdfactor(Aval)
    X = lyapdsolve(cache, Cval)

    dXs = ntuple(Val(N)) do i
        Base.@_inline_meta
        rhs = map(x -> partials(x, i), C)
        dA = map(x -> partials(x, i), A)
        rhs .+= dA * X * Aval'
        rhs .+= Aval * X * dA'
        lyapdsolve(cache, rhs)
    end

    return map(CartesianIndices(X)) do idx
        Base.@_inline_meta
        Dual{T}(
            X[idx],
            Partials(ntuple(k -> dXs[k][idx], Val(N))),
        )
    end
end
