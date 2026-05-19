@concrete struct LyapDSchurCache
    T
    Z
end

function lyapdfactor(A::StridedMatrix{T}) where {T <: Union{Float32, Float64}}
    F = schur(A)
    return LyapDSchurCache(F.T, F.Z)
end

function lyapdfactor(
        A::Symmetric{T, <:StridedMatrix{T}}
    ) where {T <: Union{Float32, Float64}}
    F = schur(A)
    return LyapDSchurCache(F.T, F.Z)
end

function lyapdsolve(cache::LyapDSchurCache, C::StridedMatrix{T}) where {T}
    rhs = cache.Z' * C * cache.Z
    sylvds!(-cache.T, cache.T, rhs; adjB = true)
    rhs = cache.Z * rhs * cache.Z'
    return rhs
end

function lyapdsolve(cache::LyapDSchurCache, C::Symmetric{T, <:StridedMatrix{T}}) where {T}
    rhs = utqu(C, cache.Z)
    lyapds!(cache.T, rhs)
    utqu!(rhs, cache.Z')
    return rhs
end

@inline function _primal_argument(A::StridedMatrix)
    return map(value, A)
end

@inline function _primal_argument(A::Symmetric)
    return Symmetric(map(value, parent(A)), Symbol(A.uplo))
end

@inline function _partial_argument(A::StridedMatrix, i)
    return map(x -> partials(x, i), A)
end

@inline function _partial_argument(A::Symmetric, i)
    return Symmetric(map(x -> partials(x, i), parent(A)), Symbol(A.uplo))
end

@inline function _dense_copy(A::StridedMatrix)
    return copy(A)
end

@inline function _dense_copy(A::Symmetric)
    return Matrix(A)
end

@inline function _symmetric_like(::StridedMatrix, A)
    return A
end

@inline function _symmetric_like(C::Symmetric, A)
    return Symmetric(A, Symbol(C.uplo))
end

function _lyapd_forwarddiff(A, C, ::Type{D}) where {T, V, N, D <: Dual{T, V, N}}
    Aval = _primal_argument(A)
    Cval = _primal_argument(C)
    cache = lyapdfactor(Aval)
    X = lyapdsolve(cache, Cval)

    dXs = ntuple(Val(N)) do i
        Base.@_inline_meta
        dC = _partial_argument(C, i)
        rhs = _dense_copy(dC)
        dA = _partial_argument(A, i)
        rhs .+= dA * X * Aval'
        rhs .+= Aval * X * dA'
        lyapdsolve(cache, _symmetric_like(Cval, rhs))
    end

    return map(CartesianIndices(X)) do idx
        Base.@_inline_meta
        Dual{T}(
            X[idx],
            Partials(ntuple(k -> dXs[k][idx], Val(N))),
        )
    end
end

function lyapd(
        A::StridedMatrix{D}, C::StridedMatrix{D}
    ) where {T, V <: Union{Float32, Float64}, N, D <: Dual{T, V, N}}
    return _lyapd_forwarddiff(A, C, D)
end

function lyapd(
        A::Symmetric{D, <:StridedMatrix{D}}, C::StridedMatrix{D}
    ) where {T, V <: Union{Float32, Float64}, N, D <: Dual{T, V, N}}
    return _lyapd_forwarddiff(A, C, D)
end

function lyapd(
        A::StridedMatrix{D}, C::Symmetric{D, <:StridedMatrix{D}}
    ) where {T, V <: Union{Float32, Float64}, N, D <: Dual{T, V, N}}
    return _lyapd_forwarddiff(A, C, D)
end

function lyapd(
        A::Symmetric{D, <:StridedMatrix{D}}, C::Symmetric{D, <:StridedMatrix{D}}
    ) where {T, V <: Union{Float32, Float64}, N, D <: Dual{T, V, N}}
    return _lyapd_forwarddiff(A, C, D)
end
