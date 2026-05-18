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
        rhs = MatrixEquations.utqu(C, cache.Z)
        MatrixEquations.lyapds!(cache.T, rhs)
        MatrixEquations.utqu!(rhs, cache.Z')
    else
        rhs = cache.Z' * C * cache.Z
        MatrixEquations.sylvds!(-cache.T, cache.T, rhs; adjB = true)
        rhs = cache.Z * rhs * cache.Z'
    end
    return rhs
end

function MatrixEquations.lyapd(
        A::FDDualMatrix{T, V, N},
        C::FDDualMatrix{T, V, N}
    ) where {T, V <: Union{Float32, Float64}, N}
    Aval = map(ForwardDiff.value, A)
    Cval = map(ForwardDiff.value, C)
    cache = lyapdfactor(Aval)
    X = lyapdsolve(cache, Cval)

    dXs = ntuple(Val(N)) do i
        Base.@_inline_meta
        rhs = map(x -> ForwardDiff.partials(x, i), C)
        dA = map(x -> ForwardDiff.partials(x, i), A)
        rhs .+= dA * X * Aval'
        rhs .+= Aval * X * dA'
        lyapdsolve(cache, rhs)
    end

    return map(CartesianIndices(X)) do idx
        Base.@_inline_meta
        ForwardDiff.Dual{T}(
            X[idx],
            ForwardDiff.Partials(ntuple(k -> dXs[k][idx], Val(N))),
        )
    end
end
