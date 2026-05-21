@concrete struct GSylvSchurCache
    AS
    BS
    CS
    DS
    Q1
    Z1
    Q2
    Z2
end

function gsylvfactor(
        A::StridedMatrix{T}, B::StridedMatrix{T},
        C::StridedMatrix{T}, D::StridedMatrix{T}
    ) where {T <: Union{Float32, Float64}}
    AS, CS, Q1, Z1 = schur(A, C)
    BS, DS, Q2, Z2 = schur(B, D)
    return GSylvSchurCache(AS, BS, CS, DS, Q1, Z1, Q2, Z2)
end

function gsylvsolve(cache::GSylvSchurCache, E::StridedMatrix{T}) where {T}
    rhs = cache.Q1' * E * cache.Z2
    gsylvs!(
        cache.AS, cache.BS, cache.CS, cache.DS, rhs;
        adjAC = false, adjBD = false
    )
    return cache.Z1 * rhs * cache.Q2'
end

@concrete struct GSylvKrLUCache
    F
end

function gsylvkrfactor(
        A::StridedMatrix{T}, B::StridedMatrix{T},
        C::StridedMatrix{T}, D::StridedMatrix{T}
    ) where {T <: Union{Float32, Float64}}
    K = LinearAlgebra.kron(transpose(B), A) + LinearAlgebra.kron(transpose(D), C)
    return GSylvKrLUCache(LinearAlgebra.lu(K))
end

function gsylvkrsolve(cache::GSylvKrLUCache, E::StridedMatrix{T}) where {T}
    return reshape(cache.F \ vec(E), size(E))
end

function gsylv(
        A::StridedMatrix{<:Dual{T, V, N}}, B::StridedMatrix{<:Dual{T, V, N}},
        C::StridedMatrix{<:Dual{T, V, N}}, D::StridedMatrix{<:Dual{T, V, N}},
        E::StridedMatrix{<:Dual{T, V, N}}
    ) where {T, V <: Union{Float32, Float64}, N}
    Aval = map(value, A)
    Bval = map(value, B)
    Cval = map(value, C)
    Dval = map(value, D)
    Eval = map(value, E)

    cache = gsylvfactor(Aval, Bval, Cval, Dval)
    X = gsylvsolve(cache, Eval)
    XD = X * Dval
    AX = Aval * X

    dXs = ntuple(Val(N)) do i
        Base.@_inline_meta
        rhs = map(x -> partials(x, i), E)
        rhs .-= map(x -> partials(x, i), A) * X * Bval
        rhs .-= AX * map(x -> partials(x, i), B)
        rhs .-= map(x -> partials(x, i), C) * XD
        rhs .-= Cval * X * map(x -> partials(x, i), D)
        gsylvsolve(cache, rhs)
    end

    return map(CartesianIndices(X)) do idx
        Base.@_inline_meta
        Dual{T}(
            X[idx],
            Partials(ntuple(k -> dXs[k][idx], Val(N))),
        )
    end
end

function gsylvkr(
        A::StridedMatrix{<:Dual{T, V, N}}, B::StridedMatrix{<:Dual{T, V, N}},
        C::StridedMatrix{<:Dual{T, V, N}}, D::StridedMatrix{<:Dual{T, V, N}},
        E::StridedMatrix{<:Dual{T, V, N}}
    ) where {T, V <: Union{Float32, Float64}, N}
    Aval = map(value, A)
    Bval = map(value, B)
    Cval = map(value, C)
    Dval = map(value, D)
    Eval = map(value, E)

    cache = gsylvkrfactor(Aval, Bval, Cval, Dval)
    X = gsylvkrsolve(cache, Eval)
    XD = X * Dval
    AX = Aval * X

    dXs = ntuple(Val(N)) do i
        Base.@_inline_meta
        rhs = map(x -> partials(x, i), E)
        rhs .-= map(x -> partials(x, i), A) * X * Bval
        rhs .-= AX * map(x -> partials(x, i), B)
        rhs .-= map(x -> partials(x, i), C) * XD
        rhs .-= Cval * X * map(x -> partials(x, i), D)
        gsylvkrsolve(cache, rhs)
    end

    return map(CartesianIndices(X)) do idx
        Base.@_inline_meta
        Dual{T}(
            X[idx],
            Partials(ntuple(k -> dXs[k][idx], Val(N))),
        )
    end
end
