@inline function _dual_zero_matrix(::Type{Dual{T, V, N}}, dims) where {T, V, N}
    return zeros(Dual{T, V, N}, dims)
end

function ared(
        A::StridedMatrix{<:Dual{T, V, N}},
        B::StridedMatrix{<:Dual{T, V, N}},
        R::StridedMatrix{<:Dual{T, V, N}},
        Q::StridedMatrix{<:Dual{T, V, N}},
        S::StridedMatrix{<:Dual{T, V, N}};
        scaling = 'B', pow2 = false, as = false,
        rtol::Real = size(A, 1) * eps(real(float(one(V)))), nrm = 1
    ) where {T, V <: Union{Float32, Float64}, N}
    Aval = map(value, A)
    Bval = map(value, B)
    Rval = map(value, R)
    Qval = map(value, Q)
    Sval = map(value, S)
    X, evals, F, Z, scalinfo, Acl, cache = _ared_primal(
        Aval, Bval, Rval, Qval, Sval; scaling, pow2, as, rtol, nrm
    )

    tangents = ntuple(Val(N)) do i
        Base.@_inline_meta
        dA = map(x -> partials(x, i), A)
        dB = map(x -> partials(x, i), B)
        dR = map(x -> partials(x, i), R)
        dQ = map(x -> partials(x, i), Q)
        dS = map(x -> partials(x, i), S)
        _ared_tangent(
            Aval, Bval, Rval, Qval, Sval, X, F, Acl, cache,
            dA, dB, dR, dQ, dS
        )
    end

    Xdual = map(CartesianIndices(X)) do idx
        Base.@_inline_meta
        Dual{T}(X[idx], Partials(ntuple(k -> tangents[k][1][idx], Val(N))))
    end
    Fdual = map(CartesianIndices(F)) do idx
        Base.@_inline_meta
        Dual{T}(F[idx], Partials(ntuple(k -> tangents[k][2][idx], Val(N))))
    end

    return Xdual, evals, Fdual, Z, scalinfo
end

function ared(
        A::StridedMatrix{<:Dual{T, V, N}},
        B::StridedMatrix{<:Dual{T, V, N}},
        R::StridedMatrix{<:Dual{T, V, N}},
        Q::StridedMatrix{<:Dual{T, V, N}};
        scaling = 'B', pow2 = false, as = false,
        rtol::Real = size(A, 1) * eps(real(float(one(V)))), nrm = 1
    ) where {T, V <: Union{Float32, Float64}, N}
    S = _dual_zero_matrix(Dual{T, V, N}, size(B))
    return ared(A, B, R, Q, S; scaling, pow2, as, rtol, nrm)
end
