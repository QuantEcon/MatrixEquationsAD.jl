# Symmetric discrete Lyapunov solver via the Kronecker-vec form:
#
#     A * X * A' - X + C = 0
#
# Equivalently, `(I - A ⊗ A) * vec(X) = vec(C)`. The returned solution is
# symmetrized, so nonsymmetric perturbations of C are projected onto the
# symmetric solution manifold.

@concrete struct LyapDKrLUCache
    M
    ipiv
    info::Int
end

@inline function _build_lyapdkr_matrix!(M, A, n)
    @inbounds for l in 1:n, k in 1:n
        col = k + (l - 1) * n
        for j in 1:n, i in 1:n
            row = i + (j - 1) * n
            v = -A[j, l] * A[i, k]
            if i == k && j == l
                v += one(v)
            end
            M[row, col] = v
        end
    end
    return M
end

@inline function _symmetrize_square!(X, n)
    @inbounds for j in 1:n
        for i in (j + 1):n
            s = 0.5 * (X[i, j] + X[j, i])
            X[i, j] = s
            X[j, i] = s
        end
    end
    return X
end

@inline function _lyapdkr_check!(X::AbstractMatrix, tol_diag::Real, check_psd::Bool)
    n = checksquare(X)
    @inbounds for idx in eachindex(X)
        if !isfinite(X[idx])
            throw(ErrorException("lyapdkr: non-finite entry"))
        end
    end
    @inbounds for i in 1:n
        if abs(X[i, i]) > tol_diag
            throw(ErrorException("lyapdkr: |X[$i,$i]| exceeds tol_diag"))
        end
        if check_psd && X[i, i] < 0
            throw(ErrorException("lyapdkr: X[$i,$i] < 0 (non-PSD)"))
        end
    end
    return X
end

function lyapdkrfactor(A::StridedMatrix{T}) where {T <: Union{Float32, Float64}}
    n = checksquare(A)
    M = Matrix{T}(undef, n * n, n * n)
    _build_lyapdkr_matrix!(M, A, n)
    F = lu!(M; check = false)
    if !issuccess(F)
        throw(ErrorException("lyapdkr: LU factorization of (I - A⊗A) failed"))
    end
    return LyapDKrLUCache(M, F.ipiv, 0)
end

function lyapdkrsolve(cache::LyapDKrLUCache, C::StridedMatrix{T}) where {T}
    n = checksquare(C)
    if n * n != size(cache.M, 1)
        throw(DimensionMismatch("lyapdkr: A and C must be the same size"))
    end
    rhs = vec(copy(C))
    F = LU(cache.M, cache.ipiv, cache.info)
    ldiv!(F, rhs)
    X = reshape(rhs, n, n)
    return _symmetrize_square!(X, n)
end

function lyapdkradjointsolve(cache::LyapDKrLUCache, Xbar::StridedMatrix{T}) where {T}
    n = checksquare(Xbar)
    if n * n != size(cache.M, 1)
        throw(DimensionMismatch("lyapdkr: A and Xbar must be the same size"))
    end
    rhs_mat = _symmetrize_square!(copy(Xbar), n)
    rhs = vec(rhs_mat)
    F = LU(cache.M, cache.ipiv, cache.info)
    ldiv!(transpose(F), rhs)
    return reshape(rhs, n, n)
end

function lyapdkr(
        A::StridedMatrix{T}, C::StridedMatrix{T};
        tol_diag::Real = Inf, check_psd::Bool = false
    ) where {T <: Union{Float32, Float64}}
    cache = lyapdkrfactor(A)
    X = lyapdkrsolve(cache, C)
    _lyapdkr_check!(X, tol_diag, check_psd)
    return X
end
