# Symmetric discrete Lyapunov solver via the Kronecker-vec form:
#
#     A * X * A' - X + C = 0
#
# Equivalently, `(I - A ⊗ A) * vec(X) = vec(C)`. The returned solution is
# symmetrized, so nonsymmetric perturbations of C are projected onto the
# symmetric solution manifold.

@inline function _build_lyapdkr_matrix!(M, A, n)
    kron!(M, A, A)
    M .= .-M
    @inbounds for i in 1:(n * n)
        M[i, i] += one(eltype(M))
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

function lyapdkr(
        A::StridedMatrix{T}, C::StridedMatrix{T};
        tol_diag::Real = Inf, check_psd::Bool = false,
    ) where {T <: Union{Float32, Float64}}
    n = size(A, 1)
    M = Matrix{T}(undef, n * n, n * n)
    _build_lyapdkr_matrix!(M, A, n)
    F = lu!(M)
    X = copy(C)
    ldiv!(F, vec(X))
    _symmetrize_square!(X, n)
    _lyapdkr_check!(X, tol_diag, check_psd)
    return X
end
