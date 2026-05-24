# Symmetric discrete Lyapunov solver via the Kronecker-vec form:
#
#     A * X * A' - X + C = 0
#
# Equivalently, `(I - A ⊗ A) * vec(X) = vec(C)`. The returned solution is
# symmetrized, so nonsymmetric perturbations of C are projected onto the
# symmetric solution manifold.

@inline function build_M!!(M, A)
    kron!(M, A, A)
    M .= .-M
    @inbounds for i in 1:size(M, 1)
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

function lyapdkr(
        A::StridedMatrix{T}, C::StridedMatrix{T},
    ) where {T <: Union{Float32, Float64}}
    n = size(A, 1)
    M = Matrix{T}(undef, n * n, n * n)
    M = build_M!!(M, A)
    F = lu!(M)
    X = copy(C)
    ldiv!(F, vec(X))
    _symmetrize_square!(X, n)
    return X
end
