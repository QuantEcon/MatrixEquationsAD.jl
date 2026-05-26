# Symmetric discrete Lyapunov solver via the Kronecker-vec form (docs
# § Primal):
#
#     A · X · A' − X + C = 0
#   ⇔ (I_{n²} − A ⊗ A) · vec(X) = vec(C)        [column-major vec]
#   ⇔ M · vec(X) = vec(C),                       M := I_{n²} − A ⊗ A.
#
# The returned solution is symmetrised (docs § Primal: X = P(reshape(…))),
# so non-symmetric perturbations of C are projected onto the symmetric
# solution manifold.

# Build M = I_{n²} − A ⊗ A in place. The `!!` suffix marks "mutates and
# returns the same buffer" (caller-supplied workspace pattern).
@inline function build_M!!(M, A)
    kron!(M, A, A)                          # M ← A ⊗ A
    M .= .-M                                # M ← −A ⊗ A
    @inbounds for i in 1:size(M, 1)
        M[i, i] += one(eltype(M))           # M ← I − A ⊗ A
    end
    return M
end

# Symmetric projection P(X) = (X + X')/2, in place. Walks only the strict
# upper triangle to avoid double-writes.
@inline function symmetrize!!(X::AbstractMatrix)
    @inbounds for j in axes(X, 2)
        for i in 1:(j - 1)
            v = (X[i, j] + X[j, i]) * 0.5
            X[i, j] = v
            X[j, i] = v
        end
    end
    return X
end

function lyapdkr(
        A::StridedMatrix{T}, C::StridedMatrix{T};
        M_ws::Union{Nothing, StridedMatrix{T}} = nothing,
    ) where {T <: Union{Float32, Float64}}
    # X = P(reshape(M⁻¹ · vec(C), n, n)). The `M_ws` kwarg lets the caller
    # share one n²×n² scratch across hot-loop calls.
    n = size(A, 1)
    M = isnothing(M_ws) ? Matrix{T}(undef, n * n, n * n) : M_ws
    build_M!!(M, A)                         # M = I − A ⊗ A
    F = lu!(M)                              # one LU shared with the AD rules
    X = copy(C)
    ldiv!(F, vec(X))                        # X ← M⁻¹ · vec(C), reshape-as-(n, n)
    symmetrize!!(X)                         # X ← P(X)
    return X
end

function lyapdkr!(
        X::StridedMatrix{T},
        A::StridedMatrix{T}, C::StridedMatrix{T};
        M_ws::Union{Nothing, StridedMatrix{T}} = nothing,
    ) where {T <: Union{Float32, Float64}}
    # Same algebra as `lyapdkr`, just writes the solution into the
    # caller-supplied `X` buffer.
    n = size(A, 1)
    M = isnothing(M_ws) ? Matrix{T}(undef, n * n, n * n) : M_ws
    build_M!!(M, A)
    F = lu!(M)
    copyto!(X, C)
    ldiv!(F, vec(X))
    symmetrize!!(X)
    return X
end
