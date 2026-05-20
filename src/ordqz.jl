const DEFAULT_BK_THRESHOLD = 1.0e-6

function qzselection(F::GeneralizedSchur, ordering::Symbol, threshold)
    threshold >= zero(threshold) ||
        throw(ArgumentError("threshold must be nonnegative"))
    return qzselection(F, Val(ordering), threshold)
end

function qzselection(F::GeneralizedSchur, ::Val{:bk}, threshold)
    n = length(F.α)
    selection = Vector{Bool}(undef, n)
    scale2 = (1 - threshold)^2
    @inbounds for i in 1:n
        selection[i] = abs2(F.α[i]) >= scale2 * abs2(F.β[i])
    end
    return selection
end

function qzselection(::GeneralizedSchur, ::Val{ordering}, threshold) where {ordering}
    throw(ArgumentError("unsupported QZ ordering :$ordering; only :bk is supported"))
end

function ordqz(
        A::AbstractMatrix, B::AbstractMatrix, ordering::Symbol = :bk;
        threshold = DEFAULT_BK_THRESHOLD, regularize_A = 0,
    )
    n = checksquare(A)
    checksquare(B) == n ||
        throw(DimensionMismatch("A and B must have matching square sizes"))
    Tel = promote_type(eltype(A), eltype(B))
    S = Matrix{Tel}(undef, n, n)
    T = Matrix{Tel}(undef, n, n)
    Q = Matrix{Tel}(undef, n, n)
    Z = Matrix{Tel}(undef, n, n)
    sdim = ordqz!(S, T, Q, Z, A, B, ordering; threshold, regularize_A)
    return (; S, T, Q, Z, sdim)
end

function ordqz!(
        S::AbstractMatrix, T::AbstractMatrix, Q::AbstractMatrix, Z::AbstractMatrix,
        A::AbstractMatrix, B::AbstractMatrix, ordering::Symbol = :bk;
        threshold = DEFAULT_BK_THRESHOLD, regularize_A = 0,
    )
    # Regularization semantics: a positive `regularize_A` factors
    # (A + regularize_A·I, B), which breaks coincident generalized eigenvalues
    # at the problem level so the tangent is smooth.
    if iszero(regularize_A)
        return _ordqz!(S, T, Q, Z, A, B, ordering, threshold)
    end
    A_reg = _ordqz_regularize_diagonal(A, regularize_A)
    return _ordqz!(S, T, Q, Z, A_reg, B, ordering, threshold)
end

function _ordqz_regularize_diagonal(A::AbstractMatrix, δ)
    A_reg = copy(A)
    @inbounds for i in axes(A_reg, 1)
        A_reg[i, i] += δ
    end
    return A_reg
end

function _ordqz!(
        S::AbstractMatrix, T::AbstractMatrix, Q::AbstractMatrix, Z::AbstractMatrix,
        A::AbstractMatrix, B::AbstractMatrix, ordering::Symbol, threshold
    )
    F = schur(A, B)
    selection = qzselection(F, ordering, threshold)
    sdim = count(selection)
    ordschur!(F, selection)
    copyto!(S, F.S)
    copyto!(T, F.T)
    copyto!(Q, F.Q)
    copyto!(Z, F.Z)
    return sdim
end
