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
        threshold = DEFAULT_BK_THRESHOLD
    )
    F = schur(A, B)
    selection = qzselection(F, ordering, threshold)
    sdim = count(selection)
    ordschur!(F, selection)
    return F, sdim
end

function ordqz!(
        S::AbstractMatrix, T::AbstractMatrix, Q::AbstractMatrix, Z::AbstractMatrix,
        A::AbstractMatrix, B::AbstractMatrix, ordering::Symbol = :bk;
        threshold = DEFAULT_BK_THRESHOLD
    )
    return _ordqz!(S, T, Q, Z, A, B, ordering, threshold)
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
