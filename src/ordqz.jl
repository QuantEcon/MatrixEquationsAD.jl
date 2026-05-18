qzselect_right_half_plane(alpha, beta) = real(alpha / beta) > 0
qzselect_left_half_plane(alpha, beta) = real(alpha / beta) < 0
qzselect_outside_unit(alpha, beta) = abs(alpha) > abs(beta)
qzselect_inside_unit(alpha, beta) = abs(alpha) < abs(beta)

function qzselection(F::GeneralizedSchur, select)
    n = length(F.α)
    selection = Vector{Bool}(undef, n)
    @inbounds for i in 1:n
        selection[i] = select(F.α[i], F.β[i])
    end
    return selection
end

function qzselection(F::GeneralizedSchur, select::AbstractVector{Bool})
    length(select) == length(F.α) ||
        throw(DimensionMismatch("selection vector length must match pencil dimension"))
    return collect(select)
end

function ordqz(A::AbstractMatrix, B::AbstractMatrix, select)
    F = schur(A, B)
    ordschur!(F, qzselection(F, select))
    return F
end

function ordqz!(
        S::AbstractMatrix, T::AbstractMatrix, Q::AbstractMatrix, Z::AbstractMatrix,
        A::AbstractMatrix, B::AbstractMatrix, select
    )
    F = schur(A, B)
    selection = qzselection(F, select)
    sdim = count(selection)
    ordschur!(F, selection)
    copyto!(S, F.S)
    copyto!(T, F.T)
    copyto!(Q, F.Q)
    copyto!(Z, F.Z)
    return sdim
end
