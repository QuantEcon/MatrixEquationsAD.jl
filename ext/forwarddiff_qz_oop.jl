# Out-of-place ordqz on Dual matrices. Routes the Dual case through the
# in-place `ordqz!` rule which already has a Dual specialization.

function _qz_oop_perturb(A::AbstractMatrix, δ)
    A_reg = copy(A)
    @inbounds for i in axes(A_reg, 1)
        A_reg[i, i] += δ
    end
    return A_reg
end

function ordqz(
        A::StridedMatrix{<:Dual{T, V, N}}, B::StridedMatrix{<:Dual{T, V, N}},
        ordering::Symbol = :bk;
        threshold = DEFAULT_BK_THRESHOLD, regularize_A = 0,
    ) where {T, V <: Union{Float32, Float64}, N}
    n = size(A, 1)
    DT = eltype(A)
    S = Matrix{DT}(undef, n, n)
    Tmat = Matrix{DT}(undef, n, n)
    Q = Matrix{DT}(undef, n, n)
    Z = Matrix{DT}(undef, n, n)
    A_eff = iszero(regularize_A) ? A : _qz_oop_perturb(A, regularize_A)
    sdim = _ordqz!(S, Tmat, Q, Z, A_eff, B, ordering, threshold)
    return (; S, T = Tmat, Q, Z, sdim)
end
