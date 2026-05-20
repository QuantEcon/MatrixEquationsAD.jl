# Out-of-place gges/ordqz on Dual matrices. The base `gges` rejects non-Float
# eltypes for safety; here we route the Dual case through the in-place
# `gges!` / `ordqz!` rules which already have Dual specializations.

function gges(
        A::StridedMatrix{<:Dual{T, V, N}}, B::StridedMatrix{<:Dual{T, V, N}};
        select::Symbol = :ed, criterium = (1 - DEFAULT_BK_THRESHOLD)^2,
    ) where {T, V <: Union{Float32, Float64}, N}
    n = size(A, 1)
    DT = eltype(A)
    S = Matrix{DT}(undef, n, n)
    Tmat = Matrix{DT}(undef, n, n)
    Q = Matrix{DT}(undef, n, n)
    Z = Matrix{DT}(undef, n, n)
    return _gges!(S, Tmat, Q, Z, A, B, select, criterium)
end

function ordqz(
        A::StridedMatrix{<:Dual{T, V, N}}, B::StridedMatrix{<:Dual{T, V, N}},
        ordering::Symbol = :bk;
        threshold = DEFAULT_BK_THRESHOLD,
    ) where {T, V <: Union{Float32, Float64}, N}
    n = size(A, 1)
    DT = eltype(A)
    S = Matrix{DT}(undef, n, n)
    Tmat = Matrix{DT}(undef, n, n)
    Q = Matrix{DT}(undef, n, n)
    Z = Matrix{DT}(undef, n, n)
    sdim = _ordqz!(S, Tmat, Q, Z, A, B, ordering, threshold)
    return (; S, T = Tmat, Q, Z, sdim)
end
