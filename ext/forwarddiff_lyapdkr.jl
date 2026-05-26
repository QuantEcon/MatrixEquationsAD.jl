# Shared core for both `lyapdkr` (OOP) and `lyapdkr!` (in-place) under
# ForwardDiff Duals. Strip Duals → value layer, build M = I − A ⊗ A once,
# factor once, solve primal:
#     vec(X) = M⁻¹ · vec(C),    X = P(reshape(·, n, n)).
# Then per partial direction i (docs § ForwardDiff JVP):
#     vec(dX_raw_i) = M⁻¹ · vec(dC_i + dA_i·X·A' + A·X·dA_i'),
#     dX_i = P(dX_raw_i).
# All N RHSs are packed into one (n, n, N) tensor for a single BLAS-3
# multi-RHS solve. Returns (X_value, RHS_partials) for the caller to
# pack into Duals.

function _lyapdkr_forwarddiff_solve(A, C, M_ws, ::Type{V}, ::Val{N}) where {V, N}
    Aval = map(value, A)
    Cval = map(value, C)
    n = size(Aval, 1)
    M = isnothing(M_ws) ? Matrix{V}(undef, n * n, n * n) : M_ws::StridedMatrix{V}
    build_M!!(M, Aval)
    F = lu!(M)
    X = copy(Cval)
    ldiv!(F, vec(X))
    symmetrize!!(X)

    # `XAt = X·A'`, `AX = A·X` shared across all N tangent directions.
    RHS = Array{V, 3}(undef, n, n, N)
    dA_scratch = Matrix{V}(undef, n, n)
    XAt = X * Aval'
    AX = Aval * X
    @inbounds for i in 1:N
        dX = view(RHS, :, :, i)
        # Initialise dX with partial(C, i) and stash partial(A, i) in
        # dA_scratch — one index walk for both.
        for ix in eachindex(dX, C)
            dX[ix] = partials(C[ix], i)
            dA_scratch[ix] = partials(A[ix], i)
        end
        mul!(dX, dA_scratch, XAt, one(V), one(V))   # dX += dA · (X · A')
        mul!(dX, AX, dA_scratch', one(V), one(V))   # dX += (A · X) · dA'
    end
    # Single multi-RHS solve, then per-direction symmetric projection.
    ldiv!(F, reshape(RHS, n * n, N))
    @inbounds for i in 1:N
        symmetrize!!(view(RHS, :, :, i))
    end
    return X, RHS
end

function lyapdkr(
        A::StridedMatrix{<:Dual{T, V, N}},
        C::StridedMatrix{<:Dual{T, V, N}};
        M_ws::Union{Nothing, StridedMatrix{V}} = nothing,
    ) where {T, V <: Union{Float32, Float64}, N}
    X, RHS = _lyapdkr_forwarddiff_solve(A, C, M_ws, V, Val(N))
    return map(CartesianIndices(X)) do idx
        Base.@_inline_meta
        Dual{T}(X[idx], Partials(ntuple(k -> RHS[idx, k], Val(N))))
    end
end

function lyapdkr!(
        Xout::StridedMatrix{<:Dual{T, V, N}},
        A::StridedMatrix{<:Dual{T, V, N}},
        C::StridedMatrix{<:Dual{T, V, N}};
        M_ws::Union{Nothing, StridedMatrix{V}} = nothing,
    ) where {T, V <: Union{Float32, Float64}, N}
    X, RHS = _lyapdkr_forwarddiff_solve(A, C, M_ws, V, Val(N))
    @inbounds for idx in CartesianIndices(X)
        Xout[idx] = Dual{T}(X[idx], Partials(ntuple(k -> RHS[idx, k], Val(N))))
    end
    return Xout
end
