function lyapdkr(
        A::StridedMatrix{<:Dual{T, V, N}},
        C::StridedMatrix{<:Dual{T, V, N}},
    ) where {T, V <: Union{Float32, Float64}, N}
    Aval = map(value, A)
    Cval = map(value, C)
    n = size(Aval, 1)
    M = Matrix{V}(undef, n * n, n * n)
    M = build_M!!(M, Aval)
    F = lu!(M)
    X = copy(Cval)
    ldiv!(F, vec(X))
    symmetrize!!(X)

    # Pack tangent RHSs into a single n × n × N tensor so we can do one
    # BLAS-3 multi-RHS solve instead of N per-tangent solves. `XAt` / `AX`
    # are reused across all tangents; `dA_scratch` is filled in place from
    # the i'th partial of `A` each iteration.
    RHS = Array{V, 3}(undef, n, n, N)
    dA_scratch = Matrix{V}(undef, n, n)
    XAt = X * Aval'
    AX = Aval * X
    @inbounds for i in 1:N
        dX = view(RHS, :, :, i)
        for ix in eachindex(dX, C)
            dX[ix] = partials(C[ix], i)
            dA_scratch[ix] = partials(A[ix], i)
        end
        mul!(dX, dA_scratch, XAt, one(V), one(V))
        mul!(dX, AX, dA_scratch', one(V), one(V))
    end
    ldiv!(F, reshape(RHS, n * n, N))
    @inbounds for i in 1:N
        symmetrize!!(view(RHS, :, :, i))
    end

    return map(CartesianIndices(X)) do idx
        Base.@_inline_meta
        Dual{T}(
            X[idx],
            Partials(ntuple(k -> RHS[idx, k], Val(N))),
        )
    end
end
