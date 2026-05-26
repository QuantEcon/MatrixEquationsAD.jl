function lyapdkr(
        A::StridedMatrix{<:Dual{T, V, N}},
        C::StridedMatrix{<:Dual{T, V, N}};
        M_ws::Union{Nothing, StridedMatrix{V}} = nothing,
    ) where {T, V <: Union{Float32, Float64}, N}
    # Docs § ForwardDiff JVP / Enzyme VJP setup. Strip Duals → value
    # layer, build M = I − A ⊗ A once, factor once, solve primal:
    #     vec(X) = M⁻¹ · vec(C),    X = P(reshape(·, n, n)).
    Aval = map(value, A)
    Cval = map(value, C)
    n = size(Aval, 1)
    M = isnothing(M_ws) ? Matrix{V}(undef, n * n, n * n) : M_ws
    build_M!!(M, Aval)
    F = lu!(M)
    X = copy(Cval)
    ldiv!(F, vec(X))
    symmetrize!!(X)

    # Docs § ForwardDiff JVP — per partial direction i:
    #     vec(dX_raw_i) = M⁻¹ · vec(dC_i + dA_i·X·A' + A·X·dA_i'),
    #     dX_i = P(dX_raw_i).
    # Pack all N tangent RHSs side-by-side into an n × n × N tensor and
    # issue one BLAS-3 multi-RHS `ldiv!`. `XAt = X·A'` and `AX = A·X`
    # are shared across all N directions; `dA_scratch` holds the i-th
    # partial of `A` in dense form.
    RHS = Array{V, 3}(undef, n, n, N)
    dA_scratch = Matrix{V}(undef, n, n)
    XAt = X * Aval'
    AX = Aval * X
    @inbounds for i in 1:N
        dX = view(RHS, :, :, i)
        # Initialise dX with partial(C, i) and stash partial(A, i) in
        # dA_scratch — both in one index walk.
        for ix in eachindex(dX, C)
            dX[ix] = partials(C[ix], i)
            dA_scratch[ix] = partials(A[ix], i)
        end
        mul!(dX, dA_scratch, XAt, one(V), one(V))   # dX += dA · (X · A')
        mul!(dX, AX, dA_scratch', one(V), one(V))   # dX += (A · X) · dA'
    end
    # One multi-RHS solve over all N tangent directions at once …
    ldiv!(F, reshape(RHS, n * n, N))
    # … then symmetric projection per direction.
    @inbounds for i in 1:N
        symmetrize!!(view(RHS, :, :, i))
    end

    # Pack value + N partials back into Duals at each index.
    return map(CartesianIndices(X)) do idx
        Base.@_inline_meta
        Dual{T}(
            X[idx],
            Partials(ntuple(k -> RHS[idx, k], Val(N))),
        )
    end
end

function lyapdkr!(
        Xout::StridedMatrix{<:Dual{T, V, N}},
        A::StridedMatrix{<:Dual{T, V, N}},
        C::StridedMatrix{<:Dual{T, V, N}};
        M_ws::Union{Nothing, StridedMatrix{V}} = nothing,
    ) where {T, V <: Union{Float32, Float64}, N}
    # Same docs § ForwardDiff JVP path as the OOP overload; the only
    # difference is the output Duals are written into the caller's
    # `Xout` buffer.
    Aval = map(value, A)
    Cval = map(value, C)
    n = size(Aval, 1)
    M = isnothing(M_ws) ? Matrix{V}(undef, n * n, n * n) : M_ws
    build_M!!(M, Aval)
    F = lu!(M)
    X = copy(Cval)
    ldiv!(F, vec(X))
    symmetrize!!(X)

    # Multi-RHS tangent solve (see OOP overload for the math).
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

    # Pack value + partials into Duals at each index of the caller buffer.
    @inbounds for idx in CartesianIndices(X)
        Xout[idx] = Dual{T}(
            X[idx],
            Partials(ntuple(k -> RHS[idx, k], Val(N))),
        )
    end
    return Xout
end
