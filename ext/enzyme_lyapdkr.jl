function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(lyapdkr)},
        ::Type{RT},
        A::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:StridedMatrix{T}},
    ) where {RT <: Union{Const, Duplicated, DuplicatedNoNeed, BatchDuplicated, BatchDuplicatedNoNeed}, T <: Union{Float32, Float64}}
    N = EnzymeRules.width(config)
    n = size(A.val, 1)
    M = Matrix{T}(undef, n * n, n * n)
    M = build_M!!(M, A.val)
    F = lu!(M)
    X = copy(C.val)
    ldiv!(F, vec(X))
    symmetrize!!(X)

    # Pack tangent RHSs into a single n × n × N tensor so we can do one
    # BLAS-3 multi-RHS solve instead of N per-tangent solves. `XAt` / `AX`
    # are reused across tangents.
    RHS = Array{T, 3}(undef, n, n, N)
    if !(typeof(A) <: Const)
        XAt = X * A.val'
        AX = A.val * X
    end
    @inbounds for i in 1:N
        dX = view(RHS, :, :, i)
        if typeof(C) <: Const
            fill!(dX, zero(T))
        else
            dX .= N == 1 ? C.dval : C.dval[i]
        end
        if !(typeof(A) <: Const)
            dA = N == 1 ? A.dval : A.dval[i]
            mul!(dX, dA, XAt, one(T), one(T))
            mul!(dX, AX, dA', one(T), one(T))
        end
    end
    ldiv!(F, reshape(RHS, n * n, N))
    @inbounds for i in 1:N
        symmetrize!!(view(RHS, :, :, i))
    end
    # Materialize shadows as standalone Matrix — Enzyme requires the
    # shadow type to match the primal (Matrix), not a SubArray view.
    dXs = ntuple(Val(N)) do i
        Base.@_inline_meta
        copy(view(RHS, :, :, i))
    end

    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return N == 1 ? Duplicated(X, dXs[1]) : BatchDuplicated(X, dXs)
    elseif EnzymeRules.needs_shadow(config)
        return N == 1 ? dXs[1] : dXs
    elseif EnzymeRules.needs_primal(config)
        return X
    else
        return nothing
    end
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(lyapdkr)},
        ::Type{RT},
        A::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:StridedMatrix{T}},
    ) where {RT, T <: Union{Float32, Float64}}
    n = size(A.val, 1)
    M = Matrix{T}(undef, n * n, n * n)
    M = build_M!!(M, A.val)
    F = lu!(M)
    X = copy(C.val)
    ldiv!(F, vec(X))
    symmetrize!!(X)
    dXs = EnzymeRules.width(config) == 1 ? zero(X) :
        ntuple(_ -> zero(X), Val(EnzymeRules.width(config)))

    primal = EnzymeRules.needs_primal(config) ? X : nothing
    tape = (copy(X), dXs, F, copy(A.val))
    return EnzymeRules.AugmentedReturn(
        primal::EnzymeRules.primal_type(config, RT),
        dXs, tape,
    )
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(lyapdkr)},
        ::Type{RT},
        tape,
        A::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:StridedMatrix{T}},
    ) where {RT, T <: Union{Float32, Float64}}
    X, dXs, F, Aval = tape
    N = EnzymeRules.width(config)
    n = size(X, 1)

    for i in 1:N
        Xbar = N == 1 ? dXs : dXs[i]
        Y = copy(Xbar)
        symmetrize!!(Y)
        ldiv!(transpose(F), vec(Y))

        if !(typeof(C) <: Const)
            dC = N == 1 ? C.dval : C.dval[i]
            dC .+= Y
        end
        if !(typeof(A) <: Const)
            dA = N == 1 ? A.dval : A.dval[i]
            tmp = Y * Aval
            mul!(dA, tmp, X', one(T), one(T))
            tmp = Y' * Aval
            mul!(dA, tmp, X, one(T), one(T))
        end

        fill!(Xbar, zero(T))
    end

    return (nothing, nothing)
end
