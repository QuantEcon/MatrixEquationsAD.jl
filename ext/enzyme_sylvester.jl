@concrete struct GSylvQZCache
    AS
    BS
    CS
    DS
    Q1
    Z1
    Q2
    Z2
end

function gsylvfactor(
        A::StridedMatrix{T}, B::StridedMatrix{T},
        C::StridedMatrix{T}, D::StridedMatrix{T}
    ) where {T <: Union{Float32, Float64}}
    AS, CS, Q1, Z1 = schur(A, C)
    BS, DS, Q2, Z2 = schur(B, D)
    return GSylvQZCache(AS, BS, CS, DS, Q1, Z1, Q2, Z2)
end

function gsylvsolve(cache::GSylvQZCache, E::StridedMatrix{T}) where {T}
    rhs = cache.Q1' * E * cache.Z2
    gsylvs!(
        cache.AS, cache.BS, cache.CS, cache.DS, rhs;
        adjAC = false, adjBD = false
    )
    return cache.Z1 * rhs * cache.Q2'
end

function gsylvadjointsolve(cache::GSylvQZCache, Xbar::StridedMatrix{T}) where {T}
    rhs = cache.Z1' * Xbar * cache.Q2
    gsylvs!(
        cache.AS, cache.BS, cache.CS, cache.DS, rhs;
        adjAC = true, adjBD = true
    )
    return cache.Q1 * rhs * cache.Z2'
end

@concrete struct GSylvKrLUCache
    F
end

function gsylvkrfactor(
        A::StridedMatrix{T}, B::StridedMatrix{T},
        C::StridedMatrix{T}, D::StridedMatrix{T}
    ) where {T <: Union{Float32, Float64}}
    K = LinearAlgebra.kron(transpose(B), A) + LinearAlgebra.kron(transpose(D), C)
    return GSylvKrLUCache(LinearAlgebra.lu(K))
end

function gsylvkrsolve(cache::GSylvKrLUCache, E::StridedMatrix{T}) where {T}
    return reshape(cache.F \ vec(E), size(E))
end

function gsylvkradjointsolve(cache::GSylvKrLUCache, Xbar::StridedMatrix{T}) where {T}
    return reshape(cache.F' \ vec(Xbar), size(Xbar))
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(gsylv)},
        ::Type{RT},
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:StridedMatrix{T}},
        D::Annotation{<:StridedMatrix{T}},
        E::Annotation{<:StridedMatrix{T}}
    ) where {RT <: Union{Const, Duplicated, DuplicatedNoNeed, BatchDuplicated, BatchDuplicatedNoNeed}, T <: Union{Float32, Float64}}
    N = EnzymeRules.width(config)
    cache = gsylvfactor(A.val, B.val, C.val, D.val)
    X = gsylvsolve(cache, E.val)

    XD = X * D.val
    AX = A.val * X

    dXs = ntuple(Val(N)) do i
        Base.@_inline_meta
        rhs = if typeof(E) <: Const
            zero(E.val)
        elseif N == 1
            copy(E.dval)
        else
            copy(E.dval[i])
        end

        if !(typeof(A) <: Const)
            dA = N == 1 ? A.dval : A.dval[i]
            rhs .-= dA * X * B.val
        end
        if !(typeof(B) <: Const)
            dB = N == 1 ? B.dval : B.dval[i]
            rhs .-= AX * dB
        end
        if !(typeof(C) <: Const)
            dC = N == 1 ? C.dval : C.dval[i]
            rhs .-= dC * XD
        end
        if !(typeof(D) <: Const)
            dD = N == 1 ? D.dval : D.dval[i]
            rhs .-= C.val * X * dD
        end

        gsylvsolve(cache, rhs)
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
        func::Const{typeof(gsylv)},
        ::Type{RT},
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:StridedMatrix{T}},
        D::Annotation{<:StridedMatrix{T}},
        E::Annotation{<:StridedMatrix{T}}
    ) where {RT, T <: Union{Float32, Float64}}
    cache = gsylvfactor(A.val, B.val, C.val, D.val)
    X = gsylvsolve(cache, E.val)
    dXs = EnzymeRules.width(config) == 1 ? zero(X) :
        ntuple(_ -> zero(X), Val(EnzymeRules.width(config)))

    primal = EnzymeRules.needs_primal(config) ? X : nothing
    tape = (copy(X), dXs, cache, copy(A.val), copy(B.val), copy(C.val), copy(D.val))
    return EnzymeRules.AugmentedReturn(
        primal::EnzymeRules.primal_type(config, RT),
        dXs, tape
    )
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(gsylv)},
        ::Type{RT},
        tape,
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:StridedMatrix{T}},
        D::Annotation{<:StridedMatrix{T}},
        E::Annotation{<:StridedMatrix{T}}
    ) where {RT, T <: Union{Float32, Float64}}
    X, dXs, cache, Aval, Bval, Cval, Dval = tape
    N = EnzymeRules.width(config)

    for i in 1:N
        Xbar = N == 1 ? dXs : dXs[i]
        Y = gsylvadjointsolve(cache, Xbar)

        if !(typeof(E) <: Const)
            dE = N == 1 ? E.dval : E.dval[i]
            dE .+= Y
        end
        if !(typeof(A) <: Const)
            dA = N == 1 ? A.dval : A.dval[i]
            tmp = Y * Bval'
            LinearAlgebra.mul!(dA, tmp, X', -one(T), one(T))
        end
        if !(typeof(B) <: Const)
            dB = N == 1 ? B.dval : B.dval[i]
            tmp = X' * Aval'
            LinearAlgebra.mul!(dB, tmp, Y, -one(T), one(T))
        end
        if !(typeof(C) <: Const)
            dC = N == 1 ? C.dval : C.dval[i]
            tmp = Y * Dval'
            LinearAlgebra.mul!(dC, tmp, X', -one(T), one(T))
        end
        if !(typeof(D) <: Const)
            dD = N == 1 ? D.dval : D.dval[i]
            tmp = X' * Cval'
            LinearAlgebra.mul!(dD, tmp, Y, -one(T), one(T))
        end

        fill!(Xbar, zero(T))
    end

    return (nothing, nothing, nothing, nothing, nothing)
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(gsylvkr)},
        ::Type{RT},
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:StridedMatrix{T}},
        D::Annotation{<:StridedMatrix{T}},
        E::Annotation{<:StridedMatrix{T}}
    ) where {RT <: Union{Const, Duplicated, DuplicatedNoNeed, BatchDuplicated, BatchDuplicatedNoNeed}, T <: Union{Float32, Float64}}
    N = EnzymeRules.width(config)
    cache = gsylvkrfactor(A.val, B.val, C.val, D.val)
    X = gsylvkrsolve(cache, E.val)

    XD = X * D.val
    AX = A.val * X

    dXs = ntuple(Val(N)) do i
        Base.@_inline_meta
        rhs = if typeof(E) <: Const
            zero(E.val)
        elseif N == 1
            copy(E.dval)
        else
            copy(E.dval[i])
        end

        if !(typeof(A) <: Const)
            dA = N == 1 ? A.dval : A.dval[i]
            rhs .-= dA * X * B.val
        end
        if !(typeof(B) <: Const)
            dB = N == 1 ? B.dval : B.dval[i]
            rhs .-= AX * dB
        end
        if !(typeof(C) <: Const)
            dC = N == 1 ? C.dval : C.dval[i]
            rhs .-= dC * XD
        end
        if !(typeof(D) <: Const)
            dD = N == 1 ? D.dval : D.dval[i]
            rhs .-= C.val * X * dD
        end

        gsylvkrsolve(cache, rhs)
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
        func::Const{typeof(gsylvkr)},
        ::Type{RT},
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:StridedMatrix{T}},
        D::Annotation{<:StridedMatrix{T}},
        E::Annotation{<:StridedMatrix{T}}
    ) where {RT, T <: Union{Float32, Float64}}
    cache = gsylvkrfactor(A.val, B.val, C.val, D.val)
    X = gsylvkrsolve(cache, E.val)
    dXs = EnzymeRules.width(config) == 1 ? zero(X) :
        ntuple(_ -> zero(X), Val(EnzymeRules.width(config)))

    primal = EnzymeRules.needs_primal(config) ? X : nothing
    tape = (copy(X), dXs, cache, copy(A.val), copy(B.val), copy(C.val), copy(D.val))
    return EnzymeRules.AugmentedReturn(
        primal::EnzymeRules.primal_type(config, RT),
        dXs, tape
    )
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(gsylvkr)},
        ::Type{RT},
        tape,
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:StridedMatrix{T}},
        D::Annotation{<:StridedMatrix{T}},
        E::Annotation{<:StridedMatrix{T}}
    ) where {RT, T <: Union{Float32, Float64}}
    X, dXs, cache, Aval, Bval, Cval, Dval = tape
    N = EnzymeRules.width(config)

    for i in 1:N
        Xbar = N == 1 ? dXs : dXs[i]
        Y = gsylvkradjointsolve(cache, Xbar)

        if !(typeof(E) <: Const)
            dE = N == 1 ? E.dval : E.dval[i]
            dE .+= Y
        end
        if !(typeof(A) <: Const)
            dA = N == 1 ? A.dval : A.dval[i]
            tmp = Y * Bval'
            LinearAlgebra.mul!(dA, tmp, X', -one(T), one(T))
        end
        if !(typeof(B) <: Const)
            dB = N == 1 ? B.dval : B.dval[i]
            tmp = X' * Aval'
            LinearAlgebra.mul!(dB, tmp, Y, -one(T), one(T))
        end
        if !(typeof(C) <: Const)
            dC = N == 1 ? C.dval : C.dval[i]
            tmp = Y * Dval'
            LinearAlgebra.mul!(dC, tmp, X', -one(T), one(T))
        end
        if !(typeof(D) <: Const)
            dD = N == 1 ? D.dval : D.dval[i]
            tmp = X' * Cval'
            LinearAlgebra.mul!(dD, tmp, Y, -one(T), one(T))
        end

        fill!(Xbar, zero(T))
    end

    return (nothing, nothing, nothing, nothing, nothing)
end
