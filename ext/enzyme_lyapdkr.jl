function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(lyapdkr)},
        ::Type{RT},
        A::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:StridedMatrix{T}};
        tol_diag::Real = Inf, check_psd::Bool = false
    ) where {RT <: Union{Const, Duplicated, DuplicatedNoNeed, BatchDuplicated, BatchDuplicatedNoNeed}, T <: Union{Float32, Float64}}
    N = EnzymeRules.width(config)
    cache = lyapdkrfactor(A.val)
    X = lyapdkrsolve(cache, C.val)
    _lyapdkr_check!(X, tol_diag, check_psd)

    dXs = ntuple(Val(N)) do i
        Base.@_inline_meta
        rhs = if typeof(C) <: Const
            zero(C.val)
        elseif N == 1
            copy(C.dval)
        else
            copy(C.dval[i])
        end
        if !(typeof(A) <: Const)
            dA = N == 1 ? A.dval : A.dval[i]
            rhs .+= dA * X * A.val'
            rhs .+= A.val * X * dA'
        end
        lyapdkrsolve(cache, rhs)
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
        C::Annotation{<:StridedMatrix{T}};
        tol_diag::Real = Inf, check_psd::Bool = false
    ) where {RT, T <: Union{Float32, Float64}}
    cache = lyapdkrfactor(A.val)
    X = lyapdkrsolve(cache, C.val)
    _lyapdkr_check!(X, tol_diag, check_psd)
    dXs = EnzymeRules.width(config) == 1 ? zero(X) :
        ntuple(_ -> zero(X), Val(EnzymeRules.width(config)))

    primal = EnzymeRules.needs_primal(config) ? X : nothing
    tape = (copy(X), dXs, cache, copy(A.val))
    return EnzymeRules.AugmentedReturn(
        primal::EnzymeRules.primal_type(config, RT),
        dXs, tape
    )
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(lyapdkr)},
        ::Type{RT},
        tape,
        A::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:StridedMatrix{T}};
        tol_diag::Real = Inf, check_psd::Bool = false
    ) where {RT, T <: Union{Float32, Float64}}
    X, dXs, cache, Aval = tape
    N = EnzymeRules.width(config)

    for i in 1:N
        Xbar = N == 1 ? dXs : dXs[i]
        Y = lyapdkradjointsolve(cache, Xbar)

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
