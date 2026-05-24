function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(lyapdkr)},
        ::Type{RT},
        A::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:StridedMatrix{T}};
        tol_diag::Real = Inf, check_psd::Bool = false,
    ) where {RT <: Union{Const, Duplicated, DuplicatedNoNeed, BatchDuplicated, BatchDuplicatedNoNeed}, T <: Union{Float32, Float64}}
    N = EnzymeRules.width(config)
    n = size(A.val, 1)
    M = Matrix{T}(undef, n * n, n * n)
    _build_lyapdkr_matrix!(M, A.val, n)
    F = lu!(M)
    X = copy(C.val)
    ldiv!(F, vec(X))
    _symmetrize_square!(X, n)
    _lyapdkr_check!(X, tol_diag, check_psd)

    dXs = ntuple(Val(N)) do i
        Base.@_inline_meta
        dX = if typeof(C) <: Const
            zero(C.val)
        else
            copy(N == 1 ? C.dval : C.dval[i])
        end
        if !(typeof(A) <: Const)
            dA = N == 1 ? A.dval : A.dval[i]
            dX .+= dA * X * A.val'
            dX .+= A.val * X * dA'
        end
        ldiv!(F, vec(dX))
        _symmetrize_square!(dX, n)
        dX
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
        tol_diag::Real = Inf, check_psd::Bool = false,
    ) where {RT, T <: Union{Float32, Float64}}
    n = size(A.val, 1)
    M = Matrix{T}(undef, n * n, n * n)
    _build_lyapdkr_matrix!(M, A.val, n)
    F = lu!(M)
    X = copy(C.val)
    ldiv!(F, vec(X))
    _symmetrize_square!(X, n)
    _lyapdkr_check!(X, tol_diag, check_psd)
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
        C::Annotation{<:StridedMatrix{T}};
        tol_diag::Real = Inf, check_psd::Bool = false,
    ) where {RT, T <: Union{Float32, Float64}}
    X, dXs, F, Aval = tape
    N = EnzymeRules.width(config)
    n = size(X, 1)

    for i in 1:N
        Xbar = N == 1 ? dXs : dXs[i]
        Y = copy(Xbar)
        _symmetrize_square!(Y, n)
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
