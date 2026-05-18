@concrete struct LyapDSchurCache
    T
    Z
end

function lyapdfactor(A::StridedMatrix{T}) where {T <: Union{Float32, Float64}}
    F = LinearAlgebra.issymmetric(A) ? schur(Symmetric(A)) : schur(A)
    return LyapDSchurCache(F.T, F.Z)
end

function lyapdsolve(cache::LyapDSchurCache, C::StridedMatrix{T}) where {T}
    if LinearAlgebra.issymmetric(C)
        rhs = MatrixEquations.utqu(C, cache.Z)
        MatrixEquations.lyapds!(cache.T, rhs)
        MatrixEquations.utqu!(rhs, cache.Z')
    else
        rhs = cache.Z' * C * cache.Z
        MatrixEquations.sylvds!(-cache.T, cache.T, rhs; adjB = true)
        rhs = cache.Z * rhs * cache.Z'
    end
    return rhs
end

function lyapdadjointsolve(cache::LyapDSchurCache, C::StridedMatrix{T}) where {T}
    if LinearAlgebra.issymmetric(C)
        rhs = MatrixEquations.utqu(C, cache.Z)
        MatrixEquations.lyapds!(cache.T, rhs; adj = true)
        MatrixEquations.utqu!(rhs, cache.Z')
    else
        rhs = cache.Z' * C * cache.Z
        MatrixEquations.sylvds!(-cache.T, cache.T, rhs; adjA = true)
        rhs = cache.Z * rhs * cache.Z'
    end
    return rhs
end

function MatrixEquations.lyapd(A::StridedMatrix{T}, C::StridedMatrix{T}) where {T <: Union{Float32, Float64}}
    return lyapdsolve(lyapdfactor(A), C)
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(MatrixEquations.lyapd)},
        ::Type{RT},
        A::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:StridedMatrix{T}}
    ) where {RT <: Union{Const, Duplicated, DuplicatedNoNeed, BatchDuplicated, BatchDuplicatedNoNeed}, T <: Union{Float32, Float64}}
    N = EnzymeRules.width(config)
    cache = lyapdfactor(A.val)
    retval = lyapdsolve(cache, C.val)

    dAs = if typeof(A) <: Const
        ntuple(Returns(nothing), Val(N))
    elseif typeof(A) <: Union{BatchDuplicated, BatchDuplicatedNoNeed}
        A.dval
    else
        ntuple(Returns(A.dval), Val(N))
    end
    dCs = if typeof(C) <: Const
        ntuple(Returns(nothing), Val(N))
    elseif typeof(C) <: Union{BatchDuplicated, BatchDuplicatedNoNeed}
        C.dval
    else
        ntuple(Returns(C.dval), Val(N))
    end

    dretvals = ntuple(Val(N)) do i
        Base.@_inline_meta
        dA = dAs[i]
        dC = dCs[i]
        rhs = if typeof(C) <: Const
            zero(C.val)
        else
            copy(dC)
        end
        if !(typeof(A) <: Const)
            rhs .+= dA * retval * A.val'
            rhs .+= A.val * retval * dA'
        end
        lyapdsolve(cache, rhs)
    end

    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return N == 1 ? Duplicated(retval, dretvals[1]) :
            BatchDuplicated(retval, dretvals)
    elseif EnzymeRules.needs_shadow(config)
        return N == 1 ? dretvals[1] : dretvals
    elseif EnzymeRules.needs_primal(config)
        return retval
    else
        return nothing
    end
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(MatrixEquations.lyapd)},
        ::Type{RT},
        A::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:StridedMatrix{T}}
    ) where {RT, T <: Union{Float32, Float64}}
    cache = lyapdfactor(A.val)
    X = lyapdsolve(cache, C.val)
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
        func::Const{typeof(MatrixEquations.lyapd)},
        ::Type{RT},
        tape,
        A::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:StridedMatrix{T}}
    ) where {RT, T <: Union{Float32, Float64}}
    X, dXs, cache, Aval = tape
    N = EnzymeRules.width(config)
    Xbars = N == 1 ? (dXs,) : dXs
    dAs = if typeof(A) <: Const
        ntuple(Returns(nothing), Val(N))
    elseif typeof(A) <: Union{BatchDuplicated, BatchDuplicatedNoNeed}
        A.dval
    else
        ntuple(Returns(A.dval), Val(N))
    end
    dCs = if typeof(C) <: Const
        ntuple(Returns(nothing), Val(N))
    elseif typeof(C) <: Union{BatchDuplicated, BatchDuplicatedNoNeed}
        C.dval
    else
        ntuple(Returns(C.dval), Val(N))
    end

    for (Xbar, dA, dC) in zip(Xbars, dAs, dCs)
        Y = lyapdadjointsolve(cache, Xbar)

        if !(typeof(C) <: Const)
            dC .+= Y
        end
        if !(typeof(A) <: Const)
            tmp = Y * Aval
            LinearAlgebra.mul!(dA, tmp, X', one(T), one(T))
            tmp = Y' * Aval
            LinearAlgebra.mul!(dA, tmp, X, one(T), one(T))
        end

        fill!(Xbar, zero(T))
    end

    return (nothing, nothing)
end
