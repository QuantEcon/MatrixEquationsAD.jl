const AredScaleInfo{T} = NamedTuple{
    (:Sx, :Sxi, :Sr),
    Tuple{Diagonal{T, Vector{T}}, Diagonal{T, Vector{T}}, Diagonal{T, Vector{T}}},
}

const AredOutput{T} = Tuple{
    Matrix{T}, Vector{Complex{T}}, Matrix{T}, Matrix{T}, AredScaleInfo{T},
}

# Enzyme validates tuple-return rules exactly; normalize MatrixEquations' auxiliary
# scaling output so the Riccati return type is concrete under this extension.
function _ared_scale_matrix(scale, n::Integer, ::Type{T}) where {T}
    if scale isa LinearAlgebra.UniformScaling
        return Diagonal(fill(T(scale.λ), n))
    else
        return Diagonal(T.(LinearAlgebra.diag(scale)))
    end
end

function ared(
        A::StridedMatrix{T},
        B::StridedMatrix{T},
        R::StridedMatrix{T},
        Q::StridedMatrix{T},
        S::StridedMatrix{T};
        scaling = 'B', pow2 = false, as = false,
        rtol::Real = size(A, 1) * eps(real(float(one(T)))), nrm = 1
    )::AredOutput{T} where {T <: Union{Float32, Float64}}
    n = size(A, 1)
    m = size(B, 2)
    if iszero(n)
        scalinfo = (
            Sx = Diagonal(Vector{T}()),
            Sxi = Diagonal(Vector{T}()),
            Sr = Diagonal(ones(T, m)),
        )
        return zeros(T, 0, 0), Vector{Complex{T}}(), zeros(T, m, 0),
            zeros(T, m, 0), scalinfo
    end

    X, evals, F, Z, scalinfo = MatrixEquations.gared(
        A, LinearAlgebra.I, B, R, Q, S; scaling, pow2, as, rtol, nrm
    )
    stable_scalinfo = (
        Sx = _ared_scale_matrix(scalinfo.Sx, n, T),
        Sxi = _ared_scale_matrix(scalinfo.Sxi, n, T),
        Sr = _ared_scale_matrix(scalinfo.Sr, m, T),
    )
    return X, Complex{T}.(evals), F, Z, stable_scalinfo
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(ared)},
        ::Type{RT},
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}},
        R::Annotation{<:StridedMatrix{T}},
        Q::Annotation{<:StridedMatrix{T}},
        S::Annotation{<:StridedMatrix{T}};
        scaling = 'B', pow2 = false, as = false,
        rtol::Real = size(A.val, 1) * eps(real(float(one(T)))), nrm = 1
    ) where {RT <: Union{Const, Duplicated, DuplicatedNoNeed, BatchDuplicated, BatchDuplicatedNoNeed}, T <: Union{Float32, Float64}}
    X, evals, F, Z, scalinfo, Acl, cache = _ared_primal(
        A.val, B.val, R.val, Q.val, S.val; scaling, pow2, as, rtol, nrm
    )
    primal = _ared_output(X, evals, F, Z, scalinfo)

    if !EnzymeRules.needs_shadow(config)
        return EnzymeRules.needs_primal(config) ? primal : nothing
    end

    N = EnzymeRules.width(config)
    tangents = ntuple(Val(N)) do i
        Base.@_inline_meta
        _ared_tangent(
            A.val, B.val, R.val, Q.val, S.val, X, F, Acl, cache,
            _copy_shadow_or_zero(A, i, N),
            _copy_shadow_or_zero(B, i, N),
            _copy_shadow_or_zero(R, i, N),
            _copy_shadow_or_zero(Q, i, N),
            _copy_shadow_or_zero(S, i, N),
        )
    end
    shadows = ntuple(Val(N)) do i
        Base.@_inline_meta
        _ared_shadow_output(tangents[i][1], tangents[i][2], X, evals, F, Z, scalinfo)
    end

    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return N == 1 ? Duplicated(primal, shadows[1]) : BatchDuplicated(primal, shadows)
    elseif EnzymeRules.needs_shadow(config)
        return N == 1 ? shadows[1] : shadows
    else
        return nothing
    end
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(ared)},
        ::Type{RT},
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}},
        R::Annotation{<:StridedMatrix{T}},
        Q::Annotation{<:StridedMatrix{T}},
        S::Annotation{<:StridedMatrix{T}};
        scaling = 'B', pow2 = false, as = false,
        rtol::Real = size(A.val, 1) * eps(real(float(one(T)))), nrm = 1
    ) where {RT, T <: Union{Float32, Float64}}
    X, evals, F, Z, scalinfo, Acl, cache = _ared_primal(
        A.val, B.val, R.val, Q.val, S.val; scaling, pow2, as, rtol, nrm
    )
    primal_value = _ared_output(X, evals, F, Z, scalinfo)
    dvalue = EnzymeRules.width(config) == 1 ?
        _ared_zero_output(X, evals, F, Z, scalinfo) :
        ntuple(
            _ -> _ared_zero_output(X, evals, F, Z, scalinfo),
            Val(EnzymeRules.width(config))
        )

    primal = EnzymeRules.needs_primal(config) ? primal_value : nothing
    shadow = EnzymeRules.needs_shadow(config) ? dvalue : nothing
    tape = (copy(X), copy(F), copy(A.val), copy(B.val), copy(R.val), Acl, cache, dvalue)
    return EnzymeRules.AugmentedReturn(
        primal::EnzymeRules.primal_type(config, RT),
        shadow, tape
    )
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(ared)},
        ::Type{RT},
        tape,
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}},
        R::Annotation{<:StridedMatrix{T}},
        Q::Annotation{<:StridedMatrix{T}},
        S::Annotation{<:StridedMatrix{T}};
        scaling = 'B', pow2 = false, as = false,
        rtol::Real = size(A.val, 1) * eps(real(float(one(T)))), nrm = 1
    ) where {RT, T <: Union{Float32, Float64}}
    X, F, Aval, Bval, Rval, Acl, cache, dvalue = tape
    N = EnzymeRules.width(config)

    for i in 1:N
        shadow = N == 1 ? dvalue : dvalue[i]
        Xbar = shadow[1]
        Fbar = shadow[3]
        _ared_adjoint!(
            _shadow_or_zero(A, i, N),
            _shadow_or_zero(B, i, N),
            _shadow_or_zero(R, i, N),
            _shadow_or_zero(Q, i, N),
            _shadow_or_zero(S, i, N),
            Aval, Bval, Rval, X, F, Acl, cache, Xbar, Fbar,
        )
        fill!(Xbar, zero(T))
        fill!(Fbar, zero(T))
    end

    return (nothing, nothing, nothing, nothing, nothing)
end
