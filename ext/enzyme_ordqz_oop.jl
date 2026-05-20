# Enzyme rules for the out-of-place ordqz wrapper. These forward to the same
# adjoint kernels (`ordqz_tangent!`, `ordqz_adjoint!`) used by the in-place
# `_ordqz!` rule.

const _ORDQZ_OOP_FLOAT = Union{Float32, Float64}

function _oop_qz_run_primal(
        ::typeof(ordqz), A::AbstractMatrix{T}, B::AbstractMatrix{T},
        ordering::Symbol, threshold,
    ) where {T}
    n = size(A, 1)
    S = Matrix{T}(undef, n, n)
    Tm = Matrix{T}(undef, n, n)
    Q = Matrix{T}(undef, n, n)
    Z = Matrix{T}(undef, n, n)
    sdim = _ordqz!(S, Tm, Q, Z, A, B, ordering, threshold)
    return (S, Tm, Q, Z, sdim)
end

_oop_qz_named(S, T, Q, Z, sdim) = (; S, T, Q, Z, sdim)

function _oop_qz_perturb(A::AbstractMatrix, δ)
    A_reg = copy(A)
    @inbounds for i in axes(A_reg, 1)
        A_reg[i, i] += δ
    end
    return A_reg
end

# Build an output shadow matching the primal NamedTuple shape (mutable matrices
# downstream code may write tangents into).
function _oop_qz_make_shadow(::Type{Tel}, n) where {Tel}
    return (;
        S = zeros(Tel, n, n),
        T = zeros(Tel, n, n),
        Q = zeros(Tel, n, n),
        Z = zeros(Tel, n, n),
        sdim = 0,
    )
end

# ----------------------------------------------------------------------------
# Forward rule
# ----------------------------------------------------------------------------

function _oop_qz_forward_impl(
        config, F, RT, A, B, select_or_ordering, threshold_or_criterium,
        regularize_A,
    )
    Tel = eltype(A.val)
    n = size(A.val, 1)
    Aval = iszero(regularize_A) ? A.val : _oop_qz_perturb(A.val, regularize_A)
    Sv, Tv, Qv, Zv, sdim = _oop_qz_run_primal(
        F, Aval, B.val, select_or_ordering, threshold_or_criterium,
    )
    primal = _oop_qz_named(Sv, Tv, Qv, Zv, sdim)

    if RT <: Const || !EnzymeRules.needs_shadow(config)
        return EnzymeRules.needs_primal(config) ? primal : nothing
    end

    N = EnzymeRules.width(config)
    shadows = ntuple(Val(N)) do i
        dA = if typeof(A) <: Const
            zeros(Tel, n, n)
        elseif N == 1
            A.dval
        else
            A.dval[i]
        end
        dB = if typeof(B) <: Const
            zeros(Tel, n, n)
        elseif N == 1
            B.dval
        else
            B.dval[i]
        end
        dS = zeros(Tel, n, n)
        dT = zeros(Tel, n, n)
        dQ = zeros(Tel, n, n)
        dZ = zeros(Tel, n, n)
        ordqz_tangent!(dS, dT, dQ, dZ, Sv, Tv, Qv, Zv, dA, dB)
        return _oop_qz_named(dS, dT, dQ, dZ, 0)
    end

    if RT <: DuplicatedNoNeed
        return N == 1 ? shadows[1] : shadows
    end
    if RT <: BatchDuplicatedNoNeed
        return shadows
    end
    if RT <: Duplicated
        return Duplicated(primal, shadows[1])
    end
    if RT <: BatchDuplicated
        return BatchDuplicated(primal, shadows)
    end
    return EnzymeRules.needs_primal(config) ? primal : nothing
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(ordqz)},
        RT::Type,
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}},
        ordering::Const{Symbol} = Const(:bk);
        threshold = DEFAULT_BK_THRESHOLD,
        regularize_A = 0,
    ) where {T <: _ORDQZ_OOP_FLOAT}
    return _oop_qz_forward_impl(
        config, ordqz, RT, A, B, ordering.val, threshold, regularize_A,
    )
end

# ----------------------------------------------------------------------------
# Reverse rule (augmented_primal + reverse)
# ----------------------------------------------------------------------------

function _oop_qz_augmented_impl(
        config, F, RT, A, B, select_or_ordering, threshold_or_criterium,
        regularize_A,
    )
    Tel = eltype(A.val)
    n = size(A.val, 1)
    Aval = iszero(regularize_A) ? A.val : _oop_qz_perturb(A.val, regularize_A)
    Sv, Tv, Qv, Zv, sdim = _oop_qz_run_primal(
        F, Aval, B.val, select_or_ordering, threshold_or_criterium,
    )
    primal = _oop_qz_named(Sv, Tv, Qv, Zv, sdim)

    # Always allocate a shadow when the output is a Duplicated/BatchDuplicated
    # NamedTuple so downstream code has somewhere to accumulate cotangents.
    if RT <: Const
        shadow = nothing
    else
        N = EnzymeRules.width(config)
        if N == 1
            shadow = _oop_qz_make_shadow(Tel, n)
        else
            shadow = ntuple(_ -> _oop_qz_make_shadow(Tel, n), Val(N))
        end
    end

    # Tape carries the primal factors plus the shadow so the reverse pass can
    # read cotangents that downstream code has written.
    tape = (Sv, Tv, Qv, Zv, shadow)
    returned_primal = EnzymeRules.needs_primal(config) ? primal : nothing
    return EnzymeRules.AugmentedReturn(returned_primal, shadow, tape)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(ordqz)},
        RT::Type,
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}},
        ordering::Const{Symbol} = Const(:bk);
        threshold = DEFAULT_BK_THRESHOLD,
        regularize_A = 0,
    ) where {T <: _ORDQZ_OOP_FLOAT}
    return _oop_qz_augmented_impl(
        config, ordqz, RT, A, B, ordering.val, threshold, regularize_A,
    )
end

function _oop_qz_reverse_impl(config, A, B, tape)
    Sv, Tv, Qv, Zv, shadow = tape
    shadow === nothing && return nothing

    Tel = eltype(A.val)
    n = size(A.val, 1)
    N = EnzymeRules.width(config)
    for i in 1:N
        sh = N == 1 ? shadow : shadow[i]
        dA = if typeof(A) <: Const
            zeros(Tel, n, n)
        elseif N == 1
            A.dval
        else
            A.dval[i]
        end
        dB = if typeof(B) <: Const
            zeros(Tel, n, n)
        elseif N == 1
            B.dval
        else
            B.dval[i]
        end
        ordqz_adjoint!(dA, dB, Sv, Tv, Qv, Zv, sh.S, sh.T, sh.Q, sh.Z)
    end
    return nothing
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(ordqz)},
        RT::Type,
        tape,
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}},
        ordering::Const{Symbol} = Const(:bk);
        threshold = DEFAULT_BK_THRESHOLD,
        regularize_A = 0,
    ) where {T <: _ORDQZ_OOP_FLOAT}
    _oop_qz_reverse_impl(config, A, B, tape)
    return (nothing, nothing, nothing)
end
