# AD-rule plumbing and Enzyme rules for `MatrixEquations.lyapd` and the
# in-place `MatrixEquationsAD.lyapd!`. Primal kernels and the cache type
# (`LyapDSchurCache`, `lyapdfactor`, `lyapdsolve`, `lyapdadjointsolve`,
# the cache-aware `lyapd` shadow, and all `lyapd!` Float methods) live in
# `src/lyapd.jl`.

@inline function _dense_copy(A::StridedMatrix)
    return copy(A)
end

@inline function _dense_copy(A::Symmetric)
    return Matrix(A)
end

@inline function _dense_zero(A::AbstractMatrix{T}) where {T}
    return zeros(T, size(A))
end

@inline function _symmetric_like(::StridedMatrix, A)
    return A
end

@inline function _symmetric_like(C::Symmetric, A)
    return Symmetric(A, Symbol(C.uplo))
end

function _symmetric_part_like(C::Symmetric, A)
    rhs = copy(A)
    _symmetrize_square!(rhs, size(rhs, 1))
    return Symmetric(rhs, Symbol(C.uplo))
end

@inline function _symmetric_part_like(::StridedMatrix, A)
    return A
end

@inline function _shadow_dense(x, i, N)
    if typeof(x) <: Const
        return _dense_zero(x.val)
    elseif N == 1
        return _dense_copy(x.dval)
    else
        return _dense_copy(x.dval[i])
    end
end

@inline function _shadow_ref(x, i, N)
    return N == 1 ? x.dval : x.dval[i]
end

function _add_shadow!(shadow::StridedMatrix, grad)
    shadow .+= grad
    return shadow
end

function _add_shadow!(shadow::Symmetric, grad)
    parent(shadow) .+= grad
    return shadow
end

function _add_parameter_shadow!(primal::StridedMatrix, shadow, grad)
    _add_shadow!(shadow, grad)
    return shadow
end

function _add_parameter_shadow!(primal::Symmetric, shadow, grad)
    projected = _symmetric_part_like(primal, grad)
    _add_shadow!(shadow, projected)
    return shadow
end

function _lyapd_enzyme_forward(
        config::EnzymeRules.FwdConfig, ::Type{RT}, A, C
    ) where {RT <: Union{Const, Duplicated, DuplicatedNoNeed, BatchDuplicated, BatchDuplicatedNoNeed}}
    N = EnzymeRules.width(config)
    cache = lyapdfactor(A.val)
    retval = lyapdsolve(cache, C.val)

    dretvals = ntuple(Val(N)) do i
        Base.@_inline_meta
        rhs = _shadow_dense(C, i, N)
        if !(typeof(A) <: Const)
            dA = _shadow_ref(A, i, N)
            rhs .+= dA * retval * A.val'
            rhs .+= A.val * retval * dA'
        end
        lyapdsolve(cache, _symmetric_like(C.val, rhs))
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

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(lyapd)},
        ::Type{RT},
        A::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:StridedMatrix{T}}
    ) where {RT <: Union{Const, Duplicated, DuplicatedNoNeed, BatchDuplicated, BatchDuplicatedNoNeed}, T <: Union{Float32, Float64}}
    return _lyapd_enzyme_forward(config, RT, A, C)
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(lyapd)},
        ::Type{RT},
        A::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:Symmetric{T, <:StridedMatrix{T}}}
    ) where {RT <: Union{Const, Duplicated, DuplicatedNoNeed, BatchDuplicated, BatchDuplicatedNoNeed}, T <: Union{Float32, Float64}}
    return _lyapd_enzyme_forward(config, RT, A, C)
end

function _lyapd_enzyme_augmented_primal(
        config::EnzymeRules.RevConfig, ::Type{RT}, A, C
    ) where {RT}
    cache = lyapdfactor(A.val)
    X = lyapdsolve(cache, C.val)
    dXs = EnzymeRules.width(config) == 1 ? zero(X) :
        ntuple(_ -> zero(X), Val(EnzymeRules.width(config)))

    primal = EnzymeRules.needs_primal(config) ? X : nothing
    tape = (copy(X), dXs, cache, _dense_copy(A.val), A.val, C.val)
    return EnzymeRules.AugmentedReturn(
        primal::EnzymeRules.primal_type(config, RT),
        dXs, tape
    )
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(lyapd)},
        ::Type{RT},
        A::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:StridedMatrix{T}}
    ) where {RT, T <: Union{Float32, Float64}}
    return _lyapd_enzyme_augmented_primal(config, RT, A, C)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(lyapd)},
        ::Type{RT},
        A::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:Symmetric{T, <:StridedMatrix{T}}}
    ) where {RT, T <: Union{Float32, Float64}}
    return _lyapd_enzyme_augmented_primal(config, RT, A, C)
end

function _lyapd_enzyme_reverse(
        config::EnzymeRules.RevConfig,
        ::Type{RT}, tape, A, C
    ) where {RT}
    X, dXs, cache, Aval, Aprimal, Cprimal = tape
    N = EnzymeRules.width(config)
    for i in 1:N
        Xbar = N == 1 ? dXs : dXs[i]
        Y = lyapdadjointsolve(cache, _symmetric_part_like(Cprimal, Xbar))

        if !(typeof(C) <: Const)
            dC = _shadow_ref(C, i, N)
            _add_parameter_shadow!(Cprimal, dC, Y)
        end
        if !(typeof(A) <: Const)
            dA = _shadow_ref(A, i, N)
            tmp = Y * Aval
            Abar = tmp * X'
            tmp = Y' * Aval
            Abar .+= tmp * X
            _add_parameter_shadow!(Aprimal, dA, Abar)
        end

        fill!(Xbar, zero(eltype(Xbar)))
    end

    return (nothing, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(lyapd)},
        ::Type{RT},
        tape,
        A::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:StridedMatrix{T}}
    ) where {RT, T <: Union{Float32, Float64}}
    return _lyapd_enzyme_reverse(config, RT, tape, A, C)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(lyapd)},
        ::Type{RT},
        tape,
        A::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:Symmetric{T, <:StridedMatrix{T}}}
    ) where {RT, T <: Union{Float32, Float64}}
    return _lyapd_enzyme_reverse(config, RT, tape, A, C)
end
# ─── Enzyme rules for lyapd! ─────────────────────────────────────────────────
#
# `lyapd!` returns `nothing` and mutates `X`, so the function-value annotation
# is always `Const`; tangents/cotangents on the output flow through
# `X.dval` (or `X.dval[i]` under `BatchDuplicated`). The cache rides the tape
# for reverse mode so the reverse pass never re-schurs `A`.

function _lyapd_inplace_enzyme_forward(
        config::EnzymeRules.FwdConfig, ::Type{RT}, X, A, C,
    ) where {RT}
    cache = lyapdfactor(A.val)
    lyapd!(X.val, cache, C.val)

    # If X is `Const`, the caller is asking only for the primal write into
    # `X.val`; no tangent buffer to fill.
    typeof(X) <: Const && return nothing

    N = EnzymeRules.width(config)
    for i in 1:N
        # `_shadow_dense(C, …)` already gives a fresh copy of `C.dval[i]`
        # (or zeros for `Const`); mutate it in place to build the tangent
        # RHS. `dA` is only read, so `_shadow_ref` is enough — copying it
        # would be a wasted full-matrix alloc per lane.
        rhs = _shadow_dense(C, i, N)
        if !(typeof(A) <: Const)
            dA = _shadow_ref(A, i, N)
            rhs .+= dA * X.val * A.val'
            rhs .+= A.val * X.val * dA'
        end
        # Solve into the caller-supplied shadow buffer directly — saves
        # the intermediate `dX = lyapdsolve(…); copyto!(…)` alloc.
        dX_target = N == 1 ? X.dval : X.dval[i]
        lyapd!(dX_target, cache, _symmetric_like(C.val, rhs))
    end
    return nothing
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(lyapd!)},
        ::Type{RT},
        X::Annotation{<:StridedMatrix{T}},
        A::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:StridedMatrix{T}},
    ) where {RT, T <: Union{Float32, Float64}}
    return _lyapd_inplace_enzyme_forward(config, RT, X, A, C)
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(lyapd!)},
        ::Type{RT},
        X::Annotation{<:StridedMatrix{T}},
        A::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:Symmetric{T, <:StridedMatrix{T}}},
    ) where {RT, T <: Union{Float32, Float64}}
    return _lyapd_inplace_enzyme_forward(config, RT, X, A, C)
end

function _lyapd_inplace_augmented_primal(
        config::EnzymeRules.RevConfig, ::Type{RT}, X, A, C,
    ) where {RT}
    cache = lyapdfactor(A.val)
    lyapd!(X.val, cache, C.val)
    tape = (copy(X.val), cache, copy(A.val), A.val, C.val)
    return EnzymeRules.AugmentedReturn(nothing, nothing, tape)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(lyapd!)},
        ::Type{RT},
        X::Annotation{<:StridedMatrix{T}},
        A::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:StridedMatrix{T}},
    ) where {RT, T <: Union{Float32, Float64}}
    return _lyapd_inplace_augmented_primal(config, RT, X, A, C)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(lyapd!)},
        ::Type{RT},
        X::Annotation{<:StridedMatrix{T}},
        A::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:Symmetric{T, <:StridedMatrix{T}}},
    ) where {RT, T <: Union{Float32, Float64}}
    return _lyapd_inplace_augmented_primal(config, RT, X, A, C)
end

function _lyapd_inplace_enzyme_reverse(
        config::EnzymeRules.RevConfig, ::Type{RT}, tape, X, A, C,
    ) where {RT}
    X_primal, cache, Aval, Aprimal, Cprimal = tape
    N = EnzymeRules.width(config)
    for i in 1:N
        Xbar = N == 1 ? X.dval : X.dval[i]
        Y = lyapdadjointsolve(cache, _symmetric_part_like(Cprimal, Xbar))

        if !(typeof(C) <: Const)
            dC = _shadow_ref(C, i, N)
            _add_parameter_shadow!(Cprimal, dC, Y)
        end
        if !(typeof(A) <: Const)
            dA = _shadow_ref(A, i, N)
            tmp = Y * Aval
            Abar = tmp * X_primal'
            tmp = Y' * Aval
            Abar .+= tmp * X_primal
            _add_parameter_shadow!(Aprimal, dA, Abar)
        end

        fill!(Xbar, zero(eltype(Xbar)))
    end
    return (nothing, nothing, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(lyapd!)},
        ::Type{RT},
        tape,
        X::Annotation{<:StridedMatrix{T}},
        A::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:StridedMatrix{T}},
    ) where {RT, T <: Union{Float32, Float64}}
    return _lyapd_inplace_enzyme_reverse(config, RT, tape, X, A, C)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(lyapd!)},
        ::Type{RT},
        tape,
        X::Annotation{<:StridedMatrix{T}},
        A::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:Symmetric{T, <:StridedMatrix{T}}},
    ) where {RT, T <: Union{Float32, Float64}}
    return _lyapd_inplace_enzyme_reverse(config, RT, tape, X, A, C)
end
