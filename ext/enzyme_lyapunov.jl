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
    symmetrize!!(rhs)
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
    # Docs § ForwardDiff JVP / Enzyme forward: differentiating the
    # implicit equation L_A[X] = C gives another discrete Lyapunov
    # equation against the SAME A:
    #     L_A[dX] = dC + dA·X·A' + A·X·dA'.
    # One `schur(A)` (`lyapdfactor`) is shared across the primal and
    # every tangent direction.
    N = EnzymeRules.width(config)
    cache = lyapdfactor(A.val)
    retval = lyapdsolve(cache, C.val)

    dretvals = ntuple(Val(N)) do i
        Base.@_inline_meta
        # Build the JVP RHS in lane `i`: start from dC, then add the two
        # outer-product corrections dA·X·A' and A·X·dA' (only when A is
        # not Const).
        rhs = _shadow_dense(C, i, N)
        if !(typeof(A) <: Const)
            dA = _shadow_ref(A, i, N)
            rhs .+= dA * retval * A.val'   # dA · X · A'
            rhs .+= A.val * retval * dA'   # A · X · dA'
        end
        # Solve L_A[dX] = rhs via the cached Schur factor.
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
    # Docs § Enzyme VJP:
    #   Step 1: solve the adjoint Lyapunov equation Y − A'·Y·A = X̄
    #           via `lyapdadjointsolve`, reusing the cached `schur(A)`
    #           stashed on Enzyme's tape by augmented_primal.
    #   Step 2: parameter cotangents
    #             C̄ += Y,
    #             Ā += Y·A·X' + Y'·A·X.
    # The `_symmetric_part_like` projection upstream of the solve handles
    # the case where the primal C was `Symmetric` — see docs § Enzyme VJP
    # opening paragraph about projection onto the symmetric manifold.
    X, dXs, cache, Aval, Aprimal, Cprimal = tape
    N = EnzymeRules.width(config)
    for i in 1:N
        Xbar = N == 1 ? dXs : dXs[i]
        # Step 1: Y solves L_A^*[Y] = X̄.
        Y = lyapdadjointsolve(cache, _symmetric_part_like(Cprimal, Xbar))

        if !(typeof(C) <: Const)
            # Step 2: C̄ += Y (`_add_parameter_shadow!` projects onto the
            # symmetric manifold when the primal C was `Symmetric`).
            dC = _shadow_ref(C, i, N)
            _add_parameter_shadow!(Cprimal, dC, Y)
        end
        if !(typeof(A) <: Const)
            # Step 2: Ā += Y·A·X' + Y'·A·X.
            dA = _shadow_ref(A, i, N)
            tmp = Y * Aval
            Abar = tmp * X'                # Y · A · X'
            tmp = Y' * Aval
            Abar .+= tmp * X               # Y' · A · X
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
    # Same JVP equation as the out-of-place forward rule (docs §
    # ForwardDiff JVP). The only difference: the primal solution is
    # written into `X.val` and each tangent dX is written into the
    # matching `X.dval` shadow.
    cache = lyapdfactor(A.val)
    lyapd!(X.val, cache, C.val)

    # If X is `Const`, the caller is asking only for the primal write into
    # `X.val`; no tangent buffer to fill.
    typeof(X) <: Const && return nothing

    N = EnzymeRules.width(config)
    for i in 1:N
        # Build the JVP RHS dC + dA·X·A' + A·X·dA' lane-by-lane.
        # `_shadow_dense(C, …)` already gives a fresh copy of `C.dval[i]`
        # (or zeros for `Const`); mutate it in place. `dA` is only read,
        # so `_shadow_ref` is enough — copying it would be a wasted
        # full-matrix alloc per lane.
        rhs = _shadow_dense(C, i, N)
        if !(typeof(A) <: Const)
            dA = _shadow_ref(A, i, N)
            rhs .+= dA * X.val * A.val'   # dA · X · A'
            rhs .+= A.val * X.val * dA'   # A · X · dA'
        end
        # Solve L_A[dX] = rhs into the caller-supplied shadow buffer —
        # saves the intermediate `dX = lyapdsolve(…); copyto!(…)` alloc.
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
    # Same VJP as the out-of-place reverse rule (docs § Enzyme VJP
    # Steps 1–2):
    #   Y = L_A^*[X̄] = (I − A'·Y·A)⁻¹ X̄  (in-place via cached schur(A))
    #   C̄ += Y;   Ā += Y·A·X' + Y'·A·X.
    # The primal X is read from the tape (`X_primal`), not from `X.val`,
    # because the in-place rule has already overwritten the caller's
    # buffer with subsequent calls in the program.
    X_primal, cache, Aval, Aprimal, Cprimal = tape
    N = EnzymeRules.width(config)
    for i in 1:N
        Xbar = N == 1 ? X.dval : X.dval[i]
        # Step 1: adjoint Lyapunov solve.
        Y = lyapdadjointsolve(cache, _symmetric_part_like(Cprimal, Xbar))

        if !(typeof(C) <: Const)
            # Step 2: C̄ += Y.
            dC = _shadow_ref(C, i, N)
            _add_parameter_shadow!(Cprimal, dC, Y)
        end
        if !(typeof(A) <: Const)
            # Step 2: Ā += Y·A·X' + Y'·A·X.
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
