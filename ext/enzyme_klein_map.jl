const _KLEIN_AD_FLOAT = Union{Float32, Float64}

function _klein_tangent(arg, i, dims, ::Type{T}, N) where {T}
    if typeof(arg) <: Const
        return zeros(T, dims)
    end
    return N == 1 ? arg.dval : arg.dval[i]
end

function _klein_shadow(arg, i, N)
    if typeof(arg) <: Const
        return nothing
    end
    return N == 1 ? arg.dval : arg.dval[i]
end

function _klein_zero_shadow(::Type{T}, n_y, n_x) where {T}
    return (; g_x = zeros(T, n_y, n_x), h_x = zeros(T, n_x, n_x))
end

function _klein_add_shadow!(g_bar, h_bar, shadow)
    shadow === nothing && return nothing
    g_bar .+= shadow.g_x
    h_bar .+= shadow.h_x
    fill!(shadow.g_x, zero(eltype(shadow.g_x)))
    fill!(shadow.h_x, zero(eltype(shadow.h_x)))
    return nothing
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(klein_map)},
        RT::Type,
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}};
        threshold = 1.0e-6,
    ) where {T <: _KLEIN_AD_FLOAT}
    primal = MatrixEquationsAD.klein_map(A.val, B.val; threshold)
    if RT <: Const || !EnzymeRules.needs_shadow(config)
        return EnzymeRules.needs_primal(config) ? primal : nothing
    end

    N = EnzymeRules.width(config)
    plan = _klein_bigk_plan(A.val, B.val, primal.g_x, primal.h_x)
    shadows = ntuple(Val(N)) do i
        Base.@_inline_meta
        _klein_bigk_jvp(
            plan,
            _klein_tangent(A, i, size(A.val), T, N),
            _klein_tangent(B, i, size(B.val), T, N),
        )
    end

    if RT <: DuplicatedNoNeed
        return shadows[1]
    elseif RT <: BatchDuplicatedNoNeed
        return shadows
    elseif RT <: Duplicated
        return Duplicated(primal, shadows[1])
    elseif RT <: BatchDuplicated
        return BatchDuplicated(primal, shadows)
    end
    return EnzymeRules.needs_primal(config) ? primal : nothing
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(klein_map)},
        RT::Type,
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}};
        threshold = 1.0e-6,
    ) where {T <: _KLEIN_AD_FLOAT}
    primal = MatrixEquationsAD.klein_map(A.val, B.val; threshold)
    n_x = size(primal.h_x, 1)
    n_y = size(primal.g_x, 1)
    shadow = if RT <: Const
        nothing
    elseif EnzymeRules.width(config) == 1
        _klein_zero_shadow(T, n_y, n_x)
    else
        ntuple(_ -> _klein_zero_shadow(T, n_y, n_x), Val(EnzymeRules.width(config)))
    end
    tape = (copy(A.val), copy(B.val), copy(primal.g_x), copy(primal.h_x), shadow)
    returned_primal = EnzymeRules.needs_primal(config) ? primal : nothing
    return EnzymeRules.AugmentedReturn(returned_primal, shadow, tape)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(klein_map)},
        RT::Type,
        tape,
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}};
        threshold = 1.0e-6,
    ) where {T <: _KLEIN_AD_FLOAT}
    Aval, Bval, g_val, h_val, shadow = tape
    shadow === nothing && return (nothing, nothing)

    N = EnzymeRules.width(config)
    plan = _klein_bigk_plan(Aval, Bval, g_val, h_val)
    for i in 1:N
        sh = N == 1 ? shadow : shadow[i]
        bars = _klein_bigk_vjp(plan, sh.g_x, sh.h_x)
        if !(typeof(A) <: Const)
            dA = N == 1 ? A.dval : A.dval[i]
            dA .+= bars.A
        end
        if !(typeof(B) <: Const)
            dB = N == 1 ? B.dval : B.dval[i]
            dB .+= bars.B
        end
        fill!(sh.g_x, zero(T))
        fill!(sh.h_x, zero(T))
    end
    return (nothing, nothing)
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(klein_map!)},
        RT::Type,
        g_x::Annotation{<:StridedMatrix{T}},
        h_x::Annotation{<:StridedMatrix{T}},
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}};
        threshold = 1.0e-6,
    ) where {T <: _KLEIN_AD_FLOAT}
    primal = MatrixEquationsAD.klein_map!(g_x.val, h_x.val, A.val, B.val; threshold)
    N = EnzymeRules.width(config)
    plan = _klein_structured_plan(A.val, B.val, g_x.val, h_x.val)

    return_shadows = if RT <: Const || !EnzymeRules.needs_shadow(config)
        nothing
    else
        Vector{Any}(undef, N)
    end

    for i in 1:N
        deriv = _klein_structured_jvp(
            plan,
            _klein_tangent(A, i, size(A.val), T, N),
            _klein_tangent(B, i, size(B.val), T, N),
        )
        dg = _klein_shadow(g_x, i, N)
        dh = _klein_shadow(h_x, i, N)
        if dg !== nothing
            copyto!(dg, deriv.g_x)
        end
        if dh !== nothing
            copyto!(dh, deriv.h_x)
        end
        if return_shadows !== nothing
            return_shadows[i] = (; g_x = deriv.g_x, h_x = deriv.h_x)
        end
    end

    if RT <: DuplicatedNoNeed
        return return_shadows[1]
    elseif RT <: BatchDuplicatedNoNeed
        return Tuple(return_shadows)
    elseif RT <: Duplicated
        return Duplicated(primal, return_shadows[1])
    elseif RT <: BatchDuplicated
        return BatchDuplicated(primal, Tuple(return_shadows))
    end
    return EnzymeRules.needs_primal(config) ? primal : nothing
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(klein_map!)},
        RT::Type,
        g_x::Annotation{<:StridedMatrix{T}},
        h_x::Annotation{<:StridedMatrix{T}},
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}};
        threshold = 1.0e-6,
    ) where {T <: _KLEIN_AD_FLOAT}
    primal = MatrixEquationsAD.klein_map!(g_x.val, h_x.val, A.val, B.val; threshold)
    n_x = size(h_x.val, 1)
    n_y = size(g_x.val, 1)
    shadow = if RT <: Const
        nothing
    elseif EnzymeRules.width(config) == 1
        _klein_zero_shadow(T, n_y, n_x)
    else
        ntuple(_ -> _klein_zero_shadow(T, n_y, n_x), Val(EnzymeRules.width(config)))
    end
    tape = (copy(A.val), copy(B.val), copy(g_x.val), copy(h_x.val), shadow)
    returned_primal = EnzymeRules.needs_primal(config) ? primal : nothing
    return EnzymeRules.AugmentedReturn(returned_primal, shadow, tape)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(klein_map!)},
        RT::Type,
        tape,
        g_x::Annotation{<:StridedMatrix{T}},
        h_x::Annotation{<:StridedMatrix{T}},
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}};
        threshold = 1.0e-6,
    ) where {T <: _KLEIN_AD_FLOAT}
    Aval, Bval, g_val, h_val, return_shadow = tape
    N = EnzymeRules.width(config)
    plan = _klein_structured_plan(Aval, Bval, g_val, h_val)

    for i in 1:N
        g_bar = zeros(T, size(g_val))
        h_bar = zeros(T, size(h_val))

        dg = _klein_shadow(g_x, i, N)
        dh = _klein_shadow(h_x, i, N)
        if dg !== nothing
            g_bar .+= dg
            fill!(dg, zero(T))
        end
        if dh !== nothing
            h_bar .+= dh
            fill!(dh, zero(T))
        end
        if return_shadow !== nothing
            _klein_add_shadow!(g_bar, h_bar, N == 1 ? return_shadow : return_shadow[i])
        end

        bars = _klein_structured_vjp(plan, g_bar, h_bar)
        if !(typeof(A) <: Const)
            dA = N == 1 ? A.dval : A.dval[i]
            dA .+= bars.A
        end
        if !(typeof(B) <: Const)
            dB = N == 1 ? B.dval : B.dval[i]
            dB .+= bars.B
        end
    end
    return (nothing, nothing, nothing, nothing)
end
