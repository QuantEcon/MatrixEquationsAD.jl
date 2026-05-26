const _KLEIN_AD_FLOAT = Union{Float32, Float64}

# Enzyme forward rule for the OOP `klein_map` — routes through the
# big-K factorisation (docs § Big-K JVP). Build the plan once (which
# LU-factorises the dense (n n_x) × (n n_x) K), then issue one
# back-substitution per tangent direction.

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(klein_map)},
        RT::Type,
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}};
        threshold = 1.0e-6,
    ) where {T <: _KLEIN_AD_FLOAT}
    # Primal first — produces (g_x, h_x) needed to build the plan.
    primal = MatrixEquationsAD.klein_map(A.val, B.val; threshold)
    if RT <: Const || !EnzymeRules.needs_shadow(config)
        return EnzymeRules.needs_primal(config) ? primal : nothing
    end

    # Cache LU of K (docs § Big-K JVP Step 2) — reused across all
    # tangent directions.
    N = EnzymeRules.width(config)
    plan = _klein_bigk_plan(A.val, B.val, primal.g_x, primal.h_x)
    shadows = ntuple(Val(N)) do i
        Base.@_inline_meta
        # Const-annotation handling: a Const input contributes a zero
        # tangent matrix on that side.
        dA = if typeof(A) <: Const
            zeros(T, size(A.val))
        else
            N == 1 ? A.dval : A.dval[i]
        end
        dB = if typeof(B) <: Const
            zeros(T, size(B.val))
        else
            N == 1 ? B.dval : B.dval[i]
        end
        # One JVP solve per direction (Step 3): K⁻¹ · rhs, split into
        # (d g_x, d h_x).
        _klein_bigk_jvp(
            plan,
            dA,
            dB,
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
        (; g_x = zeros(T, n_y, n_x), h_x = zeros(T, n_x, n_x))
    else
        ntuple(
            _ -> (; g_x = zeros(T, n_y, n_x), h_x = zeros(T, n_x, n_x)),
            Val(EnzymeRules.width(config)),
        )
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
    # Docs § Big-K VJP: Λ = K⁻ᵀ · u, then Ā += −Λ · h_x' · Ψ', B̄ += −Λ · Ψ'.
    # The plan (incl. LU of K) is rebuilt from the tape's A, B, g, h
    # snapshots — same K as the augmented_primal would have produced.
    Aval, Bval, g_val, h_val, shadow = tape
    shadow === nothing && return (nothing, nothing)

    N = EnzymeRules.width(config)
    plan = _klein_bigk_plan(Aval, Bval, g_val, h_val)
    for i in 1:N
        sh = N == 1 ? shadow : shadow[i]
        # One VJP solve per cotangent direction: Λ-solve + two outer
        # products into bars.A / bars.B.
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
    # Enzyme forward rule for the in-place `klein_map!` — routes through
    # the reduced-Sylvester factorisation (docs § Reduced-Sylvester JVP).
    # Build the plan once (one LU of n × n C₀, two Schurs on n_y × n_y and
    # n_x × n_x), then issue one reduced-Sylvester solve per tangent.
    primal = MatrixEquationsAD.klein_map!(g_x.val, h_x.val, A.val, B.val; threshold)
    N = EnzymeRules.width(config)
    plan = _klein_structured_plan(A.val, B.val, g_x.val, h_x.val)

    if RT <: Const || !EnzymeRules.needs_shadow(config)
        for i in 1:N
            dA = if typeof(A) <: Const
                zeros(T, size(A.val))
            else
                N == 1 ? A.dval : A.dval[i]
            end
            dB = if typeof(B) <: Const
                zeros(T, size(B.val))
            else
                N == 1 ? B.dval : B.dval[i]
            end
            deriv = _klein_structured_jvp(plan, dA, dB)
            dg = if typeof(g_x) <: Const
                nothing
            else
                N == 1 ? g_x.dval : g_x.dval[i]
            end
            dh = if typeof(h_x) <: Const
                nothing
            else
                N == 1 ? h_x.dval : h_x.dval[i]
            end
            if dg !== nothing
                copyto!(dg, deriv.g_x)
            end
            if dh !== nothing
                copyto!(dh, deriv.h_x)
            end
        end
        return EnzymeRules.needs_primal(config) ? primal : nothing
    end

    return_shadows = ntuple(Val(N)) do i
        Base.@_inline_meta
        dA = if typeof(A) <: Const
            zeros(T, size(A.val))
        else
            N == 1 ? A.dval : A.dval[i]
        end
        dB = if typeof(B) <: Const
            zeros(T, size(B.val))
        else
            N == 1 ? B.dval : B.dval[i]
        end
        deriv = _klein_structured_jvp(plan, dA, dB)
        dg = if typeof(g_x) <: Const
            nothing
        else
            N == 1 ? g_x.dval : g_x.dval[i]
        end
        dh = if typeof(h_x) <: Const
            nothing
        else
            N == 1 ? h_x.dval : h_x.dval[i]
        end
        if dg !== nothing
            copyto!(dg, deriv.g_x)
        end
        if dh !== nothing
            copyto!(dh, deriv.h_x)
        end
        (; g_x = deriv.g_x, h_x = deriv.h_x)
    end

    if RT <: DuplicatedNoNeed
        return return_shadows[1]
    elseif RT <: BatchDuplicatedNoNeed
        return return_shadows
    elseif RT <: Duplicated
        return Duplicated(primal, return_shadows[1])
    elseif RT <: BatchDuplicated
        return BatchDuplicated(primal, return_shadows)
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
        (; g_x = zeros(T, n_y, n_x), h_x = zeros(T, n_x, n_x))
    else
        ntuple(
            _ -> (; g_x = zeros(T, n_y, n_x), h_x = zeros(T, n_x, n_x)),
            Val(EnzymeRules.width(config)),
        )
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
    # Docs § Reduced-Sylvester VJP: adjoint Stein + transposed C₀ solve,
    # then the same Ā / B̄ outer products as the big-K rule. Plan rebuilt
    # from the tape's A/B/g/h snapshots — same C₀ and Schurs as
    # augmented_primal would have produced.
    Aval, Bval, g_val, h_val, return_shadow = tape
    N = EnzymeRules.width(config)
    plan = _klein_structured_plan(Aval, Bval, g_val, h_val)

    for i in 1:N
        # Per-direction cotangent accumulator: sum the shadow buffer
        # (from `Duplicated(g_x, ·)`) and the augmented-primal return
        # shadow (from the `_NoNeed` variant of the rule).
        g_bar = zeros(T, size(g_val))
        h_bar = zeros(T, size(h_val))

        dg = if typeof(g_x) <: Const
            nothing
        else
            N == 1 ? g_x.dval : g_x.dval[i]
        end
        dh = if typeof(h_x) <: Const
            nothing
        else
            N == 1 ? h_x.dval : h_x.dval[i]
        end
        if dg !== nothing
            g_bar .+= dg
            fill!(dg, zero(T))
        end
        if dh !== nothing
            h_bar .+= dh
            fill!(dh, zero(T))
        end
        if return_shadow !== nothing
            sh = N == 1 ? return_shadow : return_shadow[i]
            g_bar .+= sh.g_x
            h_bar .+= sh.h_x
            fill!(sh.g_x, zero(T))
            fill!(sh.h_x, zero(T))
        end

        # Single VJP call: corrected adjoint Stein → C₀⁻ᵀ solve → bars.A, bars.B.
        bars = _klein_structured_vjp(plan, g_bar, h_bar)
        if !(typeof(A) <: Const)
            dA = N == 1 ? A.dval : A.dval[i]
            dA .+= bars.A          # Ā += −Λ · h_x' · Ψ'
        end
        if !(typeof(B) <: Const)
            dB = N == 1 ? B.dval : B.dval[i]
            dB .+= bars.B          # B̄ += −Λ · Ψ'
        end
    end
    return (nothing, nothing, nothing, nothing)
end
