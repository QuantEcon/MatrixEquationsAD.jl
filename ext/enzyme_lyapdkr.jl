function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(lyapdkr)},
        ::Type{RT},
        A::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:StridedMatrix{T}};
        M_ws::Union{Nothing, StridedMatrix{T}} = nothing,
    ) where {RT <: Union{Const, Duplicated, DuplicatedNoNeed, BatchDuplicated, BatchDuplicatedNoNeed}, T <: Union{Float32, Float64}}
    # Docs § ForwardDiff JVP — same JVP equation as the FD path:
    #   vec(dX_raw) = M⁻¹ · vec(dC + dA·X·A' + A·X·dA'),
    #   dX = P(dX_raw).
    # Build M = I − A ⊗ A and its LU once; reuse for primal + all N tangents.
    N = EnzymeRules.width(config)
    n = size(A.val, 1)
    M = isnothing(M_ws) ? Matrix{T}(undef, n * n, n * n) : M_ws
    build_M!!(M, A.val)
    F = lu!(M)
    X = copy(C.val)
    ldiv!(F, vec(X))
    symmetrize!!(X)

    # Per-direction RHS construction, packed into one n × n × N tensor for
    # a single BLAS-3 multi-RHS solve.
    #   XAt = X · A',  AX = A · X — shared across all tangents (don't
    #   depend on dA / dC).
    RHS = Array{T, 3}(undef, n, n, N)
    if !(typeof(A) <: Const)
        XAt = X * A.val'
        AX = A.val * X
    end
    @inbounds for i in 1:N
        dX = view(RHS, :, :, i)
        # dX ← dC_i (or zero if C is Const).
        if typeof(C) <: Const
            fill!(dX, zero(T))
        else
            dX .= N == 1 ? C.dval : C.dval[i]
        end
        if !(typeof(A) <: Const)
            # dX += dA · X · A' + A · X · dA'.
            dA = N == 1 ? A.dval : A.dval[i]
            mul!(dX, dA, XAt, one(T), one(T))
            mul!(dX, AX, dA', one(T), one(T))
        end
    end
    # Single multi-RHS solve over all N stacked tangent directions.
    ldiv!(F, reshape(RHS, n * n, N))
    # Symmetric projection per direction (docs § Primal: P(·) on the
    # reshape-as-(n,n) output).
    @inbounds for i in 1:N
        symmetrize!!(view(RHS, :, :, i))
    end
    # Materialize shadows as standalone Matrix — Enzyme requires the
    # shadow type to match the primal (Matrix), not a SubArray view.
    dXs = ntuple(Val(N)) do i
        Base.@_inline_meta
        copy(view(RHS, :, :, i))
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
        M_ws::Union{Nothing, StridedMatrix{T}} = nothing,
    ) where {RT, T <: Union{Float32, Float64}}
    n = size(A.val, 1)
    M = isnothing(M_ws) ? Matrix{T}(undef, n * n, n * n) : M_ws
    build_M!!(M, A.val)
    F = lu!(M)
    X = copy(C.val)
    ldiv!(F, vec(X))
    symmetrize!!(X)
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
        M_ws::Union{Nothing, StridedMatrix{T}} = nothing,
    ) where {RT, T <: Union{Float32, Float64}}
    # Docs § Enzyme VJP:
    #   1. Symmetrise the upstream cotangent: S = P(X̄).
    #   2. Solve the transposed Kronecker system: vec(Y) = M⁻ᵀ · vec(S).
    #   3. Parameter cotangents
    #        C̄ += Y,
    #        Ā += Y·A·X' + Y'·A·X.
    # The LU `F` of M is re-used from the tape (built once by
    # augmented_primal); no re-LU on the reverse pass.
    X, dXs, F, Aval = tape
    N = EnzymeRules.width(config)
    n = size(X, 1)

    for i in 1:N
        Xbar = N == 1 ? dXs : dXs[i]
        # Step 1 + 2: Y = M⁻ᵀ · vec(P(X̄)), in place.
        Y = copy(Xbar)
        symmetrize!!(Y)
        ldiv!(transpose(F), vec(Y))

        if !(typeof(C) <: Const)
            # Step 3: C̄ += Y.
            dC = N == 1 ? C.dval : C.dval[i]
            dC .+= Y
        end
        if !(typeof(A) <: Const)
            # Step 3: Ā += Y·A·X' + Y'·A·X — two GEMMs into dA.
            dA = N == 1 ? A.dval : A.dval[i]
            tmp = Y * Aval
            mul!(dA, tmp, X', one(T), one(T))     # Y · A · X'
            tmp = Y' * Aval
            mul!(dA, tmp, X, one(T), one(T))      # Y' · A · X
        end

        fill!(Xbar, zero(T))
    end

    return (nothing, nothing)
end

# ─── Enzyme rules for lyapdkr! ──────────────────────────────────────────────
#
# `lyapdkr!` writes the solution into the caller-supplied `X`. The rule
# overwrites `X.val` with the primal and (in forward mode) `X.dval[i]` with
# each tangent solution. Reverse mode pulls upstream gradients from
# `X.dval`, solves the adjoint into `A.dval` / `C.dval`, then zeros
# `X.dval` (X was overwritten so any prior gradient on X is gone).

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(lyapdkr!)},
        ::Type{RT},
        X::Annotation{<:StridedMatrix{T}},
        A::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:StridedMatrix{T}};
        M_ws::Union{Nothing, StridedMatrix{T}} = nothing,
    ) where {RT, T <: Union{Float32, Float64}}
    # Same docs § ForwardDiff JVP as the OOP forward rule. Build M and
    # its LU once, solve the primal into the caller-supplied X.val, then
    # solve each tangent into the matching X.dval slot.
    n = size(A.val, 1)
    M = isnothing(M_ws) ? Matrix{T}(undef, n * n, n * n) : M_ws
    build_M!!(M, A.val)
    F = lu!(M)
    copyto!(X.val, C.val)
    ldiv!(F, vec(X.val))
    symmetrize!!(X.val)

    typeof(X) <: Const && return nothing

    N = EnzymeRules.width(config)
    for i in 1:N
        # Per-direction RHS = dC + dA·X·A' + A·X·dA' assembled in-place
        # into the caller's shadow buffer, then solved + projected.
        dX = N == 1 ? X.dval : X.dval[i]
        if typeof(C) <: Const
            fill!(dX, zero(T))
        else
            copyto!(dX, N == 1 ? C.dval : C.dval[i])
        end
        if !(typeof(A) <: Const)
            dA = N == 1 ? A.dval : A.dval[i]
            dX .+= dA * X.val * A.val'      # dA · X · A'
            dX .+= A.val * X.val * dA'      # A · X · dA'
        end
        ldiv!(F, vec(dX))                   # vec(dX) ← M⁻¹ · vec(rhs)
        symmetrize!!(dX)                    # dX ← P(dX)
    end
    return nothing
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(lyapdkr!)},
        ::Type{RT},
        X::Annotation{<:StridedMatrix{T}},
        A::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:StridedMatrix{T}};
        M_ws::Union{Nothing, StridedMatrix{T}} = nothing,
    ) where {RT, T <: Union{Float32, Float64}}
    n = size(A.val, 1)
    M = isnothing(M_ws) ? Matrix{T}(undef, n * n, n * n) : M_ws
    build_M!!(M, A.val)
    F = lu!(M)
    copyto!(X.val, C.val)
    ldiv!(F, vec(X.val))
    symmetrize!!(X.val)
    tape = (copy(X.val), F, copy(A.val))
    return EnzymeRules.AugmentedReturn(nothing, nothing, tape)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(lyapdkr!)},
        ::Type{RT},
        tape,
        X::Annotation{<:StridedMatrix{T}},
        A::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:StridedMatrix{T}};
        M_ws::Union{Nothing, StridedMatrix{T}} = nothing,
    ) where {RT, T <: Union{Float32, Float64}}
    # Same docs § Enzyme VJP as the OOP reverse rule:
    #   1. Y = M⁻ᵀ · vec(P(X̄))    (LU `F` of M reused from the tape).
    #   2. C̄ += Y,  Ā += Y·A·X' + Y'·A·X.
    # Primal X read from the tape (`Xval`) rather than X.val because the
    # in-place call may have been chained.
    Xval, F, Aval = tape
    N = EnzymeRules.width(config)
    n = size(Xval, 1)

    typeof(X) <: Const && return (nothing, nothing, nothing)

    for i in 1:N
        Xbar = N == 1 ? X.dval : X.dval[i]
        # Step 1: Y = M⁻ᵀ · vec(P(X̄)).
        Y = copy(Xbar)
        symmetrize!!(Y)
        ldiv!(transpose(F), vec(Y))

        if !(typeof(C) <: Const)
            # Step 2: C̄ += Y.
            dC = N == 1 ? C.dval : C.dval[i]
            dC .+= Y
        end
        if !(typeof(A) <: Const)
            # Step 2: Ā += Y·A·X' + Y'·A·X.
            dA = N == 1 ? A.dval : A.dval[i]
            tmp = Y * Aval
            mul!(dA, tmp, Xval', one(T), one(T))   # Y · A · X'
            tmp = Y' * Aval
            mul!(dA, tmp, Xval, one(T), one(T))    # Y' · A · X
        end

        fill!(Xbar, zero(T))
    end

    return (nothing, nothing, nothing)
end
