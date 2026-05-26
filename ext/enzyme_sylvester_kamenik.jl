# Enzyme rules for `gsylv_kamenik` (allocating + in-place).
#
# Forward (JVP): Width = 1 and `BatchDuplicated` / `BatchDuplicatedNoNeed`
#   supported for both forms; the Kamenik factorisation is shared
#   across the primal and every tangent solve.
# Reverse (VJP): Width = 1 only for both forms.
#
# Derivations and pullback formulas: `docs/src/sylvester_kamenik.md`.

# Allocating form: gsylv_kamenik(A, B, C, D).

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(gsylv_kamenik)},
        ::Type{RT},
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:StridedMatrix{T}},
        D::Annotation{<:StridedMatrix{T}},
    ) where {
        RT <: Union{
            Const, Duplicated, DuplicatedNoNeed,
            BatchDuplicated, BatchDuplicatedNoNeed,
        },
        T <: Union{Float64},
    }
    N = EnzymeRules.width(config)
    n = size(A.val, 1)
    m = size(C.val, 1)

    # Docs § Enzyme JVP (forward) — the JVP equation
    #   A · dX + B · dX · K = dD − dA·X − dB·(X·K) − B·X·(dC ⊗ C + C ⊗ dC)
    # is itself a Kamenik order-2 system with the SAME triple (A, B, C).
    # Factor once, reuse for the primal and all N tangent solves.
    cache = MatrixEquationsAD._gsylv_kamenik_factor(A.val, B.val, C.val)
    X = copy(D.val)
    MatrixEquationsAD._gsylv_kamenik_solve!(cache, X)

    is_A_const = typeof(A) <: Const
    is_B_const = typeof(B) <: Const
    is_C_const = typeof(C) <: Const
    is_D_const = typeof(D) <: Const

    # Precompute the two RHS factors shared across all tangents.
    #   BX = B · X        — appears in the dC term.
    #   XK = X · (C ⊗ C)  — appears in the dB term.
    # XK is computed via the same two-pass GEMM trick the primal uses for
    # `D · (U_C ⊗ U_C)` (docs § Primal algorithm, stage 2 closing line):
    # reshape (n × m²) as (n, m, m); contract first against C on the
    # `j`-axis, then per `l` against C on the `i`-axis.
    BX = if !is_B_const || !is_C_const
        M = Matrix{T}(undef, n, m * m)
        LinearAlgebra.mul!(M, B.val, X)
        M
    end
    XK = if !is_B_const
        out = Matrix{T}(undef, n, m * m)
        s1 = Array{T}(undef, n, m, m)
        s2 = Array{T}(undef, n, m, m)
        LinearAlgebra.mul!(reshape(s1, n * m, m), reshape(X, n * m, m), C.val)
        for l in 1:m
            @views LinearAlgebra.mul!(s2[:, :, l], s1[:, :, l], C.val)
        end
        copyto!(out, reshape(s2, n, m * m))
        out
    end
    # Per-tangent scratch for `BX · (dC ⊗ C + C ⊗ dC)`.
    s1 = !is_C_const ? Array{T}(undef, n, m, m) : nothing
    s2 = !is_C_const ? Array{T}(undef, n, m, m) : nothing

    dXs = ntuple(Val(N)) do i
        Base.@_inline_meta
        # Assemble the JVP RHS for tangent `i`, term-by-term following the
        # JVP equation above. Const arguments contribute nothing.
        rhs = is_D_const ?
            zeros(T, n, m * m) :
            copy(N == 1 ? D.dval : D.dval[i])
        if !is_A_const
            # − dA · X
            dA = N == 1 ? A.dval : A.dval[i]
            LinearAlgebra.mul!(rhs, dA, X, -one(T), one(T))
        end
        if !is_B_const
            # − dB · (X · K) = − dB · XK
            dB = N == 1 ? B.dval : B.dval[i]
            LinearAlgebra.mul!(rhs, dB, XK, -one(T), one(T))
        end
        if !is_C_const
            # Two two-pass GEMMs for the Kronecker product rule
            # d(C ⊗ C) = (dC ⊗ C) + (C ⊗ dC) — same trick as XK above
            # but with one factor swapped in turn.
            dC = N == 1 ? C.dval : C.dval[i]
            # BX · (C ⊗ dC):  contract C on outer (j) leg, dC on inner (i).
            LinearAlgebra.mul!(reshape(s1, n * m, m), reshape(BX, n * m, m), C.val)
            for l in 1:m
                @views LinearAlgebra.mul!(s2[:, :, l], s1[:, :, l], dC)
            end
            rhs .-= reshape(s2, n, m * m)
            # BX · (dC ⊗ C):  contract dC on outer leg, C on inner.
            LinearAlgebra.mul!(reshape(s1, n * m, m), reshape(BX, n * m, m), dC)
            for l in 1:m
                @views LinearAlgebra.mul!(s2[:, :, l], s1[:, :, l], C.val)
            end
            rhs .-= reshape(s2, n, m * m)
        end
        # Solve the JVP system. Same factorisation as the primal — only the
        # RHS changes per tangent.
        MatrixEquationsAD._gsylv_kamenik_solve!(cache, rhs)
        rhs
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
        func::Const{typeof(gsylv_kamenik)},
        ::Type{RT},
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:StridedMatrix{T}},
        D::Annotation{<:StridedMatrix{T}},
    ) where {RT, T <: Union{Float64}}
    EnzymeRules.width(config) == 1 ||
        error("gsylv_kamenik Enzyme reverse rule supports Width = 1 only")
    X = gsylv_kamenik(A.val, B.val, C.val, D.val)
    dX = zero(X)
    primal = EnzymeRules.needs_primal(config) ? X : nothing
    tape = (copy(X), dX, copy(A.val), copy(B.val), copy(C.val))
    return EnzymeRules.AugmentedReturn(primal, dX, tape)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(gsylv_kamenik)},
        ::Type{RT},
        tape,
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:StridedMatrix{T}},
        D::Annotation{<:StridedMatrix{T}},
    ) where {RT, T <: Union{Float64}}
    X, dX, Aval, Bval, Cval = tape
    n = size(Aval, 1)
    m = size(Cval, 1)

    # Docs § Enzyme VJP, Step 1: adjoint Sylvester
    #   A'·Λ + B'·Λ·(C' ⊗ C') = X̄.
    # Same equation shape as the primal with transposed inputs, so a
    # second `gsylv_kamenik` call on (A', B', C') solves it. `lu` / `schur`
    # want plain `Matrix{T}`, not lazy `Transpose` wrappers, so materialise.
    Xbar = dX
    Λ = gsylv_kamenik(
        Matrix(transpose(Aval)),
        Matrix(transpose(Bval)),
        Matrix(transpose(Cval)),
        Xbar,
    )

    # Docs § Enzyme VJP, Step 2: parameter cotangents
    #   D̄ += Λ,   Ā -= Λ · X',   B̄ -= Λ · (X · K)'.

    # D̄ += Λ
    if !(typeof(D) <: Const)
        D.dval .+= Λ
    end

    # Ā -= Λ · X'
    if !(typeof(A) <: Const)
        LinearAlgebra.mul!(A.dval, Λ, transpose(X), -one(T), one(T))
    end

    # B̄ -= Λ · (X · K)'. Compute X·K via the same two-pass GEMM trick the
    # primal and forward rule use — no m²×m² Kronecker materialised.
    if !(typeof(B) <: Const)
        XK = Matrix{T}(undef, n, m * m)
        T1 = Array{T}(undef, n, m, m)
        LinearAlgebra.mul!(reshape(T1, n * m, m), reshape(X, n * m, m), Cval)
        XK_t = reshape(XK, n, m, m)
        for l in 1:m
            @views LinearAlgebra.mul!(XK_t[:, :, l], T1[:, :, l], Cval)
        end
        LinearAlgebra.mul!(B.dval, Λ, transpose(XK), -one(T), one(T))
    end

    # Docs § Enzyme VJP, Step 3: pullback through the Kronecker leg
    #   K̄ = − R    with R := X' · B' · Λ ∈ ℝ^{m² × m²},
    # then the column-major Kronecker convention
    #   (C ⊗ C)_{(i-1)m+k, (j-1)m+l} = C_{i,j} · C_{k,l}
    # gives, after applying d(C ⊗ C) = (dC ⊗ C) + (C ⊗ dC) and pairing,
    #   C̄_{i,j} = Σ_{k,l} K̄_{(i-1)m+k, (j-1)m+l} · C_{k,l}
    #           + Σ_{k,l} K̄_{(k-1)m+i, (l-1)m+j} · C_{k,l}.
    # We accumulate `−Σ_{k,l} R_{…} · C_{k,l}` directly into C̄, absorbing
    # the minus sign from `K̄ = −R`.
    if !(typeof(C) <: Const)
        BΛ = Matrix{T}(undef, n, m * m)
        LinearAlgebra.mul!(BΛ, transpose(Bval), Λ)
        R = Matrix{T}(undef, m * m, m * m)
        LinearAlgebra.mul!(R, transpose(X), BΛ)
        @inbounds for j in 1:m, i in 1:m
            s = zero(T)
            for l in 1:m, k in 1:m
                s += R[(i - 1) * m + k, (j - 1) * m + l] * Cval[k, l]   # leg 1
                s += R[(k - 1) * m + i, (l - 1) * m + j] * Cval[k, l]   # leg 2
            end
            C.dval[i, j] -= s
        end
    end

    fill!(Xbar, zero(T))
    return (nothing, nothing, nothing, nothing)
end

# In-place form: gsylv_kamenik!(D, A, B, C).
#
# The `Duplicated(D, D̄)` slot is both RHS-in / solution-out for the
# primal. Reverse: `D̄` enters as X̄ (cotangent of the post-call value)
# and the rule overwrites it with Λ (cotangent of the pre-call value).
# Forward: `D.dval` enters as dD and is overwritten with dX.

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(gsylv_kamenik!)},
        ::Type{RT},
        D::Annotation{<:StridedMatrix{T}},
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:StridedMatrix{T}},
    ) where {RT, T <: Union{Float64}}
    n = size(A.val, 1)
    m = size(C.val, 1)

    # Same JVP equation as the allocating form. Factor once, solve primal
    # into D.val, then reuse the cache for each tangent.
    cache = MatrixEquationsAD._gsylv_kamenik_factor(A.val, B.val, C.val)
    MatrixEquationsAD._gsylv_kamenik_solve!(cache, D.val)

    typeof(D) <: Const && return nothing

    N = EnzymeRules.width(config)
    X = D.val   # after solve!, D.val is the primal solution X.

    is_A_const = typeof(A) <: Const
    is_B_const = typeof(B) <: Const
    is_C_const = typeof(C) <: Const

    # Same shared factors as the allocating forward rule — see comments
    # there for the math (BX = B·X, XK = X·K via two-pass GEMM).
    BX = if !is_B_const || !is_C_const
        M = Matrix{T}(undef, n, m * m)
        LinearAlgebra.mul!(M, B.val, X)
        M
    end
    XK = if !is_B_const
        out = Matrix{T}(undef, n, m * m)
        s1 = Array{T}(undef, n, m, m)
        s2 = Array{T}(undef, n, m, m)
        LinearAlgebra.mul!(reshape(s1, n * m, m), reshape(X, n * m, m), C.val)
        for l in 1:m
            @views LinearAlgebra.mul!(s2[:, :, l], s1[:, :, l], C.val)
        end
        copyto!(out, reshape(s2, n, m * m))
        out
    end
    s1 = !is_C_const ? Array{T}(undef, n, m, m) : nothing
    s2 = !is_C_const ? Array{T}(undef, n, m, m) : nothing

    for i in 1:N
        # Build the JVP RHS in place on D.dval (entering as dD), then
        # solve! overwrites with dX. Reads happen before writes so the
        # rhs / output aliasing is safe.
        dX = N == 1 ? D.dval : D.dval[i]
        if !is_A_const
            # − dA · X
            dA = N == 1 ? A.dval : A.dval[i]
            LinearAlgebra.mul!(dX, dA, X, -one(T), one(T))
        end
        if !is_B_const
            # − dB · (X · K) = − dB · XK
            dB = N == 1 ? B.dval : B.dval[i]
            LinearAlgebra.mul!(dX, dB, XK, -one(T), one(T))
        end
        if !is_C_const
            # − BX · (dC ⊗ C + C ⊗ dC), as two two-pass GEMMs.
            dC = N == 1 ? C.dval : C.dval[i]
            LinearAlgebra.mul!(reshape(s1, n * m, m), reshape(BX, n * m, m), C.val)
            for l in 1:m
                @views LinearAlgebra.mul!(s2[:, :, l], s1[:, :, l], dC)
            end
            dX .-= reshape(s2, n, m * m)   # − BX · (C ⊗ dC)
            LinearAlgebra.mul!(reshape(s1, n * m, m), reshape(BX, n * m, m), dC)
            for l in 1:m
                @views LinearAlgebra.mul!(s2[:, :, l], s1[:, :, l], C.val)
            end
            dX .-= reshape(s2, n, m * m)   # − BX · (dC ⊗ C)
        end
        # Solve the Kamenik system on the assembled RHS — in-place into dX.
        MatrixEquationsAD._gsylv_kamenik_solve!(cache, dX)
    end
    return nothing
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(gsylv_kamenik!)},
        ::Type{RT},
        D::Annotation{<:StridedMatrix{T}},
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:StridedMatrix{T}},
    ) where {RT, T <: Union{Float64}}
    EnzymeRules.width(config) == 1 ||
        error("gsylv_kamenik! Enzyme reverse rule supports Width = 1 only")
    gsylv_kamenik!(D.val, A.val, B.val, C.val)
    tape = (copy(D.val), copy(A.val), copy(B.val), copy(C.val))
    primal = EnzymeRules.needs_primal(config) ? D.val : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, tape)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(gsylv_kamenik!)},
        ::Type{RT},
        tape,
        D::Annotation{<:StridedMatrix{T}},
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}},
        C::Annotation{<:StridedMatrix{T}},
    ) where {RT, T <: Union{Float64}}
    if typeof(D) <: Const
        return (nothing, nothing, nothing, nothing)
    end

    X, Aval, Bval, Cval = tape
    n = size(Aval, 1)
    m = size(Cval, 1)

    # Docs § Enzyme VJP — same math as the allocating reverse rule, only
    # the D̄ delivery differs.
    #
    # Step 1: adjoint Sylvester  A'·Λ + B'·Λ·(C' ⊗ C') = X̄.
    # `gsylv_kamenik!` overwrites its first arg, so seed it with the
    # upstream X̄ delivered in D.dval; it returns Λ in that same buffer.
    Λ = copy(D.dval)
    gsylv_kamenik!(
        Λ,
        Matrix(transpose(Aval)),
        Matrix(transpose(Bval)),
        Matrix(transpose(Cval))
    )

    # Step 2 (D̄ delivery, in-place variant): D̄ ← Λ. Because `gsylv_kamenik!`
    # consumed the input buffer, the output-side cotangent X̄ is replaced
    # by the input-side cotangent Λ on the same `D.dval` slot.
    copyto!(D.dval, Λ)

    # Step 2 (Ā): Ā -= Λ · X'
    if !(typeof(A) <: Const)
        LinearAlgebra.mul!(A.dval, Λ, transpose(X), -one(T), one(T))
    end

    # Step 2 (B̄): B̄ -= Λ · (X · K)' — X·K via the same two-pass GEMM.
    if !(typeof(B) <: Const)
        XK = Matrix{T}(undef, n, m * m)
        T1 = Array{T}(undef, n, m, m)
        LinearAlgebra.mul!(reshape(T1, n * m, m), reshape(X, n * m, m), Cval)
        XK_t = reshape(XK, n, m, m)
        for l in 1:m
            @views LinearAlgebra.mul!(XK_t[:, :, l], T1[:, :, l], Cval)
        end
        LinearAlgebra.mul!(B.dval, Λ, transpose(XK), -one(T), one(T))
    end

    # Step 3 (C̄): K̄ = −R with R = X' · B' · Λ; sum the two Kronecker legs
    # (same formula as the allocating reverse rule above).
    if !(typeof(C) <: Const)
        BΛ = Matrix{T}(undef, n, m * m)
        LinearAlgebra.mul!(BΛ, transpose(Bval), Λ)
        R = Matrix{T}(undef, m * m, m * m)
        LinearAlgebra.mul!(R, transpose(X), BΛ)
        @inbounds for j in 1:m, i in 1:m
            s = zero(T)
            for l in 1:m, k in 1:m
                s += R[(i - 1) * m + k, (j - 1) * m + l] * Cval[k, l]   # leg 1
                s += R[(k - 1) * m + i, (l - 1) * m + j] * Cval[k, l]   # leg 2
            end
            C.dval[i, j] -= s
        end
    end

    return (nothing, nothing, nothing, nothing)
end
