# Enzyme rules for `gsylv_kamenik(A, B, C, D)` and `gsylv_kamenik!(D, A, B, C)`.
#
# Forward (JVP): Width = 1 and `BatchDuplicated` / `BatchDuplicatedNoNeed`
#   supported for both forms; the same Kamenik factorisation is reused
#   across the primal and every tangent solve.
# Reverse (VJP): Width = 1 only for both forms.
#
# See `docs/src/sylvester_kamenik.md` for the derivations.
#
# Math summary:
#
#   Primal:   A·X + B·X·(C ⊗ C) = D,   K := C ⊗ C
#
#   JVP (tangent): differentiating the primal gives a Kamenik equation in
#   `dX` with the SAME coefficient triple (A, B, C):
#
#       A·dX + B·dX·K = dD − dA·X − dB·(X·K) − B·X·(dC ⊗ C + C ⊗ dC).
#
#   VJP (cotangent): Frobenius-adjoint of L[X] = A·X + B·X·K is
#   L*[Λ] = A'·Λ + B'·Λ·K' — same equation shape with transposed inputs.
#   The four parameter pullbacks are
#
#       D̄ += Λ        (allocating)   /   D̄ ← Λ   (in-place)
#       Ā -= Λ · X'
#       B̄ -= Λ · (X · K)'
#       K̄  = −X' · B' · Λ      (m² × m²)
#       C̄[i,j] -= Σ_{k,l} R[(i-1)m+k, (j-1)m+l] · C[k,l]
#                   + Σ_{k,l} R[(k-1)m+i, (l-1)m+j] · C[k,l]
#       with R := X' · B' · Λ = −K̄.
#
# Apply `M · (C₁ ⊗ C₂)` to an `n × m²` matrix `M` via two passes against
# `C₁` and `C₂` (never materialising the `m² × m²` Kronecker). Result
# written into `out` (`n × m²`); two `n × m × m` scratch tensors reused.

# ─── Allocating form: gsylv_kamenik(A, B, C, D) ──────────────────────────────

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

    cache = MatrixEquationsAD._gsylv_kamenik_factor(A.val, B.val, C.val)
    X = copy(D.val)
    MatrixEquationsAD._gsylv_kamenik_solve!(cache, X)

    is_A_const = typeof(A) <: Const
    is_B_const = typeof(B) <: Const
    is_C_const = typeof(C) <: Const
    is_D_const = typeof(D) <: Const

    # BX = B · X, used for any tangent that touches B or C.
    BX = if !is_B_const || !is_C_const
        M = Matrix{T}(undef, n, m * m)
        LinearAlgebra.mul!(M, B.val, X)
        M
    end
    # XK = X · (C ⊗ C) — only the dB term needs this; computed via two
    # passes against C (no m²×m² Kronecker).
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
    kron_buf = !is_C_const ? Matrix{T}(undef, n, m * m) : nothing
    s1 = !is_C_const ? Array{T}(undef, n, m, m) : nothing
    s2 = !is_C_const ? Array{T}(undef, n, m, m) : nothing

    dXs = ntuple(Val(N)) do i
        Base.@_inline_meta
        rhs = is_D_const ?
            zeros(T, n, m * m) :
            copy(N == 1 ? D.dval : D.dval[i])
        if !is_A_const
            dA = N == 1 ? A.dval : A.dval[i]
            LinearAlgebra.mul!(rhs, dA, X, -one(T), one(T))
        end
        if !is_B_const
            dB = N == 1 ? B.dval : B.dval[i]
            LinearAlgebra.mul!(rhs, dB, XK, -one(T), one(T))
        end
        if !is_C_const
            dC = N == 1 ? C.dval : C.dval[i]
            # BX · (C ⊗ dC):  contract C on outer leg, dC on inner.
            LinearAlgebra.mul!(reshape(s1, n * m, m), reshape(BX, n * m, m), C.val)
            for l in 1:m
                @views LinearAlgebra.mul!(s2[:, :, l], s1[:, :, l], dC)
            end
            copyto!(kron_buf, reshape(s2, n, m * m))
            rhs .-= kron_buf
            # BX · (dC ⊗ C):  contract dC on outer leg, C on inner.
            LinearAlgebra.mul!(reshape(s1, n * m, m), reshape(BX, n * m, m), dC)
            for l in 1:m
                @views LinearAlgebra.mul!(s2[:, :, l], s1[:, :, l], C.val)
            end
            copyto!(kron_buf, reshape(s2, n, m * m))
            rhs .-= kron_buf
        end
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

    # Adjoint Sylvester: A'·Λ + B'·Λ·(C' ⊗ C') = X̄. Same shape, transposed
    # inputs. `lu` / `schur` want plain `Matrix{T}`, not lazy `Transpose`.
    Xbar = dX
    Λ = gsylv_kamenik(
        Matrix(transpose(Aval)),
        Matrix(transpose(Bval)),
        Matrix(transpose(Cval)),
        Xbar,
    )

    # D̄ += Λ
    if !(typeof(D) <: Const)
        D.dval .+= Λ
    end

    # Ā -= Λ · X'
    if !(typeof(A) <: Const)
        LinearAlgebra.mul!(A.dval, Λ, transpose(X), -one(T), one(T))
    end

    # B̄ -= Λ · (X · K)' — compute X·K via two-pass GEMM (no m²×m² Kron).
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

    # C̄ via K̄ = −R where R = X' · B' · Λ (m² × m²), then two index sums
    # (one per Kronecker leg).
    if !(typeof(C) <: Const)
        BΛ = Matrix{T}(undef, n, m * m)
        LinearAlgebra.mul!(BΛ, transpose(Bval), Λ)
        R = Matrix{T}(undef, m * m, m * m)
        LinearAlgebra.mul!(R, transpose(X), BΛ)
        @inbounds for j in 1:m, i in 1:m
            s = zero(T)
            for l in 1:m, k in 1:m
                s += R[(i - 1) * m + k, (j - 1) * m + l] * Cval[k, l]
                s += R[(k - 1) * m + i, (l - 1) * m + j] * Cval[k, l]
            end
            C.dval[i, j] -= s
        end
    end

    fill!(Xbar, zero(T))
    return (nothing, nothing, nothing, nothing)
end

# ─── In-place form: gsylv_kamenik!(D, A, B, C) ───────────────────────────────
#
# `gsylv_kamenik!(D, A, B, C)` overwrites `D` with the solution `X`. Enzyme
# sees `D` as a single Annotation that's both RHS-in and solution-out. The
# `Duplicated(D, D̄)` slot accordingly carries `X̄` on entry to the reverse
# pass (cotangent of the post-call value, i.e. the solution); the rule
# writes `Λ` back into the same slot (cotangent of the pre-call value, i.e.
# the original RHS — `D̄ = Λ`).
#
# Forward mode: the `dval` slot starts holding the tangent of the input
# buffer (dD) and the rule overwrites it with the tangent of the output
# (dX), matching how Enzyme treats other mutating writes.

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

    cache = MatrixEquationsAD._gsylv_kamenik_factor(A.val, B.val, C.val)
    MatrixEquationsAD._gsylv_kamenik_solve!(cache, D.val)

    typeof(D) <: Const && return nothing

    N = EnzymeRules.width(config)
    X = D.val

    is_A_const = typeof(A) <: Const
    is_B_const = typeof(B) <: Const
    is_C_const = typeof(C) <: Const

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
    kron_buf = !is_C_const ? Matrix{T}(undef, n, m * m) : nothing
    s1 = !is_C_const ? Array{T}(undef, n, m, m) : nothing
    s2 = !is_C_const ? Array{T}(undef, n, m, m) : nothing

    for i in 1:N
        # In-place RHS = upstream tangent of the input buffer (delivered
        # in D.dval). We write the output tangent back into the same
        # buffer; reads happen before writes inside this loop body.
        dX = N == 1 ? D.dval : D.dval[i]
        # rhs starts as dD = dX (aliased); contributions are subtracted.
        if !is_A_const
            dA = N == 1 ? A.dval : A.dval[i]
            LinearAlgebra.mul!(dX, dA, X, -one(T), one(T))
        end
        if !is_B_const
            dB = N == 1 ? B.dval : B.dval[i]
            LinearAlgebra.mul!(dX, dB, XK, -one(T), one(T))
        end
        if !is_C_const
            dC = N == 1 ? C.dval : C.dval[i]
            LinearAlgebra.mul!(reshape(s1, n * m, m), reshape(BX, n * m, m), C.val)
            for l in 1:m
                @views LinearAlgebra.mul!(s2[:, :, l], s1[:, :, l], dC)
            end
            copyto!(kron_buf, reshape(s2, n, m * m))
            dX .-= kron_buf
            LinearAlgebra.mul!(reshape(s1, n * m, m), reshape(BX, n * m, m), dC)
            for l in 1:m
                @views LinearAlgebra.mul!(s2[:, :, l], s1[:, :, l], C.val)
            end
            copyto!(kron_buf, reshape(s2, n, m * m))
            dX .-= kron_buf
        end
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

    # Adjoint Sylvester: same shape, transposed inputs. `gsylv_kamenik!`
    # overwrites its first arg, so seed it with the upstream X̄ from
    # D.dval; it returns Λ in that buffer.
    Λ = copy(D.dval)
    gsylv_kamenik!(
        Λ,
        Matrix(transpose(Aval)),
        Matrix(transpose(Bval)),
        Matrix(transpose(Cval))
    )

    # D̄ = Λ — overwrite D.dval (consumes the upstream X̄; semantically
    # the output-side cotangent is replaced by the input-side one for
    # the mutated buffer).
    copyto!(D.dval, Λ)

    if !(typeof(A) <: Const)
        LinearAlgebra.mul!(A.dval, Λ, transpose(X), -one(T), one(T))
    end

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

    if !(typeof(C) <: Const)
        BΛ = Matrix{T}(undef, n, m * m)
        LinearAlgebra.mul!(BΛ, transpose(Bval), Λ)
        R = Matrix{T}(undef, m * m, m * m)
        LinearAlgebra.mul!(R, transpose(X), BΛ)
        @inbounds for j in 1:m, i in 1:m
            s = zero(T)
            for l in 1:m, k in 1:m
                s += R[(i - 1) * m + k, (j - 1) * m + l] * Cval[k, l]
                s += R[(k - 1) * m + i, (l - 1) * m + j] * Cval[k, l]
            end
            C.dval[i, j] -= s
        end
    end

    return (nothing, nothing, nothing, nothing)
end
