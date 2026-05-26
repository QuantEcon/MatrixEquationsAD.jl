# Enzyme reverse-mode rule for `gsylv_kamenik(A, B, C, D)`.
#
# Single-cotangent only (Width = 1). No forward rule, no batched widths,
# no orders other than 2 — see `DERIVATIONS.md` for the derivation and
# scope statement.
#
# Math (full derivation in `DERIVATIONS.md`):
#
#   Primal:   A·X + B·X·(C ⊗ C) = D,   K := C ⊗ C
#   Adjoint:  A'·Λ + B'·Λ·(C' ⊗ C') = X̄
#             (same equation shape, solved with another gsylv_kamenik call)
#
#   Pullbacks:
#     D̄ += Λ
#     Ā -= Λ · X'
#     B̄ -= Λ · (X · K)'
#     K̄  = −X' · B' · Λ      (m² × m²)
#     C̄[i,j] -= Σ_{k,l} R[(i-1)m+k, (j-1)m+l] · C[k,l]
#                + Σ_{k,l} R[(k-1)m+i, (l-1)m+j] · C[k,l]
#     with R := X' · B' · Λ = −K̄.

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

    # Adjoint Sylvester: A'·Λ + B'·Λ·(C' ⊗ C') = X̄. Same equation shape —
    # solved with another `gsylv_kamenik` call on the transposed inputs.
    # Materialize the transposes; `lu` / `schur` inside the primal want
    # plain `Matrix{T}`, not lazy `Transpose` wrappers.
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

    # B̄ -= Λ · (X · K)'   where K = C ⊗ C.  Compute X·K via the same
    # two-pass GEMM trick the primal uses — no m²×m² Kronecker.
    if !(typeof(B) <: Const)
        XK = Matrix{T}(undef, n, m^2)
        T1 = Array{T}(undef, n, m, m)
        LinearAlgebra.mul!(reshape(T1, n * m, m), reshape(X, n * m, m), Cval)
        XK_t = reshape(XK, n, m, m)
        for l in 1:m
            @views LinearAlgebra.mul!(XK_t[:, :, l], T1[:, :, l], Cval)
        end
        LinearAlgebra.mul!(B.dval, Λ, transpose(XK), -one(T), one(T))
    end

    # C̄ via K̄ = −R where R = X' · B' · Λ  (m² × m²), then two index sums
    # (one per Kronecker leg).
    if !(typeof(C) <: Const)
        BΛ = Matrix{T}(undef, n, m^2)
        LinearAlgebra.mul!(BΛ, transpose(Bval), Λ)
        R = Matrix{T}(undef, m^2, m^2)
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

# ─── In-place form ───────────────────────────────────────────────────────────
#
# `gsylv_kamenik!(D, A, B, C)` overwrites `D` with the solution `X`. Enzyme
# sees `D` as a single Annotation that's both RHS-in and solution-out. The
# `Duplicated(D, D̄)` slot accordingly carries `X̄` on entry to the reverse
# pass (cotangent of the post-call value, i.e. the solution); the rule
# writes `Λ` back into the same slot (cotangent of the pre-call value, i.e.
# the original RHS — `D̄ = Λ`).
#
# Same math as the allocating rule above; the only difference is single-
# buffer aliasing.

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
    # Primal overwrote D.val with the solution X. Snapshot it for the
    # reverse pass; A/B/C are not modified by the primal so refs suffice.
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

    # Adjoint Sylvester: A'·Λ + B'·Λ·(C' ⊗ C') = X̄. Same shape, transposed
    # inputs. `gsylv_kamenik!` overwrites its first arg, so seed it with
    # the upstream X̄ from D.dval; it returns Λ in that buffer.
    Λ = copy(D.dval)
    gsylv_kamenik!(Λ,
        Matrix(transpose(Aval)),
        Matrix(transpose(Bval)),
        Matrix(transpose(Cval)))

    # D̄ = Λ — overwrite D.dval (consumes the upstream X̄; semantically the
    # output-side cotangent is replaced by the input-side one for the
    # mutated buffer).
    copyto!(D.dval, Λ)

    # Ā -= Λ · X'
    if !(typeof(A) <: Const)
        LinearAlgebra.mul!(A.dval, Λ, transpose(X), -one(T), one(T))
    end

    # B̄ -= Λ · (X · K)'   where K = C ⊗ C.
    if !(typeof(B) <: Const)
        XK = Matrix{T}(undef, n, m^2)
        T1 = Array{T}(undef, n, m, m)
        LinearAlgebra.mul!(reshape(T1, n * m, m), reshape(X, n * m, m), Cval)
        XK_t = reshape(XK, n, m, m)
        for l in 1:m
            @views LinearAlgebra.mul!(XK_t[:, :, l], T1[:, :, l], Cval)
        end
        LinearAlgebra.mul!(B.dval, Λ, transpose(XK), -one(T), one(T))
    end

    # C̄ via K̄ = −R where R = X' · B' · Λ (m² × m²), then two index sums.
    if !(typeof(C) <: Const)
        BΛ = Matrix{T}(undef, n, m^2)
        LinearAlgebra.mul!(BΛ, transpose(Bval), Λ)
        R = Matrix{T}(undef, m^2, m^2)
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
