"""
    gsylv_kamenik(A, B, C, D, ::Val{order} = Val(2)) -> X

Solve the compact Kronecker Sylvester equation

```math
A \\, X + B \\, X \\, (C \\otimes C \\otimes \\cdots \\otimes C) = D
```

for `X`. Only `order = 2` (one Kronecker square `C ⊗ C`) is implemented.

`A`, `B` ∈ ℝⁿˣⁿ; `C` ∈ ℝᵐˣᵐ; `D` ∈ ℝⁿˣᵐ²; the returned `X` has the same
shape as `D`. Inputs are not modified — the function copies `D` and calls
the in-place [`gsylv_kamenik!`](@ref). The Enzyme forward and reverse
rules attach to this allocating variant.

The equation arises in second-order perturbation solutions of nonlinear
DSGE models around a saddle-path steady state — see the
[`docs/src/sylvester_kamenik.md`](@ref) page for the Kamenik (2005)
reference, the AD derivations, and worked examples.

See also [`gsylv_kamenik!`](@ref).
"""
function gsylv_kamenik(
        A::AbstractMatrix{Float64},
        B::AbstractMatrix{Float64},
        C::AbstractMatrix{Float64},
        D::AbstractMatrix{Float64},
        order::Val = Val(2),
    )
    X = copy(D)
    gsylv_kamenik!(X, A, B, C, order)
    return X
end

"""
    gsylv_kamenik!(D, A, B, C, ::Val{order} = Val(2))

In-place form: overwrites `D` (RHS on entry) with the solution `X`.
`A`, `B`, `C` are not modified. Only `order = 2` is implemented.

Dimensions: `A`, `B` ∈ ℝⁿˣⁿ; `C` ∈ ℝᵐˣᵐ; `D` ∈ ℝⁿˣᵐ². Internally
factors via a private `_gsylv_kamenik_factor` / `_gsylv_kamenik_solve!`
split so the Enzyme forward rule can amortise one Schur factorisation
across the primal and every tangent solve.

See the [docs page](@ref "Compact Order-2 Kronecker Sylvester (Kamenik)")
for the algorithm derivation, references, and AD rules.

See also [`gsylv_kamenik`](@ref).
"""
function gsylv_kamenik!(
        D::AbstractMatrix{Float64},
        A::AbstractMatrix{Float64},
        B::AbstractMatrix{Float64},
        C::AbstractMatrix{Float64},
        order::Val = Val(2),
    )
    cache = _gsylv_kamenik_factor(A, B, C, order)
    _gsylv_kamenik_solve!(cache, D)
    return D
end

# Private factor/solve API. The split exists so the Enzyme forward rule
# can amortise one Schur factorisation across the primal + every tangent
# RHS: `_gsylv_kamenik_factor` caches lu(A), the two Schur factors, and
# persistent scratch; `_gsylv_kamenik_solve!(cache, D)` reads only the
# cache + per-call RHS and overwrites D in place with X.

@concrete struct KamenikCache
    F                    # lu(A)
    T                    # real Schur of A⁻¹·B, quasi-upper-triangular
    UB                   # orthogonal basis of T
    S                    # real Schur of C, quasi-upper-triangular
    UC                   # orthogonal basis of S
    n::Int
    m::Int
    # Persistent scratch — sized at factor time so that solve! is
    # alloc-free apart from the F\D (n × m²) and UB' · (F\D) work GEMMs.
    T1                   # n × m × m
    Dt                   # n × m × m
    Xt                   # n × m × m
    slice_buf            # n × m
    TX_buf               # n × m
    Tscaled              # n × n
    RHS_buf              # 2n × m  (2×2 Schur block path)
    Bblk_buf             # 2n × 2n (2×2 Schur block path)
    I_m                  # m × m identity (gsylv arg)
    I_2n                 # 2n × 2n identity (gsylv arg)
end

function _gsylv_kamenik_factor(
        A::AbstractMatrix{Float64},
        B::AbstractMatrix{Float64},
        C::AbstractMatrix{Float64},
        ::Val{order} = Val(2),
    ) where {order}
    @assert order == 2 "gsylv_kamenik implements order = 2 only"
    n = size(A, 1)
    m = size(C, 1)
    @assert size(A) == (n, n)
    @assert size(B) == (n, n)
    @assert size(C) == (m, m)

    # Docs § Primal algorithm, stages 1-2 — the rhs-independent setup.
    #   1. Preprocess: B̃ = A⁻¹·B via one LU of A.
    #   2. Double real Schur: B̃ = U_B·T·U_B' and C = U_C·S·U_C', both
    #      quasi-upper-triangular. The whole forward iteration only ever
    #      reads (F, T, U_B, S, U_C); D enters per call via solve!.
    F = lu(A)
    Bp = F \ B
    SB = schur(Bp); T = SB.T; UB = SB.Z
    SC = schur(C);  S = SC.T;  UC = SC.Z

    T1 = Array{Float64}(undef, n, m, m)
    Dt = Array{Float64}(undef, n, m, m)
    Xt = Array{Float64}(undef, n, m, m)
    slice_buf = Matrix{Float64}(undef, n, m)
    TX_buf = Matrix{Float64}(undef, n, m)
    Tscaled = Matrix{Float64}(undef, n, n)
    RHS_buf = Matrix{Float64}(undef, 2n, m)
    Bblk_buf = Matrix{Float64}(undef, 2n, 2n)
    I_m = Matrix{Float64}(I, m, m)
    I_2n = Matrix{Float64}(I, 2n, 2n)

    return KamenikCache(
        F, T, UB, S, UC, n, m,
        T1, Dt, Xt, slice_buf, TX_buf, Tscaled,
        RHS_buf, Bblk_buf, I_m, I_2n,
    )
end

function _gsylv_kamenik_solve!(cache::KamenikCache, D::AbstractMatrix{Float64})
    (;
        F, T, UB, S, UC, n, m,
        T1, Dt, Xt, slice_buf, TX_buf, Tscaled,
        RHS_buf, Bblk_buf, I_m, I_2n,
    ) = cache
    @assert size(D) == (n, m^2)

    # Docs § Primal algorithm, stage 2: D̃ = U_B' · (A⁻¹·D) · (U_C ⊗ U_C).
    # The (U_C ⊗ U_C) leg is two GEMMs against U_C (no m²×m² Kron):
    # reshape (n, m²) → (n, m, m), contract along the j-axis, then per
    # l-slice along the i-axis.
    Dp = F \ D
    DpB = UB' * Dp
    mul!(reshape(T1, n * m, m), reshape(DpB, n * m, m), UC)     # j-axis
    for l in 1:m
        @views mul!(Dt[:, :, l], T1[:, :, l], UC)               # i-axis
    end

    # Docs § stage 3: in Schur coords X̃ + T·X̃·(S ⊗ S) = D̃; column-major
    # Kron view X̃[:, p, o] depends only on o' < o (S quasi-upper-tri).
    fill!(Xt, 0.0)

    o = 1
    while o <= m
        # w = 2 on a 2×2 complex Schur block straddling (o, o+1).
        w = (o < m && S[o + 1, o] != 0.0) ? 2 : 1

        if w == 1
            # 1×1 block:
            #   X̃[:, :, o] + (S[o,o]·T)·X̃[:, :, o]·S = RHS_o,
            #   RHS_o = D̃[:, :, o] − Σ_{o' < o} S[o',o]·T·X̃[:, :, o']·S.
            copyto!(slice_buf, view(Dt, :, :, o))
            for op in 1:(o - 1)
                s_op = S[op, o]
                if s_op != 0.0
                    mul!(TX_buf, T, view(Xt, :, :, op))
                    mul!(slice_buf, TX_buf, S, -s_op, 1.0)
                end
            end
            # Both factors already quasi-upper-triangular → no re-Schur.
            Tscaled .= S[o, o] .* T
            MatrixEquations.sylvds!(Tscaled, S, slice_buf)
            @views Xt[:, :, o] .= slice_buf
        else
            # 2×2 block at (o, o+1):
            #   Y = [X̃[:, :, o]; X̃[:, :, o+1]] (2n × m),
            #   Y + B_block · Y · S = RHS_stack,
            #   B_block = [S[o,o]·T S[o+1,o]·T; S[o,o+1]·T S[o+1,o+1]·T].
            for k in 0:1
                copyto!(slice_buf, view(Dt, :, :, o + k))
                for op in 1:(o - 1)
                    s_op = S[op, o + k]
                    if s_op != 0.0
                        mul!(TX_buf, T, view(Xt, :, :, op))
                        mul!(slice_buf, TX_buf, S, -s_op, 1.0)
                    end
                end
                @views RHS_buf[(k * n + 1):((k + 1) * n), :] .= slice_buf
            end
            @views Bblk_buf[1:n, 1:n] .= S[o, o] .* T
            @views Bblk_buf[1:n, (n + 1):(2n)] .= S[o + 1, o] .* T
            @views Bblk_buf[(n + 1):(2n), 1:n] .= S[o, o + 1] .* T
            @views Bblk_buf[(n + 1):(2n), (n + 1):(2n)] .= S[o + 1, o + 1] .* T
            # B_block is not quasi-upper-triangular → use generic gsylv.
            Y = MatrixEquations.gsylv(I_2n, I_m, Bblk_buf, S, RHS_buf)
            @views Xt[:, :, o] .= Y[1:n, :]
            @views Xt[:, :, o + 1] .= Y[(n + 1):(2n), :]
        end
        o += w
    end

    # Inverse basis change X = U_B · X̃ · (U_C ⊗ U_C)' — same two-pass
    # trick with UC'; T1 / Dt reused as scratch.
    mul!(reshape(T1, n * m, m), reshape(Xt, n * m, m), UC')
    for l in 1:m
        @views mul!(Dt[:, :, l], T1[:, :, l], UC')
    end
    mul!(D, UB, reshape(Dt, n, m^2))
    return D
end
