"""
    gsylv_kamenik(A, B, C, D, ::Val{order} = Val(2)) -> X

Solve the compact Kronecker Sylvester equation

```math
A \\, X + B \\, X \\, (C \\otimes C \\otimes \\cdots \\otimes C) = D
```

for `X`. Only `order = 2` (one Kronecker square `C ⊗ C`) is implemented.

`A`, `B` ∈ ℝⁿˣⁿ; `C` ∈ ℝᵐˣᵐ; `D` ∈ ℝⁿˣᵐ²; the returned `X` has the same
shape as `D`. Inputs are not modified — the function copies `D` and calls
the in-place [`gsylv_kamenik!`](@ref). The Enzyme reverse rule attaches to
this allocating variant.

The equation arises in second-order perturbation solutions of nonlinear
DSGE models around a saddle-path steady state — see the `Compact Order-2
Kronecker Sylvester (Kamenik)` section of `DERIVATIONS.md` for context,
the Kamenik (2005) reference, and the AD derivation.

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

In-place form: overwrites `D` (RHS on entry) with the solution `X`. Only
`order = 2` is implemented. `A`, `B`, `C` are not modified.

Algorithm (order=2 specialization of Kamenik 2005):

After `B̃ = A⁻¹·B`, `D̃₀ = A⁻¹·D`, take real Schur factorizations
`B̃ = U_B · T · U_B'` and `C = U_C · S · U_C'` (both `T`, `S`
quasi-upper-triangular). With `X̃ = U_B' · X · (U_C ⊗ U_C)` and
`D̃ = U_B' · D̃₀ · (U_C ⊗ U_C)`, the equation collapses to

```
X̃ + T · X̃ · (S ⊗ S) = D̃.
```

Viewing `X̃, D̃` as `(n, m, m)` tensors with `X̃[:, p, o]` = column
`(o-1)·m + p`, each slice satisfies (column-major Kron convention)

```
X̃[:, p, o] + Σ_{o', p'} S[o', o] · S[p', p] · T · X̃[:, p', o'] = D̃[:, p, o].
```

`S` is quasi-upper-triangular, so `S[o', o] ≠ 0` only for `o' ≤ o` (plus
the subdiagonal `o' = o+1` at 2×2 complex Schur blocks). Iterate
`o = 1, 2, …, m` forward; at step `o`:

* **1×1 block** (`S[o+1, o] == 0` or `o == m`):

  ```
  X̃[:, :, o] + (S[o,o]·T) · X̃[:, :, o] · S = RHS_o,
  ```

  with `RHS_o = D̃[:, :, o] − Σ_{o' < o} S[o',o] · T · X̃[:, :, o'] · S`.
  Both `S[o,o]·T` and `S` are already in real Schur form from the outer
  factorizations, so `MatrixEquations.sylvds!` solves in place without
  re-Schur'ing.

* **2×2 complex block** at `(o, o+1)`: stack
  `Y = [X̃[:, :, o]; X̃[:, :, o+1]]` (`2n × m`) and solve

  ```
  Y + B_block · Y · S = RHS_stack
  ```

  with `B_block = [S[o,o]·T  S[o+1,o]·T; S[o,o+1]·T  S[o+1,o+1]·T]`
  (`2n × 2n`). `B_block` is not quasi-upper-triangular, so use
  `MatrixEquations.gsylv` (re-Schurs internally). 2×2 blocks are uncommon
  at the FVGQ / SGU scale; the extra Schur cost is rare in practice.

Schur-basis change and its inverse apply `(U_C ⊗ U_C)` / `(U_C ⊗ U_C)'`
to a `n × m²` matrix without materializing the `m² × m²` Kronecker — for
each of the two tensor axes the operator factors as a single matrix
multiply against `U_C` (column-major reshape).
"""
function gsylv_kamenik!(
        D::AbstractMatrix{Float64},
        A::AbstractMatrix{Float64},
        B::AbstractMatrix{Float64},
        C::AbstractMatrix{Float64},
        ::Val{order} = Val(2),
    ) where {order}
    @assert order == 2 "gsylv_kamenik! implements order = 2 only"
    n = size(A, 1)
    m = size(C, 1)
    @assert size(A) == (n, n)
    @assert size(B) == (n, n)
    @assert size(C) == (m, m)
    @assert size(D) == (n, m^2)

    F = lu(A)
    Bp = F \ B
    Dp = F \ D

    SB = schur(Bp); T = SB.T; UB = SB.Z
    SC = schur(C);  S = SC.T;  UC = SC.Z

    # D̃ = UB' · Dp · (UC ⊗ UC), computed as one big GEMM + m small GEMMs
    # against UC, without forming the m²×m² Kronecker.
    DpB = UB' * Dp                                              # n × m²
    T1 = Array{Float64}(undef, n, m, m)
    mul!(reshape(T1, n * m, m), reshape(DpB, n * m, m), UC)     # j-axis
    Dt = Array{Float64}(undef, n, m, m)
    for l in 1:m
        @views mul!(Dt[:, :, l], T1[:, :, l], UC)               # i-axis
    end

    Xt = zeros(Float64, n, m, m)

    # Per-iteration scratch, allocated once.
    slice_buf = Matrix{Float64}(undef, n, m)
    TX_buf    = Matrix{Float64}(undef, n, m)
    Tscaled   = Matrix{Float64}(undef, n, n)
    # 2×2 complex-block scratch — always alloc'd; tiny vs the rest.
    RHS_buf  = Matrix{Float64}(undef, 2n, m)
    Bblk_buf = Matrix{Float64}(undef, 2n, 2n)
    I_m      = Matrix{Float64}(I, m, m)
    I_2n     = Matrix{Float64}(I, 2n, 2n)

    o = 1
    while o <= m
        w = (o < m && S[o + 1, o] != 0.0) ? 2 : 1

        if w == 1
            copyto!(slice_buf, view(Dt, :, :, o))
            for op in 1:(o - 1)
                s_op = S[op, o]
                if s_op != 0.0
                    mul!(TX_buf, T, view(Xt, :, :, op))
                    mul!(slice_buf, TX_buf, S, -s_op, 1.0)
                end
            end
            Tscaled .= S[o, o] .* T
            MatrixEquations.sylvds!(Tscaled, S, slice_buf)
            @views Xt[:, :, o] .= slice_buf
        else
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
            @views Bblk_buf[1:n,         1:n]         .= S[o,     o]     .* T
            @views Bblk_buf[1:n,         (n + 1):(2n)] .= S[o + 1, o]     .* T
            @views Bblk_buf[(n + 1):(2n), 1:n]         .= S[o,     o + 1] .* T
            @views Bblk_buf[(n + 1):(2n), (n + 1):(2n)] .= S[o + 1, o + 1] .* T
            Y = MatrixEquations.gsylv(I_2n, I_m, Bblk_buf, S, RHS_buf)
            @views Xt[:, :, o]     .= Y[1:n,         :]
            @views Xt[:, :, o + 1] .= Y[(n + 1):(2n), :]
        end
        o += w
    end

    # X = UB · Xt · (UC ⊗ UC)'.  Same two-pass trick with UC'; reuse
    # T1 / Dt as scratch (no longer needed).
    mul!(reshape(T1, n * m, m), reshape(Xt, n * m, m), UC')      # j-axis: UC'
    for l in 1:m
        @views mul!(Dt[:, :, l], T1[:, :, l], UC')               # i-axis: UC'
    end
    mul!(D, UB, reshape(Dt, n, m^2))
    return D
end
