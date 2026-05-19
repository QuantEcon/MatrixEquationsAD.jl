# AD Derivations

This file records the mathematics implemented by the custom AD rules in this
package. All formulas are for real matrices, use the Frobenius inner product

```math
\langle U, V \rangle = \operatorname{tr}(U^\top V),
```

and write reverse-mode cotangents as barred variables such as `\bar X`.
Selection decisions, integer outputs, and solver branch choices are treated as
locally constant.

All linear solves below assume the corresponding linearized operator is
nonsingular. For discrete Lyapunov equations, a sufficient condition is
`\rho(A) < 1`, while the more general uniqueness condition is that no pair of
eigenvalues of `A` has product equal to one.

## Supported Rules

| Function | AD support | Rule files | Differentiated outputs |
| --- | --- | --- | --- |
| `MatrixEquations.lyapd` | ForwardDiff, Enzyme forward, Enzyme reverse | [`ext/forwarddiff_lyapunov.jl`](ext/forwarddiff_lyapunov.jl), [`ext/enzyme_lyapunov.jl`](ext/enzyme_lyapunov.jl) | `X` |
| `MatrixEquationsAD.lyapdkr` | ForwardDiff, Enzyme forward, Enzyme reverse | [`ext/forwarddiff_lyapdkr.jl`](ext/forwarddiff_lyapdkr.jl), [`ext/enzyme_lyapdkr.jl`](ext/enzyme_lyapdkr.jl) | `X` |
| `MatrixEquations.gsylv` | ForwardDiff, Enzyme forward, Enzyme reverse | [`ext/forwarddiff_sylvester.jl`](ext/forwarddiff_sylvester.jl), [`ext/enzyme_sylvester.jl`](ext/enzyme_sylvester.jl) | `X` |
| `MatrixEquations.gsylvkr` | ForwardDiff, Enzyme forward, Enzyme reverse | [`ext/forwarddiff_sylvester.jl`](ext/forwarddiff_sylvester.jl), [`ext/enzyme_sylvester.jl`](ext/enzyme_sylvester.jl) | `X` |
| `MatrixEquations.ared` | ForwardDiff, Enzyme forward, Enzyme reverse | [`ext/forwarddiff_riccati.jl`](ext/forwarddiff_riccati.jl), [`ext/enzyme_riccati.jl`](ext/enzyme_riccati.jl), [`ext/riccati_derivatives.jl`](ext/riccati_derivatives.jl) | `X`, `F` |
| `MatrixEquationsAD.ordqz!` / `_ordqz!` | ForwardDiff, Enzyme forward, Enzyme reverse | [`ext/forwarddiff_ordqz.jl`](ext/forwarddiff_ordqz.jl), [`ext/enzyme_ordqz.jl`](ext/enzyme_ordqz.jl), [`ext/ordqz_derivatives.jl`](ext/ordqz_derivatives.jl) | `S`, `T`, `Q`, `Z` |

The public out-of-place `ordqz(A, B, ordering; threshold)` is a primal
convenience wrapper. The custom AD rules are for the mutating path that writes
`S`, `T`, `Q`, and `Z`.

## Discrete Lyapunov

`MatrixEquations.lyapd(A, C)` solves the discrete Lyapunov equation documented
by MatrixEquations.jl:

```math
A X A^\top - X + C = 0.
```

Equivalently, define the linear operator

```math
L_A[X] = X - A X A^\top,
```

so that the primal equation is

```math
L_A[X] = C.
```

The tangent equation is

```math
L_A[\Delta X]
    =
    \Delta C
    + \Delta A X A^\top
    + A X \Delta A^\top .
```

For reverse mode, define `Y` by the adjoint Lyapunov solve

```math
L_A^*[Y] = Y - A^\top Y A = \bar X.
```

Then

```math
\bar C \mathrel{+}= Y,
```

and

```math
\bar A
    \mathrel{+}=
    Y A X^\top + Y^\top A X .
```

The implementation uses a Schur cache of `A` and reuses it across tangent
directions or reverse seeds. When the right-hand side is symmetric, it uses the
symmetric `lyapds!` path. In reverse mode the implementation also symmetrizes a
symmetric cotangent seed on the symmetric path.

References:

- MatrixEquations.jl documents `lyapd` as solving `AXA' - X + C = 0` in its
  [Lyapunov solver documentation](https://andreasvarga.github.io/MatrixEquations.jl/latest/lyapunov.html).
- Kao and Hennequin derive forward and reverse rules for Lyapunov equations in
  [Automatic differentiation of Sylvester, Lyapunov, and algebraic Riccati equations](https://arxiv.org/abs/2011.11430).

## Symmetric Kronecker Lyapunov

`MatrixEquationsAD.lyapdkr(A, C)` forms the dense Kronecker system for the same
discrete Lyapunov equation and returns the symmetric projection of the result.
For nonsymmetric `C`, this is the symmetrized-output map, not the raw
full-matrix Kronecker solve.
With column-major `vec`,

```math
\operatorname{vec}(A X A^\top) = (A \otimes A)\operatorname{vec}(X),
```

so

```math
\left(I - A \otimes A\right)\operatorname{vec}(X)
    =
    \operatorname{vec}(C).
```

Let `P(M) = (M + M^\top)/2` be the symmetric projection. The implemented map is

```math
X = P\left(L_A^{-1}[C]\right).
```

The tangent equation first solves the full Kronecker linearization,

```math
L_A[\Delta X_{\mathrm{raw}}]
    =
    \Delta C
    + \Delta A X A^\top
    + A X \Delta A^\top,
```

then returns

```math
\Delta X = P(\Delta X_{\mathrm{raw}}).
```

For reverse mode, the projection is applied to the cotangent before the
transpose Kronecker solve:

```math
Y = L_A^{-*}[P(\bar X)].
```

The parameter adjoints are then the same contractions as for `lyapd`:

```math
\bar C \mathrel{+}= Y,
\qquad
\bar A \mathrel{+}= Y A X^\top + Y^\top A X.
```

References:

- The vectorization identity above is a standard Kronecker-`vec` identity; see
  Petersen and Pedersen, [The Matrix Cookbook](https://www2.imm.dtu.dk/pubdb/pubs/3274-full.html).
- MatrixEquations.jl documents the same discrete Lyapunov equation for
  [`lyapd`](https://andreasvarga.github.io/MatrixEquations.jl/latest/lyapunov.html).

## Generalized Sylvester

`MatrixEquations.gsylv(A, B, C, D, E)` solves

```math
A X B + C X D = E.
```

Define

```math
G[X] = A X B + C X D.
```

The tangent equation is

```math
G[\Delta X]
    =
    \Delta E
    - \Delta A X B
    - A X \Delta B
    - \Delta C X D
    - C X \Delta D .
```

For reverse mode, solve the adjoint generalized Sylvester equation

```math
G^*[Y]
    =
    A^\top Y B^\top + C^\top Y D^\top
    =
    \bar X.
```

Then

```math
\bar E \mathrel{+}= Y,
```

and

```math
\begin{aligned}
\bar A &\mathrel{-}= Y B^\top X^\top, \\
\bar B &\mathrel{-}= X^\top A^\top Y, \\
\bar C &\mathrel{-}= Y D^\top X^\top, \\
\bar D &\mathrel{-}= X^\top C^\top Y.
\end{aligned}
```

The `gsylv` rule reuses a generalized-Schur cache for the primal solve and all
tangent or reverse solves. These formulas assume the generalized Sylvester
operator is nonsingular. Equivalently, MatrixEquations.jl requires the pencils
`A - lambda C` and `D + lambda B` to be regular and to have no common
eigenvalues.

References:

- MatrixEquations.jl documents `gsylv` as solving `AXB + CXD = E` in its
  [Sylvester solver documentation](https://andreasvarga.github.io/MatrixEquations.jl/v2.4/sylvester.html).
- LAPACK's `DTGSYL` documentation gives the generalized Sylvester systems and
  transposed systems used in Schur-based solvers:
  [DTGSYL](https://www.netlib.org/lapack/explore-html/d4/d3b/group__tgsyl_ga96eff9d077e7600c68cd18246ca4cdc3.html).
- Kao and Hennequin give AD rules for Sylvester-type equations in
  [arXiv:2011.11430](https://arxiv.org/abs/2011.11430).

## Kronecker Generalized Sylvester

`MatrixEquations.gsylvkr(A, B, C, D, E)` solves the same equation as `gsylv`,
but by factoring the Kronecker representation

```math
\left(B^\top \otimes A + D^\top \otimes C\right)
\operatorname{vec}(X)
    =
    \operatorname{vec}(E).
```

The tangent and reverse equations are identical to the `gsylv` equations above.
Only the cached linear operator differs: the forward rule uses the LU
factorization of the Kronecker matrix, and the reverse rule uses its transpose
solve. The same nonsingularity condition applies, now as nonsingularity of
`B^\top \otimes A + D^\top \otimes C`.

References:

- MatrixEquations.jl documents `gsylvkr` in its
  [Kronecker solver documentation](https://andreasvarga.github.io/MatrixEquations.jl/v2.4/sylvkr.html).
- The Kronecker representation follows the same `vec` identity cited in
  [The Matrix Cookbook](https://www2.imm.dtu.dk/pubdb/pubs/3274-full.html).

## Discrete Algebraic Riccati

`MatrixEquations.ared(A, B, R, Q, S)` solves the discrete algebraic Riccati
equation

```math
A^\top X A - X
    - (A^\top X B + S)
      (R + B^\top X B)^{-1}
      (B^\top X A + S^\top)
    + Q
    =
    0.
```

Let

```math
G = R + B^\top X B,
\qquad
F = G^{-1}(B^\top X A + S^\top),
\qquad
A_c = A - B F.
```

The rule assumes the selected stabilizing or anti-stabilizing Riccati branch is
locally smooth, `G` is nonsingular, and the closed-loop Lyapunov operator is
nonsingular. The usual stabilizing DARE sufficient conditions are stabilizability
and detectability, together with a well-conditioned positive definite `G`.

The tangent for `X` solves the Lyapunov equation

```math
\Delta X - A_c^\top \Delta X A_c = P_n(H),
```

where

```math
\begin{aligned}
H ={}&
    \Delta Q
    + \Delta A^\top X A_c
    + A_c^\top X \Delta A \\
&   - A_c^\top X \Delta B F
    - F^\top \Delta B^\top X A_c
    + F^\top \Delta R F \\
&   - \Delta S F
    - F^\top \Delta S^\top .
\end{aligned}
```

Here `P_n` symmetrizes an `n` by `n` matrix. For symmetric perturbations of
`Q` and `R`, and arbitrary cross-term perturbations of `S`, this is just the
symmetric right-hand side; for unconstrained nonsymmetric directions in `Q` or
`R`, it documents the projection used by the implementation.

After solving for `\Delta X`, the gain tangent is

```math
\Delta F = G^{-1}(\Delta M - \Delta G F),
```

with

```math
\Delta M
    =
    \Delta B^\top X A
    + B^\top \Delta X A
    + B^\top X \Delta A
    + \Delta S^\top
```

and

```math
\Delta G
    =
    \Delta R
    + \Delta B^\top X B
    + B^\top \Delta X B
    + B^\top X \Delta B .
```

For reverse mode, cotangents for both differentiated outputs are accepted:
`\bar X` and `\bar F`. First propagate through `F`:

```math
\Lambda = G^{-\top}\bar F,
\qquad
\Theta = P_m(-\Lambda F^\top),
```

where `P_m` symmetrizes an `m` by `m` matrix. The direct contributions from
`\bar F` are

```math
\begin{aligned}
\bar A &\mathrel{+}= X^\top B \Lambda, \\
\bar B &\mathrel{+}= X A \Lambda^\top
    + X B \Theta^\top
    + X^\top B \Theta, \\
\bar R &\mathrel{+}= \Theta, \\
\bar S &\mathrel{+}= \Lambda^\top .
\end{aligned}
```

The `P_m` projection means `R` is treated as a symmetric parameter. An
unconstrained nonsymmetric `R` map would use the unsymmetrized contribution
`-\Lambda F^\top` instead.

The cotangent passed to the Lyapunov adjoint solve is

```math
\bar X_{\mathrm{total}}
    =
    P_n\left(\bar X + B\Lambda A^\top + B\Theta B^\top\right).
```

The implementation forms the sum first and then symmetrizes it, which is
equivalent by linearity of `P_n`.

Let `Y` solve the adjoint of the tangent Lyapunov operator,

```math
Y - A_c Y A_c^\top = \bar X_{\mathrm{total}}.
```

Then add the remaining parameter adjoints:

```math
\begin{aligned}
\bar Q &\mathrel{+}= Y, \\
\bar A &\mathrel{+}= X A_c Y^\top + X^\top A_c Y, \\
\bar B &\mathrel{-}= X^\top A_c Y F^\top + X A_c Y^\top F^\top, \\
\bar R &\mathrel{+}= F Y F^\top, \\
\bar S &\mathrel{-}= Y F^\top + Y^\top F^\top .
\end{aligned}
```

Only `X` and `F` are differentiated. The `evals`, `Z`, and `scalinfo` outputs
are returned with zero shadows. The default `S` method is handled by setting
`S = zeros(size(B))` before dispatching to the five-argument rule.

References:

- MatrixEquations.jl documents `ared`, the stabilizing gain `F`, and the
  discrete Riccati equation in its
  [Riccati solver documentation](https://andreasvarga.github.io/MatrixEquations.jl/dev/riccati.html).
- Arnold and Laub's generalized eigenproblem method is the Riccati reference
  cited by MatrixEquations.jl:
  [DOI:10.1109/PROC.1984.13083](https://doi.org/10.1109/PROC.1984.13083).
- Kao and Hennequin derive AD rules for algebraic Riccati equations in
  [arXiv:2011.11430](https://arxiv.org/abs/2011.11430).

## Ordered QZ

`MatrixEquationsAD.ordqz!` computes an ordered real generalized Schur
factorization

```math
A = Q S Z^\top,
\qquad
B = Q T Z^\top,
```

where `Q` and `Z` are orthogonal, `S` is quasi-upper triangular, and `T` is
upper triangular. The currently supported ordering symbol is `:bk`, which
moves generalized eigenvalues satisfying

```math
|\alpha_i|^2 \ge (1 - \tau)^2 |\beta_i|^2,
```

where `\tau` is the `threshold` keyword. The returned `sdim` is the number of
selected eigenvalues in the leading block and is non-differentiable. The
threshold is treated as a constant margin; the derivative assumes no eigenvalue
crosses the selection boundary.

The derivative is local to a fixed ordering and fixed real-Schur block
structure. The implementation detects two-by-two real-Schur blocks using the
fixed cutoff `abs(S[i + 1, i]) > 1e-14`. Define

```math
E = Q^\top \Delta A Z,
\qquad
H = Q^\top \Delta B Z,
```

and skew-symmetric matrices

```math
\Omega_Q = Q^\top \Delta Q,
\qquad
\Omega_Z = Z^\top \Delta Z.
```

The skew-symmetry follows from differentiating `Q^\top Q = I` and
`Z^\top Z = I`.
The implementation fixes a gauge by leaving the block-diagonal entries of
`\Omega_Q` and `\Omega_Z` at zero and solving only off-diagonal block entries.

Differentiating the factorizations gives

```math
\Delta S = E - \Omega_Q S + S\Omega_Z,
\qquad
\Delta T = H - \Omega_Q T + T\Omega_Z.
```

The lower off-block entries of `\Delta S` and the strictly lower entries of
`\Delta T` must be zero to preserve generalized Schur form. The implementation
solves the resulting small block generalized Sylvester systems in a
block-triangular sweep for the off-diagonal entries of `\Omega_Q` and
`\Omega_Z`, then returns

```math
\Delta Q = Q\Omega_Q,
\qquad
\Delta Z = Z\Omega_Z.
```

The reverse rule back-propagates through the same triangular dependence in the
opposite sweep order. After propagating cotangents through `S`, `T`, `Q`, and
`Z`, it ends with

```math
\bar A \mathrel{+}= Q \bar E Z^\top,
\qquad
\bar B \mathrel{+}= Q \bar H Z^\top.
```

The rule is not valid across eigenvalue selection changes, changes in
one-by-one versus two-by-two block structure, repeated or nearly coincident
generalized eigenvalues, or threshold crossings. In Enzyme, `ordering` and
`threshold` are passed as `Const`.

References:

- LAPACK's `DGGES` documentation defines the generalized real Schur
  factorization, optional eigenvalue ordering, and `sdim`-style selected count:
  [DGGES](https://www.netlib.org/lapack/explore-html/d7/d25/group__gges_ga556be4f39b39e5008c8eb36814aa7e20.html).
- Julia's `LinearAlgebra` documentation describes `GeneralizedSchur`,
  `schur(A, B)`, and `ordschur!`:
  [LinearAlgebra docs](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/).
- Sun gives perturbation bounds for the generalized Schur decomposition:
  [DOI:10.1137/S0895479892242189](https://doi.org/10.1137/S0895479892242189).
- Blanchard and Kahn's rational-expectations conditions motivate the `:bk`
  ordering used for unstable generalized eigenvalues:
  [DOI:10.2307/1912186](https://doi.org/10.2307/1912186).
- Sims discusses QZ/generalized-Schur methods for linear rational-expectations
  models:
  [DOI:10.1023/A:1020517101123](https://doi.org/10.1023/A:1020517101123).
