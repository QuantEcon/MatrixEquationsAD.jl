# Kronecker Discrete Lyapunov

`lyapdkr(A, C)` solves the discrete Lyapunov equation

```math
A X A^\top - X + C = 0
```

via the Kronecker-vectorised form. Where `MatrixEquations.lyapd` uses a
Schur-based Bartels–Stewart solver, `lyapdkr` LU-factorises the full
``(n^2) \times (n^2)`` Kronecker operator once and reuses that
factorisation across all tangent / cotangent directions. The output is
symmetrised, so nonsymmetric perturbations of ``C`` are projected onto
the symmetric solution manifold.

The same equation governs the stationary covariance of the QuantEcon
[Linear State Space Model](https://julia.quantecon.org/introduction_dynamics/linear_models.html)
``x_{t+1} = A\,x_t + w_{t+1}`` with ``w_t \sim \mathcal{N}(0, Q)``;
this page uses the Kronecker LU instead of the Schur sweep on the
[Discrete Lyapunov (Schur)](lyapd.md) page.

Implementation pointers:

- `src/lyapdkr.jl` — `lyapdkrfactor`, `lyapdkrsolve`,
  `lyapdkradjointsolve`, and the cache struct.
- `ext/enzyme_lyapdkr.jl` and `ext/forwarddiff_lyapdkr.jl` — AD rules.

## Primal

With column-major `vec` and the identity
``\operatorname{vec}(A X A^\top) = (A \otimes A)\operatorname{vec}(X)``,
the equation becomes

```math
\bigl(I_{n^2} - A \otimes A\bigr)\,\operatorname{vec}(X)
\;=\;
\operatorname{vec}(C).
```

`lyapdkrfactor` builds ``M = I - A \otimes A`` and stores its LU in a
`LyapDKrLUCache`. With ``P(N) = \tfrac{1}{2}(N + N^\top)`` the
symmetric projection, `lyapdkrsolve` performs one back-substitution
and projects:

```math
X \;=\; P\!\left(\operatorname{reshape}(M^{-1}\operatorname{vec}(C),\; n,\; n)\right).
```

Assumptions:

- ``M`` nonsingular — equivalently, no pair of eigenvalues of ``A``
  has product equal to one. ``\rho(A) < 1`` is sufficient.

A singular ``M`` raises `SingularException` from `lu!`. Callers that
need PSD / magnitude checks on the returned ``X`` own them.

## Worked example

The 2×2 sanity case from [Discrete Lyapunov (Schur)](lyapd.md) on the
Kronecker path:

```jldoctest lyapdkr_small
julia> using MatrixEquationsAD

julia> A = [0.55 0.08; -0.04 0.42]
2×2 Matrix{Float64}:
  0.55  0.08
 -0.04  0.42

julia> C = [1.0 0.2; 0.2 0.7]
2×2 Matrix{Float64}:
 1.0  0.2
 0.2  0.7

julia> X = lyapdkr(A, C)
2×2 Matrix{Float64}:
 1.47343   0.253679
 0.253679  0.84244

julia> isapprox(A * X * A' - X + C, zeros(2, 2); atol = 1.0e-12)
true
```

`lyapdkr` and `lyapd` return the same solution up to round-off, but
take different paths: `lyapd` runs an ``O(n^3)`` Schur sweep, `lyapdkr`
LU-factorises an ``n^2 \times n^2`` Kronecker matrix.

## ForwardDiff JVP

**Step 1: differentiate the implicit equation.** For one tangent
direction ``(d A, d C)``, differentiating
``M\,\operatorname{vec}(X) = \operatorname{vec}(C)`` gives

```math
\operatorname{vec}(d X_{\mathrm{raw}})
\;=\;
M^{-1}\,\operatorname{vec}\!\bigl(
d C \;+\; d A\,X\,A^\top \;+\; A\,X\,d A^\top
\bigr),
\qquad
d X \;=\; P(d X_{\mathrm{raw}}).
```

**Step 2: cached factorisation.** The LU of
``M = I_{n^2} - A \otimes A`` is built once on the value layer and
reused for every tangent.

**Step 3: solve per direction.** One LU back-substitution against the
shared factorisation, followed by symmetric projection.

**Step 4: code path.** The ForwardDiff overload runs `lyapdkrfactor`
once on the value layer and calls `lyapdkrsolve(cache, RHS_i)` once
per partial direction. A chunk of width ``N`` issues ``N``
back-substitutions; the Enzyme `BatchDuplicated` forward rule is
structurally identical.

## Enzyme VJP

**Step 1: differentiate the implicit equation (adjoint).** Let
``\bar X`` be the cotangent on the output. Symmetrise
``S = P(\bar X) = \tfrac{1}{2}(\bar X + \bar X^\top)`` and solve the
transposed system

```math
\operatorname{vec}(Y) \;=\; M^{-\top}\,\operatorname{vec}(S),
\qquad
M = I - A \otimes A.
```

**Step 2: cached factorisation.** Same LU of ``M`` as the JVP, copied
to Enzyme's tape so multiple reverse cotangents reuse it.

**Step 3: parameter cotangents.** Since ``M`` is bilinear in ``A``,

```math
\bar C \;\mathrel{+}=\; Y,
\qquad
\bar A
\;\mathrel{+}=\;
Y\,A\,X^\top \;+\; Y^\top\,A\,X.
```

These contractions are identical to the `lyapd` case.

**Step 4: code path.** `lyapdkradjointsolve` performs one transposed
back-substitution and the two outer-product accumulations.

## References

- Petersen, K. B. and Pedersen, M. S. *The Matrix Cookbook.*
  [PDF](https://www2.imm.dtu.dk/pubdb/pubs/3274-full.html). The
  Kronecker-`vec` identity
  ``\operatorname{vec}(A X A^\top) = (A \otimes A)\operatorname{vec}(X)``
  used in the primal.
- Kao, T.-T. and Hennequin, M. (2020). *Automatic differentiation of
  Sylvester, Lyapunov, and algebraic Riccati equations.*
  [arXiv:2011.11430](https://arxiv.org/abs/2011.11430). General recipe;
  the cotangent contraction ``\bar A = Y A X^\top + Y^\top A X`` is the
  discrete-Lyapunov instance.
- MatrixEquations.jl documents the same discrete Lyapunov equation for
  [`lyapd`](https://andreasvarga.github.io/MatrixEquations.jl/latest/lyapunov.html).
- QuantEcon Julia,
  [Linear State Space Models](https://julia.quantecon.org/introduction_dynamics/linear_models.html)
  — stationary covariance ``\Sigma_\infty = A\,\Sigma_\infty\,A^\top + Q``.

## Static (SMatrix) dispatch

The StaticArrays extension provides

```julia
lyapdkr(A::SMatrix{n,n,T}, C::SMatrix{n,n,T})
```

which converts to heap, calls the regular `lyapdkr`, and wraps the
result in an `SMatrix{n,n,T}`. The Kronecker matrix is
``n^2 \times n^2``, so even at modest ``n`` it is past the size where
an inline static LU beats LAPACK; the dispatch exists for API
consistency, not speed.

## Example: RBC stationary distribution

Using the RBC policy values from the
[Klein Policy Map quick start](klein_map.md#quick-start:-parameters-to-policy):

```jldoctest lyapdkr_rbc
julia> using MatrixEquationsAD

julia> h_x = [
           0.9568351489231556  6.209371005755667;
           -3.3737787177631822e-18  0.20000000000000004
       ]
2×2 Matrix{Float64}:
  0.956835     6.20937
 -3.37378e-18  0.2

julia> B_shock = reshape([0.0, -0.01], 2, 1)
2×1 Matrix{Float64}:
  0.0
 -0.01

julia> Q = B_shock * transpose(B_shock)
2×2 Matrix{Float64}:
 0.0  0.0
 0.0  0.0001

julia> V = lyapdkr(h_x, Q)
2×2 Matrix{Float64}:
 0.0700541    0.000159976
 0.000159976  0.000104167

julia> V[2, 2] ≈ 0.01^2 / (1 - 0.2^2)   # matches AR(1) closed form
true
```

TFP fluctuates at roughly 1% in stationary equilibrium; capital, which
absorbs cumulative TFP shocks through ``h_x[1,2]``, fluctuates ~25×
more.

## Differentiating through `lyapdkr`

Enzyme reverse against a scalar loss:

```julia
using LinearAlgebra: dot
using Enzyme: Active, Const, Duplicated, Reverse, autodiff, make_zero
using MatrixEquationsAD

A = [0.55  0.08; -0.04  0.42]
C = [1.0   0.2;   0.2   0.7]
W = [0.3  -0.1;   0.2   0.5]

loss(A, C, W) = dot(W, lyapdkr(A, C))

A_bar = make_zero(A); C_bar = make_zero(C)
autodiff(
    Reverse, loss, Active,
    Duplicated(copy(A), A_bar),
    Duplicated(copy(C), C_bar),
    Const(W),
)
# A_bar = Y A X' + Y' A X and C_bar = Y, where Y solves the transposed
# Kronecker system M^T vec(Y) = vec(P(W)).
```
