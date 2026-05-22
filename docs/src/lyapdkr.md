# Kronecker Discrete Lyapunov

`lyapdkr(A, C)` solves the discrete Lyapunov equation

```math
A X A^{\!\top} - X + C = 0
```

via the Kronecker-vectorised form. Unlike `MatrixEquations.lyapd`, which
uses a Schur-based Bartels-Stewart solver, `lyapdkr` LU-factorises the full
``(n^2) \times (n^2)`` Kronecker operator once and reuses that factorisation
across all tangent / cotangent directions. The output is symmetrised, so
nonsymmetric perturbations of ``C`` are projected onto the symmetric
solution manifold.

Implementation pointers:

- `src/lyapdkr.jl` — `lyapdkrfactor`, `lyapdkrsolve`, `lyapdkradjointsolve`,
  and the cache struct.
- `ext/enzyme_lyapdkr.jl` and `ext/forwarddiff_lyapdkr.jl` — AD rules.

## Primal

With column-major vec and the identity
``\operatorname{vec}(A X A^{\!\top}) = (A \otimes A)\,\operatorname{vec}(X)``,
the discrete Lyapunov equation rewrites as

```math
\bigl(I_{n^2} - A \otimes A\bigr)\,\operatorname{vec}(X)
\;=\;
\operatorname{vec}(C).
```

`lyapdkrfactor` builds ``M = I - A \otimes A`` and stores its LU
factorisation in a `LyapDKrLUCache`. `lyapdkrsolve` performs one
back-substitution against ``M`` and symmetrises the result:

```math
X \;=\; P\!\left(\operatorname{reshape}(M^{-1}\operatorname{vec}(C),\; n,\; n)\right),
\qquad
P(M) = \tfrac{1}{2}(M + M^{\!\top}).
```

The factorisation assumptions are:

- ``M`` is nonsingular — equivalently, no pair of eigenvalues of ``A`` has
  product equal to one. ``\rho(A) < 1`` is sufficient.
- Optional diagnostics: `tol_diag` bounds ``|X_{ii}|`` and `check_psd`
  rejects negative diagonals.

Failures throw `ErrorException`.

## ForwardDiff JVP

For one tangent direction ``(dA, dC)``, differentiating
``M\,\operatorname{vec}(X) = \operatorname{vec}(C)`` gives

```math
\operatorname{vec}(dX_{\mathrm{raw}})
\;=\;
M^{-1}\,\operatorname{vec}\!\bigl(
dC \;+\; dA\,X\,A^{\!\top} \;+\; A\,X\,dA^{\!\top}
\bigr),
\qquad
dX \;=\; P(dX_{\mathrm{raw}}).
```

The ForwardDiff overload runs `lyapdkrfactor` once on the value layer and
calls `lyapdkrsolve(cache, RHS_i)` once per partial direction inside the
chunk. A chunk of width ``N`` performs ``N`` LU back-substitutions against
the shared factorisation.

The Enzyme `BatchDuplicated` forward rule is structurally identical: one
factorise, ``N`` solves.

## Enzyme VJP

Let ``\bar X`` be the cotangent on the output. Symmetrise first to project
onto the symmetric manifold,

```math
S \;=\; P(\bar X) \;=\; \tfrac{1}{2}(\bar X + \bar X^{\!\top}),
```

then perform one transposed solve

```math
\operatorname{vec}(Y) \;=\; M^{-\!\top}\,\operatorname{vec}(S),
\qquad
M = I - A \otimes A.
```

Because ``M`` is bilinear in ``A``, the parameter cotangents are

```math
\bar C \;\mathrel{+}=\; Y,
\qquad
\bar A \;\mathrel{+}=\; Y\,A\,X^{\!\top} \;+\; Y^{\!\top} A\,X.
```

`lyapdkradjointsolve` performs the symmetric projection and the transposed
LU solve. The Enzyme augmented primal copies the cached factorisation and
the primal ``X`` to the tape; the reverse pass performs one transposed
back-substitution and the two outer-product accumulations.

## Static (SMatrix) Dispatch

The StaticArrays extension provides

```julia
lyapdkr(A::SMatrix{n,n,T}, C::SMatrix{n,n,T}; tol_diag=Inf, check_psd=false)
```

which converts to heap, calls the regular `lyapdkr`, and wraps the result
in an `SMatrix{n,n,T}`. The Kronecker matrix is ``n^2 \times n^2``, so
even at modest ``n`` it is past the size where an inline static LU beats
LAPACK; the dispatch exists for API consistency, not speed.

## Example: RBC Stationary Distribution

After solving for the Klein policy, the predetermined-state law of motion
is

```math
x_{t+1} = h_x\, x_t + B_x\,\varepsilon_{t+1},
```

with `x = [k, z]` for the RBC model and shock loading ``B_x = [0; \sigma]``
— the TFP innovation feeds into the AR(1) for ``z`` and nothing else.
``Q = B_x B_x^{\!\top}`` is the one-step innovation covariance and the
stationary covariance ``V = h_x V h_x^{\!\top} + Q`` solved by `lyapdkr`
is what you'd condition on to write down the long-run distribution of
``(k, z)`` or compute a Kalman filter.

The closed-form AR(1) variance ``\sigma^2/(1-\rho^2)`` gives the analytic
TFP marginal — useful as a sanity reading, and shown below:

```jldoctest
julia> using MatrixEquationsAD

julia> include(joinpath(pkgdir(MatrixEquationsAD), "test", "example_matrices", "rbc.jl"));

julia> using .RBCExampleMatrices: rbc_first_order_assembly

julia> p = [0.5, 0.95, 0.2, 0.02, 0.01, 0.01];

julia> A, B, _ = rbc_first_order_assembly(p);

julia> r = klein_map(A, B; threshold = 1.0e-6);

julia> Q = [0.0 0.0; 0.0 p[5]^2];

julia> V = lyapdkr(r.h_x, Q);

julia> round(sqrt(V[2, 2]); sigdigits = 4)         # stationary std. of z (% units)
0.01021

julia> V[2, 2] ≈ p[5]^2 / (1 - p[3]^2)             # matches AR(1) closed form
true

julia> round(sqrt(V[1, 1]); sigdigits = 3)         # std. of capital deviation
0.265
```

TFP fluctuates at roughly 1% in stationary equilibrium; capital, which
absorbs cumulative TFP shocks through the policy term ``h_x[1,2]``,
fluctuates ~25× more.

Differentiating either ``V`` or any summary of it with respect to
``p`` works straight through the `rbc_first_order_assembly → klein_map →
lyapdkr` pipeline.

Enzyme reverse mode against a scalar loss:

```julia
using LinearAlgebra: dot
using Enzyme
using MatrixEquationsAD

A = [0.55  0.08; -0.04  0.42]
C = [1.0   0.2;   0.2   0.7]
W = [0.3  -0.1;   0.2   0.5]

loss(A, C, W) = dot(W, lyapdkr(A, C))

A_bar = zero(A); C_bar = zero(C)
Enzyme.autodiff(
    Reverse, loss, Active,
    Duplicated(copy(A), A_bar),
    Duplicated(copy(C), C_bar),
    Const(W),
)
# A_bar = Y A X' + Y' A X and C_bar = Y, where Y solves the transposed
# Kronecker system M' vec(Y) = vec(P(W)).
```
