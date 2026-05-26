# Kronecker Discrete Lyapunov

`lyapdkr(A, C)` solves the discrete Lyapunov equation

```math
A X A^\top - X + C = 0
```

via the Kronecker-vectorised form. Where `MatrixEquations.lyapd` runs a
Schur-based Bartels–Stewart sweep, `lyapdkr` LU-factorises the full
``(n^2) \times (n^2)`` Kronecker operator once and reuses that
factorisation across all tangent / cotangent directions. The output is
symmetrised, so nonsymmetric perturbations of ``C`` are projected onto
the symmetric solution manifold.

The same equation governs the stationary covariance of the QuantEcon
[Linear State Space Model](https://julia.quantecon.org/introduction_dynamics/linear_models.html)
``x_{t+1} = A\,x_t + w_{t+1}`` with ``w_t \sim \mathcal{N}(0, Q)``;
this page uses the Kronecker LU instead of the Schur sweep on the
[Discrete Lyapunov (Schur)](lyapd.md) page.

`MatrixEquationsAD` also exports `lyapdkr!(X, A, C)`, which writes the
solution into a caller-supplied `X`. Both forms accept a `M_ws` kwarg
that lets the caller reuse a single ``n^2 \times n^2`` scratch matrix
across calls. The in-place form, the kwarg, and a static-native
`SMatrix` path are detailed in [API variants](#api-variants) below.

Implementation pointers:

- `src/lyapdkr.jl` — `lyapdkr` / `lyapdkr!` plus the two private
  helpers `build_M!!` and `symmetrize!!`.
- `ext/enzyme_lyapdkr.jl` — Enzyme forward + augmented_primal + reverse
  rules for both `lyapdkr` and `lyapdkr!`.
- `ext/forwarddiff_lyapdkr.jl` — ForwardDiff `Dual` dispatch.
- `ext/MatrixEquationsADStaticArraysExt.jl` plus the
  `EnzymeStaticArrays` / `ForwardDiffStaticArrays` triple extensions —
  static-native path for `SMatrix` inputs.

## Primal

With column-major `vec` and the identity
``\operatorname{vec}(A X A^\top) = (A \otimes A)\operatorname{vec}(X)``,
the equation becomes

```math
\bigl(I_{n^2} - A \otimes A\bigr)\,\operatorname{vec}(X)
\;=\;
\operatorname{vec}(C).
```

Let ``M = I_{n^2} - A \otimes A`` and ``P(N) = \tfrac{1}{2}(N + N^\top)``
the symmetric projection. `lyapdkr` builds ``M``, runs `lu!(M)`,
solves, reshapes, and projects:

```math
X \;=\; P\!\left(\operatorname{reshape}(M^{-1}\operatorname{vec}(C),\; n,\; n)\right).
```

A singular ``M`` (equivalently, two eigenvalues of ``A`` with product
``1``; ``\rho(A) < 1`` is sufficient) raises `SingularException` from
`lu!`. Callers that need PSD / magnitude checks on the returned ``X``
own them.

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

`lyapdkr` and `lyapd` agree to round-off but take different paths:
`lyapd` runs an ``O(n^3)`` Schur sweep, `lyapdkr` LU-factorises an
``n^2 \times n^2`` Kronecker matrix.

## API variants

### Workspace reuse: `M_ws` kwarg

For callers that drive `lyapdkr` inside a hot loop (parameter sweep,
optimiser step, simulator), the per-call ``n^2 \times n^2`` `M` matrix
dominates the allocation footprint. Pass a pre-allocated `M_ws` to
share the buffer across calls:

```julia
M_ws = Matrix{Float64}(undef, n*n, n*n)
X1 = lyapdkr(A1, C1; M_ws)
X2 = lyapdkr(A2, C2; M_ws)
```

The kwarg flows through every Enzyme + ForwardDiff rule as a `Const`
non-differentiated scratch. Defaults to `nothing`, which preserves the
auto-allocation behaviour.

**Caveat for Enzyme reverse:** between `augmented_primal` and
`reverse`, the LU factor lives on Enzyme's tape and aliases `M_ws`'s
memory. Don't reuse the same `M_ws` for a second `lyapdkr` call inside
that window or the tape's LU will be corrupted.

### In-place output: `lyapdkr!`

```julia
lyapdkr!(X::StridedMatrix, A, C; M_ws = nothing)
```

Writes the solution into the caller-supplied `X` and returns it.
Carries the full set of AD rules — ForwardDiff `Dual` dispatch and
Enzyme forward / augmented_primal / reverse with `X` as a `Duplicated`
or `BatchDuplicated` annotation.

```jldoctest lyapdkr_small
julia> Xip = similar(A); lyapdkr!(Xip, A, C);

julia> Xip == X
true
```

### Static dispatch (`SMatrix`)

The StaticArrays extension provides

```julia
lyapdkr(A::SMatrix{N, N, T}, C::SMatrix{N, N, T})
```

with a compile-time dispatch on `N`. StaticArrays caps its native LU at
total elements ``\le 14 \times 14 = 196``; for the ``n^2 \times n^2``
pencil that means truly heap-free only at ``N \le 3``. The dispatch:

- **``N \le 3``**: fully static — `kron`, `lu`, `\`, `reshape`, and the
  symmetric projection are all `SMatrix`-typed. **Zero heap
  allocations.** The matching Enzyme forward rule and ForwardDiff
  `Dual` method live in the
  `MatrixEquationsADEnzymeStaticArraysExt` /
  `MatrixEquationsADForwardDiffStaticArraysExt` triple extensions and
  reuse a single static `lu(M)` across the primal plus every tangent /
  partial solve.
- **``N \ge 4``**: routes through the heap solver (one
  `lyapdkr(Matrix(A), Matrix(C))` call wrapped back into `SMatrix`),
  because StaticArrays' LU fallback at this size wraps a heap LU back
  into static form which costs more than just running the heap path
  once.

No Enzyme reverse rule is specialised for `SMatrix`; if you need
reverse on static inputs, call `Matrix(A)` / `Matrix(C)` first or
accept the heap dispatch.

## ForwardDiff JVP

For one tangent direction ``(dA, dC)``, differentiating
``M\,\operatorname{vec}(X) = \operatorname{vec}(C)`` gives

```math
\operatorname{vec}(d X_{\mathrm{raw}})
\;=\;
M^{-1}\,\operatorname{vec}\!\bigl(
dC \;+\; dA\,X\,A^\top \;+\; A\,X\,dA^\top
\bigr),
\qquad
dX \;=\; P(dX_{\mathrm{raw}}).
```

The LU of ``M`` is built once on the value layer and reused for every
tangent — a chunk of width ``N`` issues ``N`` back-substitutions
against the shared factorisation, batched into a single multi-RHS
`ldiv!` for the heap path. The Enzyme `BatchDuplicated` forward rule is
structurally identical.

## Enzyme VJP

For an upstream cotangent ``\bar X``, symmetrise
``S = P(\bar X) = \tfrac{1}{2}(\bar X + \bar X^\top)`` and solve the
transposed system

```math
\operatorname{vec}(Y) \;=\; M^{-\top}\,\operatorname{vec}(S).
```

Same LU as the JVP, stashed on Enzyme's tape so the reverse pass
reuses it. Parameter cotangents (``M`` bilinear in ``A``):

```math
\bar C \;\mathrel{+}=\; Y,
\qquad
\bar A
\;\mathrel{+}=\;
Y\,A\,X^\top \;+\; Y^\top\,A\,X.
```

## Example: RBC stationary distribution

Using the RBC policy values from the
[Klein Policy Map quick start](@ref "Quick start: parameters to policy"),
on the static (``N = 2``, fully heap-free) path:

```jldoctest lyapdkr_rbc
julia> using MatrixEquationsAD, StaticArrays

julia> h_x = SMatrix{2, 2}([
           0.9568351489231556  6.209371005755667;
           -3.3737787177631822e-18  0.20000000000000004
       ]);

julia> B_shock = SMatrix{2, 1}([0.0; -0.01]);

julia> Q = B_shock * B_shock'
2×2 SMatrix{2, 2, Float64, 4} with indices SOneTo(2)×SOneTo(2):
  0.0  -0.0
 -0.0   0.0001

julia> V = lyapdkr(h_x, Q)
2×2 SMatrix{2, 2, Float64, 4} with indices SOneTo(2)×SOneTo(2):
 0.0700541    0.000159976
 0.000159976  0.000104167

julia> V[2, 2] ≈ 0.01^2 / (1 - 0.2^2)   # matches AR(1) closed form
true
```

TFP fluctuates at roughly 1% in stationary equilibrium; capital, which
absorbs cumulative TFP shocks through ``h_x[1,2]``, fluctuates ~25×
more. The same call on `Matrix` inputs returns a `Matrix` with the
same numbers via the heap path.

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

The in-place form `lyapdkr!` carries the same rules; treat `X` as a
`Duplicated` annotation alongside `A` and `C`:

```julia
loss_inplace(X, A, C, W) = (lyapdkr!(X, A, C); dot(W, X))

X = similar(A); X_bar = make_zero(X)
A_bar = make_zero(A); C_bar = make_zero(C)
autodiff(
    Reverse, loss_inplace, Active,
    Duplicated(X, X_bar),
    Duplicated(copy(A), A_bar),
    Duplicated(copy(C), C_bar),
    Const(W),
)
```

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
