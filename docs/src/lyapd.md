# Discrete Lyapunov (Schur)

`lyapd(A, C)` solves the discrete Lyapunov equation

```math
A X A^\top - X + C = 0
```

using the Schur-based Bartels–Stewart algorithm from
[`MatrixEquations.jl`](https://github.com/andreasvarga/MatrixEquations.jl):
the upstream solver Schur-factorises `A` once and runs a triangular
sweep (`MatrixEquations.lyapds!`) on the transformed right-hand side.
For non-toy state dimensions this is the right default — the cost is
``O(n^3)`` per solve, vs. ``O(n^6)`` for the
[Kronecker-vec form](lyapdkr.md). `MatrixEquationsAD` provides custom
AD rules that wrap that solver in a cache-aware shadow so a single
`schur(A)` is reused across all tangent / cotangent directions.

`MatrixEquationsAD` also exports `lyapd!(X, A, C)`, which writes the
solution into a caller-supplied `X` and returns `nothing`. The in-place
form shares the same Schur cache plumbing and carries an analogous full
set of AD rules — ForwardDiff `Dual` dispatch, Enzyme forward
(Duplicated / BatchDuplicated), and Enzyme reverse (augmented_primal +
reverse). The reverse rule stashes the Schur factors on the tape so the
adjoint pass never re-Schurs `A`. A cache-taking overload
`lyapd!(X, cache::LyapDSchurCache, C)` lets the AD rules share one
`schur(A)` across all forward tangent directions or reverse cotangents.

Implementation pointers:

- `ext/enzyme_lyapunov.jl` — `LyapDSchurCache`, `lyapdfactor`,
  `lyapdsolve`, `lyapdadjointsolve`, the cache-aware `lyapd(A, C)`
  shadow, and the Enzyme forward / reverse rules.
- `ext/forwarddiff_lyapunov.jl` — the ForwardDiff `Dual` dispatch on
  `lyapd(A, C)`, which uses the same cache primitives.
- `MatrixEquations.lyapds!` — upstream in-place kernel that operates on
  `A` in real or complex Schur form (`adj = false / true` for transpose
  variants).

## Primal

Define the linear operator

```math
L_A[X] \;=\; X \;-\; A\,X\,A^\top.
```

The primal equation is ``L_A[X] = C``. The cache-aware shadow
precomputes ``\mathrm{schur}(A) = (T, Z)`` with
``A = Z\,T\,Z^\top``, and stores it in a `LyapDSchurCache`. With that
cache available, the solve transforms the right-hand side, runs the
upstream Schur-form kernel, and untransforms:

```math
\tilde C \;=\; Z^\top\,C\,Z,
\qquad
T\,\tilde X\,T^\top \;-\; \tilde X \;+\; \tilde C \;=\; 0,
\qquad
X \;=\; Z\,\tilde X\,Z^\top.
```

For ``\texttt{C::Symmetric}`` this is exactly one call to
`lyapds!(cache.T, rhs)`; for general dense ``C`` the dispatch routes to
`sylvds!(-cache.T, cache.T, rhs; adjB = true)`, which solves the same
triangular system without enforcing symmetry.

Existence and uniqueness require that no two eigenvalues of ``A``
multiply to one. ``\rho(A) < 1`` is sufficient and is the case of
interest for stationary-covariance applications below.

## Worked example

A 2×2 sanity case:

```jldoctest lyapd_small
julia> using MatrixEquations: lyapd

julia> using MatrixEquationsAD

julia> A = [0.55 0.08; -0.04 0.42]
2×2 Matrix{Float64}:
  0.55  0.08
 -0.04  0.42

julia> C = [1.0 0.2; 0.2 0.7]
2×2 Matrix{Float64}:
 1.0  0.2
 0.2  0.7

julia> X = lyapd(A, C)
2×2 Matrix{Float64}:
 1.47343   0.253679
 0.253679  0.84244

julia> isapprox(A * X * A' - X + C, zeros(2, 2); atol = 1.0e-12)
true
```

`lyapd!` writes the same `X` into a caller-supplied buffer:

```jldoctest lyapd_small
julia> Xip = zeros(2, 2);

julia> lyapd!(Xip, A, C);

julia> Xip == X
true
```

## ForwardDiff JVP

For one tangent direction ``(d A, d C)``, differentiating
``A X A^\top - X + C = 0`` and applying the same operator gives

```math
L_A[d X]
\;=\;
d C \;+\; d A\,X\,A^\top \;+\; A\,X\,d A^\top,
```

i.e. ``d X`` solves another discrete Lyapunov equation against the same
``A``. The ForwardDiff dispatch builds the cache once and calls
`lyapdsolve(cache, rhs_i)` for each partial direction in the chunk —
``N`` `lyapds!` triangular sweeps against the shared Schur factors.

The Enzyme `BatchDuplicated` forward rule is structurally identical:
one `schur(A)` per outer call, then one triangular solve per tangent.

## Enzyme VJP

The upstream Schur solver does not enforce symmetry on ``X``
(`lyapds!` returns a symmetric result only when ``C`` is `Symmetric`),
so the formulas below apply to the general dense map. The
`Symmetric`-C dispatch uses the same adjoint solve, then projects the
accumulated parameter cotangent ``\bar C`` onto the symmetric manifold
before adding it to the shadow buffer (so a `Symmetric` parameter is
differentiated only against symmetric perturbations).

Let ``\bar X`` be the cotangent on the output. Define ``Y`` by the
adjoint Lyapunov solve

```math
L_A^*[Y] \;=\; Y \;-\; A^\top\,Y\,A \;=\; \bar X,
```

implemented as `lyapdadjointsolve(cache, X̄)` (which routes to
`lyapds!(cache.T, rhs; adj = true)` for `Symmetric` cotangents and to
the transposed `sylvds!` variant otherwise). The parameter cotangents
are then

```math
\bar C \;\mathrel{+}=\; Y,
\qquad
\bar A
\;\mathrel{+}=\;
Y\,A\,X^\top \;+\; Y^\top\,A\,X.
```

The augmented primal copies the cached Schur ``(T, Z)`` and the primal
``X`` onto the tape; the reverse pass performs one triangular adjoint
solve plus the two outer products. Multiple reverse cotangents (e.g.
under Enzyme `BatchReverse`) reuse the same cached Schur, exactly as
forward chunks do.

References:

- MatrixEquations.jl documents `lyapd` as solving ``A X A^\top - X + C = 0``
  in its
  [Lyapunov solver documentation](https://andreasvarga.github.io/MatrixEquations.jl/latest/lyapunov.html).
- Kao and Hennequin derive forward and reverse rules for Lyapunov,
  Sylvester, and Riccati equations in
  [arXiv:2011.11430](https://arxiv.org/abs/2011.11430).

## Example: RBC stationary covariance, end-to-end

After the Klein/Sims policy is in hand, the predetermined state evolves
as ``x_{t+1} = h_x\, x_t + B_{\text{shock}}\,\varepsilon_{t+1}`` with
``Q = B_{\text{shock}}\,B_{\text{shock}}^\top`` the one-step
innovation covariance. The stationary covariance ``V`` of ``x_t``
satisfies the discrete Lyapunov equation
``V = h_x\,V\,h_x^\top + Q``, which is
``A X A^\top - X + C = 0`` with ``A = h_x`` and ``C = Q``.

Using the RBC policy values from the
[Klein Policy Map quick start](klein_map.md#quick-start:-parameters-to-policy):

```jldoctest lyapd_rbc
julia> using LinearAlgebra: Symmetric

julia> using MatrixEquations: lyapd

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

julia> Q = Symmetric(B_shock * transpose(B_shock))
2×2 Symmetric{Float64, Matrix{Float64}}:
 0.0  0.0
 0.0  0.0001

julia> V = lyapd(h_x, Q)
2×2 Matrix{Float64}:
 0.0700541    0.000159976
 0.000159976  0.000104167

julia> V[2, 2] ≈ 0.01^2 / (1 - 0.2^2)   # matches AR(1) closed form
true
```

The TFP stationary standard deviation is ``\sigma/\sqrt{1-\rho^2}``,
~1%; the implied capital standard deviation ``\sqrt{V_{11}}`` is ~26%,
matching the cumulative-shock interpretation of ``h_x[1,2]``.

Wrapping `Q` as `Symmetric` routes the AD rules onto the
`lyapds!`-based path; passing a plain `Matrix` would dispatch onto the
`sylvds!`-based path with the same result. Either way `lyapd` is the
right call here — its cost is ``O(n^3)`` in the state dimension, and
`MatrixEquationsAD` adds a single `schur(A)` cache reused across all AD
directions on top.

## Differentiating through `lyapd`

ForwardDiff against a scalar loss:

```julia
using ForwardDiff
using MatrixEquations: lyapd
using MatrixEquationsAD

A = [0.55 0.08; -0.04 0.42]
C = [1.0  0.2;  0.2 0.7]

loss(A, C) = sum(lyapd(A, C))
∇A = ForwardDiff.gradient(A -> loss(A, C), A)
```

Enzyme reverse on the same loss:

```julia
using Enzyme: Active, Const, Duplicated, Reverse, autodiff, make_zero
using LinearAlgebra: dot
using MatrixEquations: lyapd
using MatrixEquationsAD

A = [0.55 0.08; -0.04 0.42]
C = [1.0  0.2;  0.2 0.7]
W = [0.4 -0.1; -0.1 0.7]

weighted(A, C, W) = dot(W, lyapd(A, C))

A_bar = make_zero(A);  C_bar = make_zero(C)
autodiff(
    Reverse, weighted, Active,
    Duplicated(A, A_bar), Duplicated(C, C_bar), Const(W),
)
# A_bar, C_bar now hold the gradients.
```
