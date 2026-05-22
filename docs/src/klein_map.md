# Klein Policy Map

`klein_map(A, B; threshold)` returns the Klein/Sims first-order policy
``(g_x, h_x)`` for a DSGE pencil ``(A, B)`` with the Blanchard-Kahn
selection at margin ``\tau = \texttt{threshold}``. Two API forms exist:

- `klein_map(A, B; threshold)` — heap-allocated, returns a NamedTuple
  `(; g_x, h_x)`.
- `klein_map!(g_x, h_x, A, B; threshold)` — mutating, writes into
  caller-supplied buffers; the in-place AD rules use a more memory-efficient
  reduced-Sylvester factorization.

Implementation pointers:

- `src/klein_map.jl` — primal solve via ordered generalized Schur (BK ordering
  on `|α| ≥ (1 − τ)|β|`) followed by the Klein/Sims algebra.
- `ext/klein_map_derivatives.jl` — `_klein_bigk_plan` / `_klein_bigk_jvp` /
  `_klein_bigk_vjp` for the OOP path; `_klein_structured_plan` /
  `_klein_structured_jvp` / `_klein_structured_vjp` for the in-place path.
- `ext/enzyme_klein_map.jl` and `ext/forwarddiff_klein_map.jl` — the AD
  frontends.

## Where the Primal Comes From

Consider the linearised equilibrium of a DSGE model around a deterministic
steady state. Partition the state into ``n_x`` predetermined variables
``x_t`` and ``n_y`` non-predetermined (jump) variables ``y_t``, and write
``z_t = [x_t;\, y_t] \in \mathbb{R}^{n}`` with ``n = n_x + n_y``. The
linearised model can be cast in the pencil form

```math
A\,\mathbb{E}_t[z_{t+1}] \;+\; B\,z_t \;=\; 0,
\qquad
A, B \in \mathbb{R}^{n \times n}.
```

Klein (2000) shows that the unique bounded solution under the
Blanchard-Kahn condition takes the linear form

```math
y_t \;=\; g_x\, x_t,
\qquad
x_{t+1} \;=\; h_x\, x_t \;+\; \text{(shock terms)},
```

i.e. the jumps are an affine function of the predetermined state, and the
predetermined state follows an autonomous linear law of motion. The
package's `klein_map(A, B; threshold)` returns exactly that
``(g_x, h_x)``.

## Algorithm

Stacked column-wise as ``z = [x;\,y]``, define the embedding

```math
\Psi
=
\begin{bmatrix}
I_{n_x} \\
g_x
\end{bmatrix}
\in \mathbb{R}^{n \times n_x},
```

so that ``z_t = \Psi\,x_t`` under the policy, and ``z_{t+1} = \Psi\,h_x\,x_t``.
Substituting into the pencil equation,

```math
A\,\Psi\,h_x\,x_t \;+\; B\,\Psi\,x_t \;=\; 0
\qquad \text{for all } x_t,
```

which is the implicit policy equation

```math
F(A, B, g_x, h_x)
\;:=\;
A\,\Psi\, h_x \;+\; B\,\Psi
\;=\;
0
\quad\in\;\mathbb{R}^{n \times n_x}.
\tag{F}
```

`klein_map` constructs the solution by Klein's QZ-based procedure:

1. Take the real generalised Schur decomposition
   ``A = Q\, S\, Z^{\!\top}``, ``B = Q\, T\, Z^{\!\top}`` with ``Q``, ``Z``
   orthogonal, ``S`` quasi-upper-triangular, ``T`` upper triangular.
   Eigenvalues are the pairs ``(\alpha_i,\beta_i)``.
2. Reorder so that the *unstable* generalised eigenvalues
   (``|\alpha_i| \ge (1 - \tau)|\beta_i|``) come first. ``\tau`` is the
   `threshold` keyword; the BK count ``n_x`` is the number of selected
   eigenvalues and must equal the predetermined-block dimension.
3. Partition into ``b = 1\!:\!n_x`` (unstable, "big") and
   ``l = (n_x+1)\!:\!n`` (stable) blocks, and read off
   ```math
   g_x \;=\; -\,Z_{l l}^{-\!\top}\, Z_{b l}^{\!\top},
   \qquad
   h_x \;=\; -\,\mathrm{blob}^{-1}\, S_{b b}^{-1}\, T_{b b}\,\mathrm{blob},
   \quad
   \mathrm{blob} \;=\; Z_{b b}^{\!\top} \;+\; Z_{l b}^{\!\top}\,g_x.
   ```

The construction is a direct port of `DifferentiablePerturbation.jl`'s
`first_order_perturbation!`; see `src/klein_map.jl` for the indexing.

## Post-Conditions

At the policy returned by `klein_map`, the following hold (each is unit-tested
on the RBC, RBC_SV, SGU, FVGQ, and SW07PFEIFER fixtures):

- **Implicit policy residual:** ``F(A, B, g_x, h_x) = A\,\Psi\,h_x + B\,\Psi = 0``
  in floating-point arithmetic (typical residual is below ``10^{-8}`` on
  the standard fixtures and is what `test/test_klein_map.jl` checks).
- **Saddle-path:** every ``x_0 \in \mathbb{R}^{n_x}`` generates a bounded
  trajectory ``z_t = \Psi\, h_x^{\,t}\, x_0``.
- **Stable transition:** the spectrum of ``h_x`` lies strictly inside the
  unit disc (``\rho(h_x) < 1``), with eigenvalues equal to the stable
  generalised eigenvalues ``\beta_i/\alpha_i`` of the inverse pencil.
- **Static algebraic post-conditions** (consequences of the QZ structure
  the algorithm reads off):
  ``Z_{l l}^{\!\top}\, g_x \;=\; -\,Z_{b l}^{\!\top}`` and
  ``\mathrm{blob}\, h_x \;=\; -\,S_{b b}^{-1}\, T_{b b}\,\mathrm{blob}``.

The factorisation assumptions for the AD rules below are:

- the BK split returns exactly ``n_x`` unstable eigenvalues with
  ``0 < n_x < n``;
- ``Z_{l l}`` and ``\mathrm{blob}`` are nonsingular (used in the primal);
- the linearisation of ``F`` (the Jacobian ``K`` below) is nonsingular at
  the primal solution.

Failures throw `ErrorException` (BK / consistency check) or the underlying
LAPACK exception (Schur / LU).

Calling `klein_map` on the canonical RBC pencil (assembled by
`RBCExampleMatrices.rbc_first_order_assembly(p)` from
[`test/example_matrices/rbc.jl`](https://github.com/QuantEcon/MatrixEquationsAD.jl/blob/main/test/example_matrices/rbc.jl),
itself a readable translation of
[`DifferentiablePerturbation.jl`'s code-generated `RBC.first_order_assembly!`](https://github.com/HighDimensionalEconLab/DifferentiablePerturbation.jl/blob/main/src/models/RBC_generated/first_order_ip.jl)):

```jldoctest
julia> using MatrixEquationsAD

julia> include(joinpath(pkgdir(MatrixEquationsAD), "test", "example_matrices", "rbc.jl"));

julia> using .RBCExampleMatrices: rbc_first_order_assembly

julia> A, B, n_x = rbc_first_order_assembly([0.5, 0.95, 0.2, 0.02, 0.01, 0.01]);

julia> r = klein_map(A, B; threshold = 1.0e-6);

julia> size(r.g_x), size(r.h_x)
((3, 2), (2, 2))
```

`r.g_x[1, 2]` is the static response of consumption to a TFP innovation;
`r.h_x[1, 2]` is how a TFP shock propagates into next period's capital.

## Why AD Rules Operate on (F), Not on the QZ Factors

In principle one could compute ``(\bar A, \bar B)`` by differentiating the
generalised-Schur factors ``(S, T, Q, Z)`` and then back-propagating through
the Klein algebra. We do **not** do that, and the docs derive the AD rules
directly from the implicit residual ``(F)``. The reasons:

- **The QZ factors are not unique.** Within each repeated or clustered
  eigenvalue group, ``Q`` and ``Z`` admit a gauge freedom: any orthogonal
  transformation that commutes with the Schur block structure leaves
  ``(S, T)`` unchanged. The tangent of ``(Q, Z)`` is therefore ill-defined
  precisely where eigenvalues coalesce — e.g. the FVGQ pencil has multiple
  clusters at ``|\alpha/\beta| = 1``.
- **Block structure changes are discontinuous.** Real-Schur ``S`` mixes
  ``1\times 1`` real blocks with ``2\times 2`` complex-conjugate blocks. A
  perturbation that pushes a real pair into a complex pair (or vice versa)
  changes the block layout discontinuously. The reordering step that
  enforces the BK selection has the same issue at threshold crossings.
- **Klein's policy ``(g_x, h_x)`` is gauge-invariant.** It depends only on
  the *stable deflating subspace*, not on the particular orthogonal basis
  chosen for it. Differentiating ``(F)`` reflects that invariance: the
  tangents ``(dg_x, dh_x)`` are uniquely defined even when ``(dQ, dZ)`` are
  not.
- **The QZ-tangent rule needs perturbation theory the IFT does not.** The
  implicit-function theorem applied to ``(F)`` only requires nonsingularity
  of the linearised operator ``K`` below, which holds generically; the
  QZ-tangent approach needs eigenvalue/eigenvector perturbation bounds
  that degrade as clusters tighten.

So the AD rules differentiate ``(F)`` itself. The locally constant
ingredients (BK selection, real-vs-complex block layout, threshold
position) are treated as fixed; the assumption is that no eigenvalue
crosses the threshold under perturbation.

## Jacobian of the Implicit Equation

Let ``E_y = [\,0_{n_x \times n_y};\, I_{n_y}\,] \in \mathbb{R}^{n \times n_y}``,
and split the columns of ``A`` and ``B`` as ``A_2 = A E_y``,
``B_2 = B E_y`` (the trailing ``n_y`` columns). Differentiating ``F = 0``
at fixed ``(A, B)`` gives

```math
A\Psi\,dh_x
\;+\;
\bigl(A_2 \, dg_x \, h_x + B_2 \, dg_x\bigr)
\;=\;
-\,dA\,D - dB\,\Psi.
```

Vectorising column-major and stacking the unknowns as
``[\operatorname{vec}(dh_x);\,\operatorname{vec}(dg_x)] \in \mathbb{R}^{n n_x}``,

```math
K
\begin{bmatrix}
\operatorname{vec}(dh_x) \\
\operatorname{vec}(dg_x)
\end{bmatrix}
\;=\;
-\operatorname{vec}\!\bigl(dA\,D + dB\,\Psi\bigr),
```

with

```math
K
\;=\;
\Bigl[\,
I_{n_x}\otimes A\Psi
\;\Big|\;
h_x^{\!\top}\otimes A_2 \;+\; I_{n_x}\otimes B_2
\,\Bigr]
\;\in\;\mathbb{R}^{n n_x \times n n_x}.
```

Two equivalent linear-algebra representations are implemented and produce
identical tangents up to floating-point roundoff. Both are correct on
degenerate-cluster pencils (e.g. the FVGQ fixture) because they
differentiate the implicit equation rather than the non-unique QZ factors.

## Big-K Factorisation (OOP `klein_map`)

The OOP rule builds ``K`` once as a dense ``(n n_x) \times (n n_x)`` block
matrix and stores its LU factorisation in the plan returned by
`_klein_bigk_plan`. Both forward and reverse modes reuse this single
factorisation.

### ForwardDiff / Enzyme JVP

For each tangent direction ``(dA, dB)``,

```math
\begin{bmatrix}
\operatorname{vec}(dh_x) \\
\operatorname{vec}(dg_x)
\end{bmatrix}
\;=\;
K^{-1}\,\operatorname{vec}\!\bigl(-(dA\,D + dB\,\Psi)\bigr).
```

`_klein_bigk_jvp` forms the right-hand side ``R = -(dA\,D + dB\,\Psi)``,
performs one LU back-substitution, and reshapes the solution into
``(dh_x, dg_x)``. ForwardDiff chunks of width ``N`` and Enzyme
`BatchDuplicated` of width ``N`` both issue ``N`` such back-substitutions
against the shared factorisation.

### Enzyme VJP

Pack the output cotangents in the same block order as ``K``:

```math
u
=
\begin{bmatrix}
\operatorname{vec}(\bar h_x) \\
\operatorname{vec}(\bar g_x)
\end{bmatrix}
\;\in\;\mathbb{R}^{n n_x},
\qquad
\lambda = K^{-\!\top} u,
\qquad
\Lambda = \operatorname{reshape}(\lambda,\, n,\, n_x).
```

Because the residual ``R`` depends on ``A`` and ``B`` only through
``R = -(dA\,D + dB\,\Psi)``, the parameter cotangents are

```math
\bar A
\;\mathrel{+}=\;
-\,\Lambda\,D^{\!\top}
\;=\;
-\,\Lambda\,h_x^{\!\top}\,\Psi^{\!\top},
\qquad
\bar B
\;\mathrel{+}=\;
-\,\Lambda\,\Psi^{\!\top}.
```

`_klein_bigk_vjp` performs the single transposed LU solve, reshapes ``\lambda``,
and applies these outer products.

## Reduced-Sylvester Factorisation (in-place `klein_map!`)

The in-place rule never materialises the full ``K``. It instead builds the
``n \times n`` block

```math
C_0
=
\begin{bmatrix}
A\Psi & B_2
\end{bmatrix}
\;\in\;\mathbb{R}^{n \times n},
```

LU-factorises ``C_0``, solves once for the auxiliary matrix
``J = C_0^{-1} A E_y \in \mathbb{R}^{n \times n_y}``, and splits it into

```math
J
=
\begin{bmatrix}
J_x \\
J_y
\end{bmatrix},
\qquad
J_x \in \mathbb{R}^{n_x \times n_y},
\qquad
J_y \in \mathbb{R}^{n_y \times n_y}.
```

Real Schur factorisations
``J_y = Q_y S_y Q_y^{\!\top}`` and ``h_x = Q_h S_h Q_h^{\!\top}`` are
computed once and reused.

### JVP

Let ``Y = C_0^{-1} R`` with ``R = -(dA\,D + dB\,\Psi)``, and split
``Y = [Y_x;\,Y_y]``. Premultiplying the tangent equation by
``C_0^{-1}`` and matching block rows gives

```math
\boxed{\;\;
dg_x \;+\; J_y\,dg_x\,h_x \;=\; Y_y
\;\;}
```

which is a discrete Sylvester / Stein equation. In the Schur frame
``\tilde X = Q_y^{\!\top} dg_x\, Q_h`` it becomes

```math
S_y\,\tilde X\,S_h \;+\; \tilde X \;=\; Q_y^{\!\top} Y_y Q_h,
```

solved in place by `MatrixEquations.sylvds!`. Back-transforming and
substituting into the ``h_x`` block,

```math
dh_x \;=\; Y_x \;-\; J_x\,dg_x\,h_x.
```

### VJP

Given output cotangents ``(\bar g_x, \bar h_x)``, the elimination above
flows backwards as follows. The ``dh_x`` formula contributes to ``\bar Y_x``
and induces a correction on the ``dg_x`` cotangent:

```math
\bar Y_x \;=\; \bar h_x,
\qquad
\tilde{\bar g}_x
\;=\;
\bar g_x \;-\; J_x^{\!\top}\,\bar h_x\,h_x^{\!\top}.
```

The corrected ``\tilde{\bar g}_x`` is the right-hand side of the adjoint
Stein equation

```math
J_y^{\!\top}\,Z\,h_x^{\!\top} \;+\; Z \;=\; \tilde{\bar g}_x,
\qquad
\bar Y_y \;=\; Z,
```

again solved with `sylvds!(S_y, S_h, ⋅; adjA = true, adjB = true)` in
Schur coordinates. Reassembling

```math
\bar Y
=
\begin{bmatrix}
\bar h_x \\
Z
\end{bmatrix},
\qquad
\Lambda \;=\; C_0^{-\!\top}\,\bar Y,
```

requires one transposed LU back-substitution against the same ``C_0``
factorisation. The parameter cotangents are then identical to the big-K
form:

```math
\bar A
\;\mathrel{+}=\;
-\,\Lambda\,h_x^{\!\top}\,\Psi^{\!\top},
\qquad
\bar B
\;\mathrel{+}=\;
-\,\Lambda\,\Psi^{\!\top}.
```

## Which Path Wins

| Pencil size | Big-K (OOP) | Reduced Sylvester (in-place) |
| --- | --- | --- |
| Small (n ≲ 15) | Faster — LU is cheap, single multi-RHS solve. | Setup overhead (two Schurs + one LU) dominates. |
| Large (n ≳ 30) | LU of ``(n n_x)^2`` becomes expensive. | Wins; only ``n_y \times n_y`` Schur + per-tangent `sylvds!`. |

At the SW07PFEIFER pencil (``n = 42``, ``n_x = 20``) the in-place reverse
runs roughly five times faster than the OOP reverse. At the RBC pencil
(``n = 5``) the OOP path wins by ~50%.

## SMatrix Dispatch

`StaticArrays` support requires explicit `Val(n_x)` so the return type is
statically sized:

```julia
using StaticArrays
As = SMatrix{5, 5, Float64}(A);  Bs = SMatrix{5, 5, Float64}(B)
r = klein_map(As, Bs, Val(2); threshold = 1.0e-6)
@assert r.g_x isa SMatrix{3, 2, Float64}
@assert r.h_x isa SMatrix{2, 2, Float64}
```

Without the `Val`, an SMatrix input falls through to the heap dispatch and
returns plain `Matrix` outputs — type-stable code-gen requires the BK split
to be known at compile time, and a runtime BK count cannot satisfy that.

## Examples

All snippets below assemble the RBC pencil from a parameter vector and
then call the policy map, so AD can flow from `p` through the assembly
into `klein_map`:

```julia
include(joinpath(pkgdir(MatrixEquationsAD), "test", "example_matrices", "rbc.jl"))
using .RBCExampleMatrices: rbc_first_order_assembly

p = [0.5, 0.95, 0.2, 0.02, 0.01, 0.01]
A, B, n_x = rbc_first_order_assembly(p)
n_y = size(A, 1) - n_x
```

A gradient of a policy-summary functional with respect to the parameter
vector uses ForwardDiff straight through:

```julia
using ForwardDiff
function policy_loss(p)
    A, B, _ = rbc_first_order_assembly(p)
    r = klein_map(A, B; threshold = 1.0e-6)
    return sum(r.g_x) + sum(r.h_x)
end
∇p = ForwardDiff.gradient(policy_loss, p)
```

ForwardDiff Jacobian of the flattened policy against the flattened pencil:

```julia
using ForwardDiff
using MatrixEquationsAD

n = size(A, 1)
function klein_vec(x)
    A_x = reshape(x[1:(n*n)], n, n)
    B_x = reshape(x[(n*n+1):end], n, n)
    r = klein_map(A_x, B_x; threshold = 1.0e-6)
    return [vec(r.g_x); vec(r.h_x)]
end
J = ForwardDiff.jacobian(klein_vec, [vec(A); vec(B)])
# size(J) == (n_y*n_x + n_x*n_x, 2*n*n) == (10, 50) for the RBC pencil.
```

Enzyme batched forward of width 4 on a scalar loss:

```julia
using Enzyme, Random

rng = MersenneTwister(0)
dA_lanes = ntuple(_ -> randn(rng, n, n), Val(4))
dB_lanes = ntuple(_ -> randn(rng, n, n), Val(4))

loss(A, B) = (r = klein_map(A, B); sum(r.g_x) + sum(r.h_x))

(out,) = Enzyme.autodiff(
    Forward, loss, BatchDuplicated,
    BatchDuplicated(copy(A), dA_lanes),
    BatchDuplicated(copy(B), dB_lanes),
)
# `out` is a NamedTuple of the four scalar JVPs, one per tangent direction.
```

In-place reverse on a `klein_map!` call:

```julia
g_x = zeros(n_y, n_x);  h_x = zeros(n_x, n_x)
A_bar = zero(A);  B_bar = zero(B)

Enzyme.autodiff(
    Reverse, (g, h, A, B) -> (klein_map!(g, h, A, B); sum(g) + sum(h)),
    Active,
    Duplicated(g_x, zero(g_x)),
    Duplicated(h_x, zero(h_x)),
    Duplicated(copy(A), A_bar),
    Duplicated(copy(B), B_bar),
)
```

The `klein_map!` rule uses the reduced-Sylvester path; the OOP `klein_map`
loss above routes through the big-K rule instead.
