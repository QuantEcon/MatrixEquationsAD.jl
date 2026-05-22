# Klein Policy Map

`klein_map(A, B; threshold)` extracts the Klein/Sims first-order policy
``(g_x, h_x)`` from a DSGE-style linearisation. Two API forms exist:

- `klein_map(A, B; threshold)` — heap-allocated, returns the
  `NamedTuple` `(; g_x, h_x)`.
- `klein_map!(g_x, h_x, A, B; threshold)` — mutating, writes into
  caller-supplied buffers; the in-place AD rules use a more
  memory-efficient reduced-Sylvester factorisation.

Implementation pointers:

- `src/klein_map.jl` — primal solve via ordered generalised Schur (BK
  ordering on ``|\alpha| \ge (1 - \tau)|\beta|``) followed by the
  Klein/Sims algebra.
- `ext/klein_map_derivatives.jl` — `_klein_bigk_plan` /
  `_klein_bigk_jvp` / `_klein_bigk_vjp` for the out-of-place path;
  `_klein_structured_plan` / `_klein_structured_jvp` /
  `_klein_structured_vjp` for the in-place path.
- `ext/enzyme_klein_map.jl`, `ext/forwarddiff_klein_map.jl` — AD
  frontends.

## Where the primal comes from

Consider the linearised equilibrium of a DSGE model around a
deterministic steady state. Partition the state into ``n_x``
predetermined variables ``x_t`` and ``n_y`` non-predetermined (jump)
variables ``y_t``, and write ``z_t = [x_t;\, y_t] \in \mathbb{R}^{n}``
with ``n = n_x + n_y``. The linearised model takes the form

```math
A\,\mathbb{E}_t[z_{t+1}] \;+\; B\,z_t \;=\; 0,
\qquad
A, B \in \mathbb{R}^{n \times n}.
```

Under the Blanchard–Kahn condition the unique bounded solution is

```math
y_t \;=\; g_x\, x_t,
\qquad
x_{t+1} \;=\; h_x\, x_t \;+\; \text{(shock terms)},
```

i.e. the jumps are an affine function of the predetermined state, and
the predetermined state follows an autonomous linear law of motion.
`klein_map(A, B; threshold)` returns exactly that ``(g_x, h_x)``.

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

so that ``z_t = \Psi\,x_t`` under the policy, and
``z_{t+1} = \Psi\,h_x\,x_t``. Substituting into ``A\,z_{t+1} + B\,z_t = 0``
gives the implicit policy equation

```math
F(A, B, g_x, h_x)
\;:=\;
A\,\Psi\,h_x \;+\; B\,\Psi
\;=\;
0
\quad\in\;\mathbb{R}^{n \times n_x}.
\tag{F}
```

`klein_map` constructs the solution by Klein's QZ-based procedure:

1. Take the real generalised Schur decomposition
   ``A = Q\,S\,Z^\top``, ``B = Q\,T\,Z^\top`` with ``Q``, ``Z``
   orthogonal, ``S`` quasi-upper-triangular, ``T`` upper triangular.
   Eigenvalues are the pairs ``(\alpha_i,\beta_i)``.
2. Reorder so that the *unstable* generalised eigenvalues
   (``|\alpha_i| \ge (1 - \tau)|\beta_i|``) come first. ``\tau`` is the
   `threshold` keyword; the BK count ``n_x`` is the number of selected
   eigenvalues and must equal the predetermined-block dimension.
3. Partition into ``b = 1:n_x`` (unstable, "big") and
   ``l = (n_x+1):n`` (stable) blocks, and read off

   ```math
   g_x \;=\; -\,Z_{ll}^{-\top}\, Z_{bl}^\top,
   \qquad
   h_x \;=\; -\,W^{-1}\, S_{bb}^{-1}\, T_{bb}\, W,
   \qquad
   W \;=\; Z_{bb}^\top \;+\; Z_{lb}^\top\,g_x.
   ```

The construction is a direct port of
[`DifferentiablePerturbation.jl`](https://github.com/HighDimensionalEconLab/DifferentiablePerturbation.jl)'s
`first_order_perturbation!`; see `src/klein_map.jl` for the indexing.

## Post-conditions

At the policy returned by `klein_map`, each of the following holds (and
is unit-tested on the RBC, RBC\_SV, SGU, FVGQ, and SW07PFEIFER fixtures):

- **Implicit policy residual:**
  ``F(A, B, g_x, h_x) = A\,\Psi\,h_x + B\,\Psi = 0`` in floating-point
  arithmetic (typical residual is below ``10^{-8}``).
- **Saddle-path:** every ``x_0 \in \mathbb{R}^{n_x}`` generates a
  bounded trajectory ``z_t = \Psi\,h_x^{\,t}\,x_0``.
- **Stable transition:** the spectrum of ``h_x`` lies strictly inside
  the unit disc (``\rho(h_x) < 1``), with eigenvalues equal to the
  stable generalised eigenvalues ``\beta_i/\alpha_i`` of the pencil.

The factorisation assumptions for the AD rules below are: the BK split
returns exactly ``n_x`` unstable eigenvalues with ``0 < n_x < n``;
``Z_{ll}`` and ``W`` are nonsingular; and the linearisation of ``F``
(the Jacobian ``K`` below) is nonsingular at the primal solution.
Failures throw `ErrorException` (BK / consistency check) or the
underlying LAPACK exception (Schur / LU).

## Quick start: parameters to policy

The bundled example is a standard RBC model. Its first-order
linearisation has the parameter vector

```math
p = [\alpha,\ \beta,\ \rho,\ \delta,\ \sigma,\ \Omega_1]
  = [0.5,\ 0.95,\ 0.2,\ 0.02,\ 0.01,\ 0.01]
```

(capital share, discount factor, TFP persistence, depreciation, TFP
innovation s.d., observation-noise s.d.). The function below is a
readable port of
[`DifferentiablePerturbation.jl`'s code-generated
`RBC.first_order_assembly!`](https://github.com/HighDimensionalEconLab/DifferentiablePerturbation.jl/blob/main/src/models/RBC_generated/first_order_ip.jl)
with each matrix entry written in natural model variables. It is pure
Julia; ForwardDiff `Dual` and Enzyme `Duplicated` perturbations of `p`
flow through it into `klein_map`.

The five rows of the linearisation are: 1. Euler equation, 2. capital
budget, 3. production, 4. TFP AR(1), 5. investment identity. The
variable ordering is ``z = [k, z_\text{proc}, c, y, i]`` (capital, TFP,
consumption, output, investment).

```jldoctest klein_rbc
julia> using LinearAlgebra: eigvals

julia> using MatrixEquationsAD

julia> function rbc_first_order_assembly(p)
           α, β, ρ, δ, _σ, _Ω_1 = p          # σ, Ω_1 enter only the shock loading
           # Deterministic SS from the Euler condition
           #   α · k_ss^(α-1) = 1/β - 1 + δ.
           rk   = (1 / β - 1 + δ) / α        # ≡ k_ss^(α-1)
           k_ss = rk^(1 / (α - 1))
           y_ss = k_ss^α                     # production at SS
           c_ss = y_ss - δ * k_ss            # consumption  c_ss = y_ss - δ·k_ss
           mpk  = α * k_ss^(α - 1)           # marginal product of capital at SS
           # Variable ordering of z_t:
           k_col, z_col, c_col, y_col, i_col = 1, 2, 3, 4, 5
           T = promote_type(typeof(α), typeof(β), typeof(δ), typeof(k_ss))
           A = zeros(T, 5, 5)
           B = zeros(T, 5, 5)
           # Row 1 — Euler equation
           #   1/c_t = β · E_t[(1/c_{t+1}) · (α·e^{z_{t+1}}·k_{t+1}^(α-1) + 1 - δ)].
           # Linearise; β·(mpk + 1 - δ) = 1 at SS collapses the c_{t+1} coefficient to 1/c_ss².
           A[1, k_col] = -β * (α - 1) * mpk / k_ss / c_ss     # ∂/∂k_{t+1}
           A[1, z_col] = -β * mpk / c_ss                       # ∂/∂z_{t+1}
           A[1, c_col] = inv(c_ss^2)                            # ∂/∂c_{t+1}
           B[1, c_col] = -inv(c_ss^2)                           # ∂/∂c_t
           # Row 2 — capital budget,  k_{t+1} = (1-δ)·k_t + y_t - c_t.
           A[2, k_col] = one(T)
           B[2, k_col] = -(one(T) - δ)
           B[2, c_col] = one(T)
           B[2, y_col] = -one(T)
           # Row 3 — production (linearised),  y_t = mpk·k_t + y_ss·z_t.
           B[3, k_col] = -mpk
           B[3, z_col] = -y_ss
           B[3, y_col] = one(T)
           # Row 4 — TFP AR(1),  z_{t+1} = ρ·z_t.
           A[4, z_col] = one(T)
           B[4, z_col] = -ρ
           # Row 5 — investment identity,  i_t = k_{t+1} - (1-δ)·k_t.
           A[5, k_col] = -one(T)
           B[5, k_col] = one(T) - δ
           B[5, i_col] = one(T)
           return A, B, 2                       # n_x = 2 predetermined states (k, z)
       end;

julia> p = [0.5, 0.95, 0.2, 0.02, 0.01, 0.01];

julia> A, B, n_x = rbc_first_order_assembly(p);

julia> r = klein_map(A, B; threshold = 1.0e-6);

julia> size(r.g_x), size(r.h_x)
((3, 2), (2, 2))

julia> r.h_x[1, 1] ≈ 0.9568351489231556 && r.h_x[2, 2] ≈ 0.2
true
```

`r.g_x` is the static policy: each row gives one jump variable
(consumption, output, investment) as a linear function of the
predetermined state ``(k, z)``. `r.h_x` is the state transition, and
its spectrum sits strictly inside the unit disc:

```jldoctest klein_rbc
julia> maximum(abs, eigvals(r.h_x)) < 1
true
```

To differentiate any summary of the policy with respect to the model
parameters, compose the assembly with `klein_map` and hand the closure
to
[`DifferentiationInterface.jl`](https://github.com/JuliaDiff/DifferentiationInterface.jl).
The backend (ForwardDiff, Enzyme reverse, etc.) is just an argument:

```jldoctest klein_rbc; setup = :(using ForwardDiff; using DifferentiationInterface: AutoForwardDiff, gradient)
julia> function policy_summary(p)
           A, B, _ = rbc_first_order_assembly(p)
           r = klein_map(A, B; threshold = 1.0e-6)
           return sum(r.g_x) + sum(r.h_x)
       end;

julia> ∇p = gradient(policy_summary, AutoForwardDiff(), p);

julia> (∇p[5], ∇p[6])                       # σ and Ω_1 don't enter (F)
(0.0, 0.0)
```

Swapping to Enzyme reverse mode is a one-line change:

```julia
using Enzyme
using DifferentiationInterface: AutoEnzyme
∇p = gradient(policy_summary, AutoEnzyme(mode = Enzyme.Reverse), p)
```

Or call Enzyme directly without the DifferentiationInterface wrapper:

```julia
using Enzyme: Active, Const, Duplicated, Reverse, autodiff, make_zero

p̄ = make_zero(p)
autodiff(Reverse, policy_summary, Active, Duplicated(copy(p), p̄))
# p̄ now holds the gradient.
```

Both backends agree with ForwardDiff to floating-point round-off; the
test suite exercises this directly.

## Jacobian of the implicit equation

Let ``E_y = [\,0_{n_x \times n_y};\, I_{n_y}\,] \in \mathbb{R}^{n \times n_y}``,
so the trailing-``n_y``-column projections of ``A`` and ``B`` are

```math
N \;=\; A\,E_y,
\qquad
P \;=\; B\,E_y,
\qquad
M \;=\; A\,\Psi.
```

Differentiating ``F = A\,\Psi\,h_x + B\,\Psi = 0`` at fixed ``(A, B)`` gives

```math
M\,d h_x
\;+\;
\bigl(N\, d g_x\, h_x + P\, d g_x\bigr)
\;=\;
-\,d A\,\Psi\, h_x \;-\; d B\,\Psi.
\tag{T}
```

Vectorising column-major and stacking the unknowns as
``[\operatorname{vec}(d h_x);\,\operatorname{vec}(d g_x)] \in \mathbb{R}^{n n_x}``,

```math
K
\begin{bmatrix}
\operatorname{vec}(d h_x) \\
\operatorname{vec}(d g_x)
\end{bmatrix}
\;=\;
-\operatorname{vec}\!\bigl(d A\,\Psi\,h_x + d B\,\Psi\bigr),
\tag{K}
```

with

```math
K
\;=\;
\Bigl[\,
I_{n_x}\otimes M
\;\Big|\;
h_x^\top\otimes N \;+\; I_{n_x}\otimes P
\,\Bigr]
\;\in\;\mathbb{R}^{n n_x \times n n_x}.
```

Two equivalent linear-algebra representations of (K) are implemented
and produce identical tangents up to floating-point roundoff. Both are
correct on degenerate-cluster pencils (e.g. the FVGQ fixture) because
they differentiate the implicit equation rather than the non-unique QZ
factors.

## Big-K factorisation (OOP `klein_map`)

The out-of-place rule builds ``K`` once as a dense
``(n n_x) \times (n n_x)`` block matrix and stores its LU factorisation
in the plan returned by `_klein_bigk_plan`. Both forward and reverse
modes reuse this single factorisation.

### JVP (ForwardDiff / Enzyme forward)

For each tangent direction ``(d A, d B)``,

```math
\begin{bmatrix}
\operatorname{vec}(d h_x) \\
\operatorname{vec}(d g_x)
\end{bmatrix}
\;=\;
K^{-1}\,\operatorname{vec}\!\bigl(-(d A\,\Psi\,h_x + d B\,\Psi)\bigr).
```

`_klein_bigk_jvp` forms the right-hand side
``R = -(d A\,\Psi\,h_x + d B\,\Psi)``, performs one LU
back-substitution, and reshapes the solution into ``(d h_x, d g_x)``.
ForwardDiff chunks of width ``N`` and Enzyme `BatchDuplicated` of width
``N`` both issue ``N`` such back-substitutions against the shared
factorisation.

### VJP (Enzyme reverse)

Pack the output cotangents in the same block order as ``K`` and solve
the transposed system:

```math
u
=
\begin{bmatrix}
\operatorname{vec}(\bar h_x) \\
\operatorname{vec}(\bar g_x)
\end{bmatrix},
\qquad
\lambda \;=\; K^{-\top} u,
\qquad
\Lambda \;=\; \operatorname{reshape}(\lambda,\, n,\, n_x).
```

Because the residual ``R`` depends on ``A`` and ``B`` only through
``R = -(d A\,\Psi\,h_x + d B\,\Psi)``, the parameter cotangents are

```math
\bar A
\;\mathrel{+}=\;
-\,\Lambda\,h_x^\top\,\Psi^\top,
\qquad
\bar B
\;\mathrel{+}=\;
-\,\Lambda\,\Psi^\top.
```

`_klein_bigk_vjp` performs the single transposed LU solve, reshapes
``\lambda``, and applies these outer products.

## Reduced-Sylvester factorisation (in-place `klein_map!`)

The in-place rule never materialises the full ``K``. It instead builds
the ``n \times n`` block

```math
C_0
=
\begin{bmatrix}
M & P
\end{bmatrix}
=
\begin{bmatrix}
A\,\Psi & B\,E_y
\end{bmatrix}
\;\in\;\mathbb{R}^{n \times n},
```

LU-factorises ``C_0``, solves once for the auxiliary matrix
``J = C_0^{-1} N \in \mathbb{R}^{n \times n_y}``, and splits it into

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
``J_y = Q_y S_y Q_y^\top`` and ``h_x = Q_h S_h Q_h^\top`` are computed
once and reused.

### JVP

Rewrite the tangent equation (T) by collecting ``(d h_x, d g_x)``
in the first ``n`` columns and isolating the cross-term:

```math
C_0
\begin{bmatrix} d h_x \\ d g_x \end{bmatrix}
\;+\; N\, d g_x\, h_x
\;=\;
R,
\qquad
R \;=\; -(d A\,\Psi\,h_x + d B\,\Psi).
```

Vectorising recovers the big-K system (K); premultiplying by ``C_0^{-1}``
and using ``J = C_0^{-1} N`` eliminates ``d h_x`` and yields a discrete
Sylvester / Stein equation in ``d g_x`` alone. Concretely, let
``Y = C_0^{-1} R`` and split ``Y = [Y_x;\,Y_y]`` (matching the
``n_x + n_y = n`` block structure). The block rows give the discrete
Sylvester / Stein equation

```math
d g_x \;+\; J_y\,d g_x\,h_x \;=\; Y_y.
```

In the Schur frame ``\tilde X = Q_y^\top d g_x\, Q_h`` it becomes

```math
S_y\,\tilde X\,S_h \;+\; \tilde X \;=\; Q_y^\top Y_y Q_h,
```

solved in place by `MatrixEquations.sylvds!`, whose convention is
``S_y\,X\,S_h + X = \text{RHS}`` (or the transposed form when
`adjA = adjB = true`). Back-transforming and
substituting into the ``h_x`` block,

```math
d h_x \;=\; Y_x \;-\; J_x\,d g_x\,h_x.
```

For ForwardDiff chunks of width ``N`` or Enzyme `BatchDuplicated` of
width ``N``, each lane reuses the precomputed ``C_0`` LU and the two
Schur factorisations ``J_y = Q_y S_y Q_y^\top``,
``h_x = Q_h S_h Q_h^\top``; only the ``C_0`` back-substitutions, the
``Q_y / Q_h`` rotations, and one triangular `sylvds!` sweep are
repeated per lane.

### VJP

Given output cotangents ``(\bar g_x, \bar h_x)``, the elimination above
flows backwards. The ``d h_x`` formula contributes to ``\bar Y_x`` and
induces a correction on the ``d g_x`` cotangent:

```math
\bar Y_x \;=\; \bar h_x,
\qquad
\tilde{\bar g}_x
\;=\;
\bar g_x \;-\; J_x^\top\,\bar h_x\,h_x^\top.
```

The corrected ``\tilde{\bar g}_x`` is the right-hand side of the adjoint
Stein equation

```math
J_y^\top\,Z\,h_x^\top \;+\; Z \;=\; \tilde{\bar g}_x,
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
\Lambda \;=\; C_0^{-\top}\,\bar Y,
```

requires one transposed LU back-substitution against the same ``C_0``
factorisation. The parameter cotangents are then identical to the big-K
form:

```math
\bar A
\;\mathrel{+}=\;
-\,\Lambda\,h_x^\top\,\Psi^\top,
\qquad
\bar B
\;\mathrel{+}=\;
-\,\Lambda\,\Psi^\top.
```

The ``C_0`` LU and the two Schur factorisations are computed once in
the augmented primal and stashed on Enzyme's tape; the reverse pass
performs one adjoint `sylvds!` plus one ``C_0^{-\top}`` solve and never
re-Schurs or re-LU-factorises.

## Which path wins

| Pencil size | Big-K (OOP) | Reduced Sylvester (in-place) |
| --- | --- | --- |
| Small (n ≲ 15) | Faster — LU is cheap, single multi-RHS solve. | Setup overhead (two Schurs + one LU) dominates. |
| Large (n ≳ 30) | LU of ``(n n_x)^2`` becomes expensive. | Wins; only ``n_y \times n_y`` Schur + per-tangent `sylvds!`. |

At the SW07PFEIFER pencil (``n = 42``, ``n_x = 20``) the in-place
reverse runs roughly five times faster than the OOP reverse. At the RBC
pencil (``n = 5``) the OOP path wins by ~50%.

## SMatrix dispatch

`StaticArrays` support requires explicit `Val(n_x)` so the return type
is statically sized:

```julia
using StaticArrays
As = SMatrix{5, 5, Float64}(A);  Bs = SMatrix{5, 5, Float64}(B)
r = klein_map(As, Bs, Val(2); threshold = 1.0e-6)
@assert r.g_x isa SMatrix{3, 2, Float64}
@assert r.h_x isa SMatrix{2, 2, Float64}
```

Without the `Val`, an SMatrix input falls through to the heap dispatch
and returns plain `Matrix` outputs — type-stable code-gen requires the
BK split to be known at compile time, and a runtime BK count cannot
satisfy that.

## More AD examples

ForwardDiff Jacobian of the flattened policy against the flattened
linearisation:

```julia
using ForwardDiff, MatrixEquationsAD

A, B, n_x = rbc_first_order_assembly([0.5, 0.95, 0.2, 0.02, 0.01, 0.01])
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
dA_lanes = ntuple(_ -> randn(rng, size(A)...), Val(4))
dB_lanes = ntuple(_ -> randn(rng, size(B)...), Val(4))

loss(A, B) = (r = klein_map(A, B; threshold = 1.0e-6); sum(r.g_x) + sum(r.h_x))

(out,) = Enzyme.autodiff(
    Enzyme.Forward, loss, Enzyme.BatchDuplicated,
    Enzyme.BatchDuplicated(copy(A), dA_lanes),
    Enzyme.BatchDuplicated(copy(B), dB_lanes),
)
# `out` is a NamedTuple of the four scalar JVPs, one per tangent direction.
```

In-place reverse on a `klein_map!` call:

```julia
n_y = size(A, 1) - n_x
g_x = zeros(n_y, n_x);  h_x = zeros(n_x, n_x)
A_bar = zero(A);  B_bar = zero(B)

Enzyme.autodiff(
    Enzyme.Reverse,
    (g, h, A, B) -> (klein_map!(g, h, A, B; threshold = 1.0e-6); sum(g) + sum(h)),
    Enzyme.Active,
    Enzyme.Duplicated(g_x, zero(g_x)),
    Enzyme.Duplicated(h_x, zero(h_x)),
    Enzyme.Duplicated(copy(A), A_bar),
    Enzyme.Duplicated(copy(B), B_bar),
)
```

The `klein_map!` rule uses the reduced-Sylvester path; the OOP
`klein_map` loss above routes through the big-K rule instead.

## Aside: why AD on (F) rather than the QZ factors

In principle one could compute ``(\bar A, \bar B)`` by differentiating
the generalised-Schur factors ``(S, T, Q, Z)`` and back-propagating
through the Klein algebra. The package does not, for four reasons:

1. The QZ factors admit a gauge freedom on repeated or clustered
   eigenvalues — any orthogonal transformation commuting with the
   block structure leaves ``(S, T)`` unchanged — so the tangent of
   ``(Q, Z)`` is ill-defined precisely where eigenvalues coalesce.
2. Real-Schur ``S`` mixes ``1 \times 1`` real blocks with
   ``2 \times 2`` complex-conjugate blocks; perturbations that swap
   block types are discontinuous, and the BK-reordering step has the
   same issue at threshold crossings.
3. Klein's ``(g_x, h_x)`` depends only on the stable deflating
   subspace, not on the orthogonal basis chosen for it.
   Differentiating ``F`` reflects that invariance: ``(d g_x, d h_x)``
   is uniquely defined even when ``(d Q, d Z)`` is not.
4. The implicit-function theorem applied to ``F`` only requires
   nonsingularity of the linearised operator ``K`` above, which holds
   generically; the QZ-tangent approach needs eigenvalue/eigenvector
   perturbation bounds that degrade as clusters tighten.

## References

- Klein, P. (2000). *Using the generalized Schur form to solve a
  multivariate linear rational expectations model.*
  [DOI:10.1016/S0165-1889(99)00045-7](https://doi.org/10.1016/S0165-1889(99)00045-7).
  Original derivation of the policy ``(g_x, h_x)`` from the QZ
  decomposition.
- Sims, C. (2001). *Solving linear rational expectations models.*
  [DOI:10.1023/A:1020517101123](https://doi.org/10.1023/A:1020517101123).
  Generalised-Schur method for linear rational-expectations models;
  motivates the BK ordering used in `klein_map`.
- Blanchard, O. and Kahn, C. (1980). *The solution of linear difference
  models under rational expectations.*
  [DOI:10.2307/1912186](https://doi.org/10.2307/1912186). Original
  saddle-path count condition (the number of explosive generalised
  eigenvalues must equal ``n_x``). The soft threshold
  ``|\alpha_i| \ge (1 - \tau)|\beta_i|`` is an implementation choice
  that approximates ``|\alpha_i| > |\beta_i|`` with numerical
  tolerance; it does not come from the 1980 paper.
- Kao, T.-T. and Hennequin, M. (2020). *Automatic differentiation of
  Sylvester, Lyapunov, and algebraic Riccati equations.*
  [arXiv:2011.11430](https://arxiv.org/abs/2011.11430). General
  recipe for differentiating implicit matrix-equation solvers via the
  IFT, which is the pattern (F) above realises for Klein's policy.
- Sun, J.-G. (1996). *Perturbation analysis of the generalized Schur
  decomposition.*
  [DOI:10.1137/S0895479892242189](https://doi.org/10.1137/S0895479892242189).
  Background on why QZ-factor tangents are ill-defined at clustered
  eigenvalues — context for the aside above.
- LAPACK
  [`DGGES`](https://www.netlib.org/lapack/explore-html/d7/d25/group__gges_ga556be4f39b39e5008c8eb36814aa7e20.html)
  documents the generalised real Schur factorisation and the
  eigenvalue-ordering interface used in `src/klein_map.jl`.
