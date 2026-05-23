# Algebraic Riccati (DARE)

`ared(A, B, R, Q, S)` solves the discrete algebraic Riccati equation

```math
A^\top X A - X
\;-\; (A^\top X B + S)\,(R + B^\top X B)^{-1}\,(B^\top X A + S^\top)
\;+\; Q
\;=\;
0
\tag{DARE}
```

via the generalised-eigenproblem method of Arnold and Laub
(`MatrixEquations.jl`'s
[Riccati solvers](https://andreasvarga.github.io/MatrixEquations.jl/dev/riccati.html)).
The four-argument call `ared(A, B, R, Q)` is the ``S = 0`` shorthand.
The solver returns the stabilising symmetric solution ``X`` and the
optimal-LQR gain

```math
F \;=\; G^{-1}(B^\top X A + S^\top),
\qquad
G \;=\; R + B^\top X B,
```

so the closed-loop dynamics are ``A_c = A - B F``.
`MatrixEquationsAD` differentiates both outputs (`X` and `F`) under
ForwardDiff and Enzyme.

Implementation pointers:

- `ext/enzyme_riccati.jl`, `ext/forwarddiff_riccati.jl` — AD frontends.
- `ext/riccati_derivatives.jl` — shared tangent / adjoint plan.

## Primal assumptions

The rule assumes the selected stabilising (or anti-stabilising) Riccati
branch is locally smooth, ``G = R + B^\top X B`` is nonsingular, and
the closed-loop Lyapunov operator
``L_{A_c}[\Delta X] = \Delta X - A_c^\top \Delta X A_c`` is
nonsingular. The usual sufficient conditions are stabilisability and
detectability, together with a well-conditioned positive-definite
``G``.

## Worked example

A 2-state, 1-input LQR:

```jldoctest ared_small
julia> using MatrixEquations: ared

julia> A = [0.95 0.0; 0.0 0.8]
2×2 Matrix{Float64}:
 0.95  0.0
 0.0   0.8

julia> B = reshape([1.0, 0.5], 2, 1)
2×1 Matrix{Float64}:
 1.0
 0.5

julia> R = reshape([0.1], 1, 1)
1×1 Matrix{Float64}:
 0.1

julia> Q = [1.0 0.0; 0.0 1.0]
2×2 Matrix{Float64}:
 1.0  0.0
 0.0  1.0

julia> X, _, F = ared(A, B, R, Q);

julia> X
2×2 Matrix{Float64}:
  1.62885   -0.937017
 -0.937017   2.61078

julia> F
1×2 Matrix{Float64}:
 0.763104  0.204009

julia> # Closed-loop Schur stability: ρ(A - B F) < 1
       using LinearAlgebra: eigvals

julia> maximum(abs, eigvals(A - B * F)) < 1
true
```

The primal residual of (DARE) is ``\approx 0``; the stabilising gain
`F` keeps the closed-loop spectrum strictly inside the unit disc.

## Example: stationary Kalman filter via DARE duality

The discrete-time linear filtering problem (Muth 1960; Ljungqvist &
Sargent, *Recursive Macroeconomic Theory*, chapter on optimal linear
filtering) is

```math
x_{t+1} \;=\; A\,x_t \;+\; C\,w_{t+1},
\qquad
y_t \;=\; G\,x_t \;+\; v_t,
```

with ``w_t \sim \mathcal{N}(0, I)`` orthogonal to ``v_t \sim \mathcal{N}(0, R)``.
The stationary one-step-ahead error covariance
``P = \lim_{t \to \infty} \mathbb{E}[(x_t - \hat x_{t|t-1})(x_t - \hat x_{t|t-1})^\top]``
satisfies the *filter* DARE

```math
P \;=\; A\,P\,A^\top \;-\; A\,P\,G^\top\,(G\,P\,G^\top + R)^{-1}\,G\,P\,A^\top \;+\; C\,C^\top,
\tag{FDARE}
```

and the stationary Kalman gain is

```math
K \;=\; A\,P\,G^\top\,(G\,P\,G^\top + R)^{-1}.
```

By the LQR ↔ Kalman duality, (FDARE) is the same equation as the
control (DARE) under the substitution
``(A_{\text{ctrl}}, B_{\text{ctrl}}, R_{\text{ctrl}}, Q_{\text{ctrl}})
= (A^\top, G^\top, R, C C^\top)``.
Calling `ared(A_filter', G_filter', R, C*C')` therefore returns
``X = P`` (stationary filter covariance) and ``F = K^\top`` (Kalman
gain, transposed). Both AD rules carry over verbatim.

### Scalar signal-extraction example

Take the canonical scalar AR(1) signal observed with noise:

```math
x_{t+1} \;=\; \rho\,x_t \;+\; w_{t+1},
\qquad
y_t \;=\; x_t \;+\; v_t,
\qquad
w_t \sim \mathcal{N}(0,\, \sigma_w^2),
\qquad
v_t \sim \mathcal{N}(0,\, \sigma_v^2).
```

In closed form ``P`` is the positive root of

```math
P^2 \;+\; P\bigl[\sigma_v^2(1 - \rho^2) - \sigma_w^2\bigr] \;-\; \sigma_w^2\,\sigma_v^2 \;=\; 0,
\qquad
K \;=\; \frac{\rho\,P}{P + \sigma_v^2}.
```

`ared` reproduces both:

```jldoctest kalman_scalar
julia> using MatrixEquations: ared

julia> ρ, σ_w, σ_v = 0.9, 0.5, 1.0;

julia> A = reshape([ρ], 1, 1); G = reshape([1.0], 1, 1);

julia> R = reshape([σ_v^2], 1, 1); Q = reshape([σ_w^2], 1, 1);

julia> X, _, F = ared(A', G', R, Q);

julia> P = X[1, 1]
0.5308991914547275

julia> K = F[1, 1]                       # equals ρ·P/(P + σ_v²)
0.31211021272747524

julia> s = σ_w^2 - σ_v^2 * (1 - ρ^2);

julia> P_closed = (s + sqrt(s^2 + 4 * σ_w^2 * σ_v^2)) / 2;

julia> isapprox(P, P_closed; atol = 1.0e-12)
true
```

### Differentiating the Kalman gain

The same closure differentiates end-to-end through `ared`. Build the
four matrices via `eltype(θ)` so ForwardDiff `Dual` partials flow into
every input:

```@example kalman_grad
ENV["GKSwstype"] = "100"   # GR headless backend for CI

using ForwardDiff
using MatrixEquations: ared
using MatrixEquationsAD

function kalman_gain(θ)
    ρ, σ_w, σ_v = θ
    T = eltype(θ)
    A = reshape(T[ρ],     1, 1)
    G = reshape(T[1.0],   1, 1)
    R = reshape(T[σ_v^2], 1, 1)
    Q = reshape(T[σ_w^2], 1, 1)
    _, _, F = ared(A, G, R, Q)        # uses A_ctrl = A_filter^⊤ = ρ (scalar)
    return F[1, 1]
end

θ₀ = [0.9, 0.5, 1.0]                  # (ρ, σ_w, σ_v)
K  = kalman_gain(θ₀)
∇K = ForwardDiff.gradient(kalman_gain, θ₀)
```

The three components ``\partial K/\partial \rho``,
``\partial K/\partial \sigma_w``, ``\partial K/\partial \sigma_v``
match the economic intuition:

```@example kalman_grad
∇K
```

`∂K/∂ρ > 0` (more persistence → the filter trusts the model more);
`∂K/∂σ_w > 0` (larger process noise → trust the data more);
`∂K/∂σ_v < 0` (noisier observations → trust the data less).

Sweeping ``\rho`` over ``[0.5, 0.99]`` at fixed ``\sigma_w, \sigma_v``
and overlaying the tangent ``K(\rho_0) + (\partial K / \partial \rho)(\rho - \rho_0)``:

```@example kalman_grad
using Plots

ρ_range = range(0.5, 0.99, length = 50)
K_curve = map(ρ_range) do ρ
    kalman_gain([ρ, θ₀[2], θ₀[3]])
end
tangent = K .+ ∇K[1] .* (ρ_range .- θ₀[1])

plot(ρ_range, K_curve;
    label = "K(ρ)", xlabel = "ρ", ylabel = "Kalman gain K",
    legend = :bottomright, linewidth = 2)
plot!(ρ_range, tangent;
    label = "tangent at ρ₀ = $(θ₀[1])", linestyle = :dash)
scatter!([θ₀[1]], [K]; label = "baseline", markersize = 5)
```

### Multivariate generalisation

The same code generalises to higher-dimensional signal–noise models —
e.g. the Muth permanent/transitory decomposition where the state is
``[\mu_t,\, \varepsilon_t]^\top`` with ``\mu`` a random walk,
``\varepsilon`` an AR(1), and a single noisy observation
``y_t = \mu_t + \varepsilon_t + v_t``. The state-space matrices are

```julia
A = [1.0  0.0;
     0.0  ρ]
G = [1.0  1.0]
C = [σ_ν  0.0;
     0.0  σ_ω]
R = reshape([σ_v^2], 1, 1)
```

and `ared(A', G', R, C*C')` returns the ``2 \times 2`` stationary error
covariance `P` and the ``1 \times 2`` Kalman gain ``K = F^\top``.
ForwardDiff and Enzyme reverse mode propagate through both outputs
unchanged.

## Differentials and AD rules

Let

```math
G \;=\; R + B^\top X B,
\qquad
F \;=\; G^{-1}(B^\top X A + S^\top),
\qquad
A_c \;=\; A - B\,F.
```

Differentiating (DARE) and using
``\Delta G = \Delta R + \Delta B^\top X B + B^\top \Delta X B + B^\top X \Delta B``,
``\Delta M = \Delta B^\top X A + B^\top \Delta X A + B^\top X \Delta A + \Delta S^\top``
gives a discrete Lyapunov equation for ``\Delta X`` against the
closed-loop ``A_c``:

```math
\Delta X \;-\; A_c^\top\,\Delta X\,A_c \;=\; P_n(H),
```

with

```math
\begin{aligned}
H \;=\;
&\;\Delta Q
\;+\; \Delta A^\top X A_c
\;+\; A_c^\top X \Delta A \\
&-\; A_c^\top X \Delta B\,F
\;-\; F^\top \Delta B^\top X A_c
\;+\; F^\top \Delta R\,F \\
&-\; \Delta S\,F
\;-\; F^\top \Delta S^\top,
\end{aligned}
```

where ``P_n(\cdot) = \tfrac{1}{2}(\cdot + \cdot^\top)`` symmetrises an
``n \times n`` matrix. For symmetric perturbations of ``Q`` and ``R``
and arbitrary cross-term perturbations of ``S``, ``P_n`` is a no-op on
the symmetric pieces and only matters in the asymmetric components.

After solving for ``\Delta X`` (one discrete Lyapunov solve against the
closed-loop ``A_c``), the gain tangent is

```math
\Delta F \;=\; G^{-1}\bigl(\Delta M - \Delta G\,F\bigr).
```

`MatrixEquationsAD` caches the closed-loop Schur factorisation
``A_c = Z S Z^\top`` and the Cholesky / LU of ``G`` on the value layer.
ForwardDiff chunks of width ``N`` and Enzyme `BatchDuplicated` of
width ``N`` reuse both caches: each lane is one triangular Lyapunov
sweep against the shared Schur factors plus one ``G^{-1}`` solve for
``\Delta F``.

### VJP

The reverse pass accepts cotangents for both differentiated outputs:
``\bar X`` and ``\bar F``. Propagate through ``F`` first:

```math
\Lambda \;=\; G^{-\top}\,\bar F,
\qquad
\Theta \;=\; P_m\bigl(-\Lambda\,F^\top\bigr),
```

with ``P_m`` symmetrising ``m \times m`` matrices. The direct
``\bar F``-contributions are

```math
\begin{aligned}
\bar A &\mathrel{+}= X^\top\,B\,\Lambda, \\
\bar B &\mathrel{+}= X\,A\,\Lambda^\top
        \;+\; X\,B\,\Theta^\top
        \;+\; X^\top\,B\,\Theta, \\
\bar R &\mathrel{+}= \Theta, \\
\bar S &\mathrel{+}= \Lambda^\top.
\end{aligned}
```

The cotangent passed to the closed-loop Lyapunov adjoint solve is

```math
\bar X_{\text{total}}
\;=\;
P_n\bigl(\bar X + B\,\Lambda\,A^\top + B\,\Theta\,B^\top\bigr).
```

Let ``Y`` solve the adjoint of the tangent Lyapunov operator,
``Y - A_c\,Y\,A_c^\top = \bar X_{\text{total}}``. Then add the
remaining adjoints:

```math
\begin{aligned}
\bar Q &\mathrel{+}= Y, \\
\bar A &\mathrel{+}= X\,A_c\,Y^\top + X^\top\,A_c\,Y, \\
\bar B &\mathrel{-}= X^\top\,A_c\,Y\,F^\top + X\,A_c\,Y^\top\,F^\top, \\
\bar R &\mathrel{+}= F\,Y\,F^\top, \\
\bar S &\mathrel{-}= Y\,F^\top + Y^\top\,F^\top.
\end{aligned}
```

Only `X` and `F` are differentiated. The `evals`, `Z`, and `scalinfo`
return values are returned with zero shadows. The default
four-argument method is handled by setting ``S = 0`` before
dispatching to the five-argument rule. The closed-loop Schur and the
``G`` factorisation are stashed on Enzyme's tape, so multiple reverse
cotangents (e.g. simultaneous ``\bar X`` and ``\bar F``) share one
factorisation pair.

References:

- MatrixEquations.jl documents `ared`, the stabilising gain ``F``, and
  (DARE) in its
  [Riccati solver documentation](https://andreasvarga.github.io/MatrixEquations.jl/dev/riccati.html).
- Arnold and Laub's generalised-eigenproblem method:
  [DOI:10.1109/PROC.1984.13083](https://doi.org/10.1109/PROC.1984.13083).
- Kao, T.-T. and Hennequin, M. (2020). *Automatic differentiation of
  Sylvester, Lyapunov, and algebraic Riccati equations.*
  [arXiv:2011.11430](https://arxiv.org/abs/2011.11430). The general
  implicit-function recipe used here is the one applied there to the
  *continuous* algebraic Riccati equation; the discrete-time formulas
  in this section are derived in-house against the closed-loop Lyapunov
  operator ``L_{A_c}[\cdot]``.
- Muth, J. F. (1960). *Optimal properties of exponentially weighted
  forecasts.* Journal of the American Statistical Association
  [DOI:10.1080/01621459.1960.10483352](https://doi.org/10.1080/01621459.1960.10483352).
  Original scalar signal-extraction problem solved by the Kalman
  filter; the canonical "stationary Kalman filter" example above.
- Ljungqvist, L. and Sargent, T. J. *Recursive Macroeconomic Theory*
  (4th ed., MIT Press, 2018). The optimal-linear-filtering chapter
  develops the same DARE-via-duality machinery and applies it to the
  permanent-income, signal-extraction, and innovations-representation
  examples cited above.
- Anderson, B. D. O. and Moore, J. B. *Optimal Filtering*
  (Prentice-Hall, 1979, repr. Dover 2005). Standard reference for the
  LQR ↔ Kalman duality
  ``(A_{\text{ctrl}}, B_{\text{ctrl}}, R_{\text{ctrl}}, Q_{\text{ctrl}})
  = (A_{\text{filt}}^\top, G_{\text{filt}}^\top, R, CC^\top)``.
