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

`MatrixEquationsAD` caches the closed-loop Schur factors so all chunked
tangents reuse a single factorisation.

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
dispatching to the five-argument rule.

References:

- MatrixEquations.jl documents `ared`, the stabilising gain ``F``, and
  (DARE) in its
  [Riccati solver documentation](https://andreasvarga.github.io/MatrixEquations.jl/dev/riccati.html).
- Arnold and Laub's generalised-eigenproblem method:
  [DOI:10.1109/PROC.1984.13083](https://doi.org/10.1109/PROC.1984.13083).
- Kao and Hennequin derive AD rules for algebraic Riccati equations in
  [arXiv:2011.11430](https://arxiv.org/abs/2011.11430).
