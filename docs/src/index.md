# MatrixEquationsAD.jl

Automatic differentiation rules for selected
[`MatrixEquations.jl`](https://github.com/andreasvarga/MatrixEquations.jl)
solvers, plus a Klein/Sims first-order policy-map primitive. The package
adds ForwardDiff `Dual` dispatches and Enzyme forward/reverse rules so
these solvers slot into AD-driven likelihood, calibration, and
gradient-based estimation pipelines.

```julia
using MatrixEquations    # primal solvers
using ForwardDiff
using Enzyme
using MatrixEquationsAD  # AD rules loaded on extension load
```

## Guide

- [Klein Policy Map](klein_map.md) — first-order policy ``(g_x, h_x)``
  from a DSGE linearisation, with JVP/VJP rules and a worked
  parameters-to-policy example.
- [Discrete Lyapunov (Schur)](lyapd.md) — `lyapd(A, C)` solving
  ``A X A^\top - X + C = 0`` via the Schur-based Bartels–Stewart kernel,
  with a `schur(A)` cache reused across AD directions.
- [Kronecker Discrete Lyapunov](lyapdkr.md) — `lyapdkr(A, C)` solving
  the same equation via the dense Kronecker LU and a symmetric
  projection of the output.
- [Generalised Sylvester](sylvester.md) — `gsylv(A, B, C, D, E)` and the
  Kronecker variant `gsylvkr` solving ``A X B + C X D = E``.
- [Algebraic Riccati (DARE)](ared.md) — `ared(A, B, R, Q, S)` solving
  the discrete algebraic Riccati equation and returning the stabilising
  gain `F` alongside `X`.

## Loaded extensions

The custom rules live in package extensions that load automatically
when the corresponding AD package is in scope:

| Extension | Trigger |
| --- | --- |
| `MatrixEquationsADForwardDiffExt` | `using ForwardDiff` |
| `MatrixEquationsADEnzymeExt` | `using Enzyme` |
| `MatrixEquationsADStaticArraysExt` | `using StaticArrays` |

The StaticArrays extension supplies `SMatrix` dispatches for `klein_map`
(requires explicit `Val(n_x)` for type-stable output sizing) and
`lyapdkr`.

## Conventions

All formulas use real matrices, the Frobenius inner product
``\langle U, V \rangle = \operatorname{tr}(U^\top V)``, and reverse-mode
cotangents written as barred variables such as ``\bar X``. Selection
decisions, integer outputs, and solver branch choices are treated as
locally constant.

State-space and filtering pages follow the
[QuantEcon Julia](https://julia.quantecon.org/) convention
``x_{t+1} = A\,x_t + w_{t+1}`` with ``w_t \sim \mathcal{N}(0, Q)`` and
``y_t = G\,x_t + v_t`` with ``v_t \sim \mathcal{N}(0, R)`` (see
[Linear State Space Models](https://julia.quantecon.org/introduction_dynamics/linear_models.html)
and [Kalman Filter](https://julia.quantecon.org/introduction_dynamics/kalman.html)).

All linear solves below assume the linearised operator is nonsingular;
for the discrete Lyapunov equation, ``\rho(A) < 1`` is sufficient and
the general uniqueness condition is that no pair of eigenvalues of
``A`` has product equal to one.

A short worked example differentiating `lyapd` through ForwardDiff:

```jldoctest index_quick
julia> using ForwardDiff, MatrixEquations, MatrixEquationsAD

julia> A = [0.55 0.08; -0.04 0.42];

julia> C = [1.0 0.2; 0.2 0.7];

julia> lyapd_sum(x) = sum(lyapd(reshape(x[1:4], 2, 2), reshape(x[5:8], 2, 2)));

julia> ForwardDiff.gradient(lyapd_sum, [vec(A); vec(C)])
8-element Vector{Float64}:
 2.374085538237574
 2.358267233900593
 1.4826474207121074
 1.4729046657398892
 1.3520319868662718
 1.3430117838358997
 1.3430117838359
 1.3342683300020852
```
