# MatrixEquationsAD.jl

Automatic differentiation rules for selected
[`MatrixEquations.jl`](https://github.com/andreasvarga/MatrixEquations.jl)
solvers plus a Klein/Sims first-order policy-map primitive. The package
supplies custom ForwardDiff `Dual` dispatches and Enzyme forward/reverse
rules so the solvers integrate cleanly into AD-driven likelihood,
calibration, and gradient-based estimation workflows.

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
  with a Schur cache reused across all AD directions.
- [Kronecker Discrete Lyapunov](lyapdkr.md) — `lyapdkr(A, C)` solving
  the same equation via the dense Kronecker LU factorisation and
  symmetric projection of the output.
- [Generalised Sylvester](sylvester.md) — `gsylv(A, B, C, D, E)` and the
  Kronecker variant `gsylvkr` solving ``A X B + C X D = E``.
- [Algebraic Riccati (DARE)](ared.md) — `ared(A, B, R, Q, S)` solving
  the discrete algebraic Riccati equation and returning the stabilising
  gain `F` alongside `X`.

## Loaded extensions

The custom rules live in package extensions and load automatically when
the corresponding AD package is in scope:

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
locally constant. All linear solves below assume the corresponding
linearised operator is nonsingular; for discrete Lyapunov, a sufficient
condition is ``\rho(A) < 1``, while the general uniqueness condition is
that no pair of eigenvalues of ``A`` has product equal to one.

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
