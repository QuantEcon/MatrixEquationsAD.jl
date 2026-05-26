# MatrixEquationsAD

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://QuantEcon.github.io/MatrixEquationsAD.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://QuantEcon.github.io/MatrixEquationsAD.jl/dev/)
[![Build Status](https://github.com/QuantEcon/MatrixEquationsAD.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/QuantEcon/MatrixEquationsAD.jl/actions/workflows/CI.yml?query=branch%3Amain)

Automatic differentiation rules for selected
[`MatrixEquations.jl`](https://github.com/andreasvarga/MatrixEquations.jl)
solvers, plus a Klein/Sims first-order policy-map primitive. AD rules
(ForwardDiff `Dual` dispatches and Enzyme forward / reverse) load
through package extensions when the relevant AD package is in scope.

See the [documentation](https://QuantEcon.github.io/MatrixEquationsAD.jl/dev/)
for derivations, worked examples, and the per-solver API:

- [Klein Policy Map](https://QuantEcon.github.io/MatrixEquationsAD.jl/dev/klein_map/) —
  first-order DSGE policy ``(g_x, h_x)``
- [Discrete Lyapunov (Schur)](https://QuantEcon.github.io/MatrixEquationsAD.jl/dev/lyapd/) —
  `lyapd(A, C)` solving ``A X A^\top - X + C = 0``
- [Kronecker Discrete Lyapunov](https://QuantEcon.github.io/MatrixEquationsAD.jl/dev/lyapdkr/) —
  `lyapdkr(A, C)` Kronecker LU variant
- [Generalised Sylvester](https://QuantEcon.github.io/MatrixEquationsAD.jl/dev/sylvester/) —
  `gsylv` / `gsylvkr` for ``A X B + C X D = E``
- [Order-2 Kronecker Sylvester (Kamenik)](https://QuantEcon.github.io/MatrixEquationsAD.jl/dev/sylvester_kamenik/) —
  `gsylv_kamenik` for ``A X + B X (C ⊗ C) = D`` (DSGE second-order
  perturbation form; Enzyme reverse only)
- [Algebraic Riccati (DARE)](https://QuantEcon.github.io/MatrixEquationsAD.jl/dev/ared/) —
  `ared(A, B, R, Q, S)` with stabilising gain `F`

## Install

```julia
using Pkg
Pkg.add(url = "https://github.com/QuantEcon/MatrixEquationsAD.jl")
```

## Example

Differentiate a small discrete Lyapunov solve with ForwardDiff:

```julia
using ForwardDiff
using MatrixEquations
using MatrixEquationsAD

A = [0.55 0.08; -0.04 0.42]
C = [1.0  0.2;  0.2 0.7]

lyapd_sum(x) =
    sum(lyapd(reshape(x[1:4], 2, 2), reshape(x[5:8], 2, 2)))

ForwardDiff.gradient(lyapd_sum, [vec(A); vec(C)])
```

The same loss with Enzyme reverse mode:

```julia
using Enzyme: Active, Const, Duplicated, Reverse, autodiff, make_zero
using LinearAlgebra: dot
using MatrixEquations
using MatrixEquationsAD

A = [0.55 0.08; -0.04 0.42]
C = [1.0  0.2;  0.2 0.7]
W = [0.4 -0.1; -0.1 0.7]

weighted(A, C, W) = dot(W, lyapd(A, C))

A_bar = make_zero(A); C_bar = make_zero(C)
autodiff(
    Reverse, weighted, Active,
    Duplicated(A, A_bar), Duplicated(C, C_bar), Const(W),
)
```
