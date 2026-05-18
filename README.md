# MatrixEquationsAD

[![Build Status](https://github.com/QuantEcon/MatrixEquationsAD.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/QuantEcon/MatrixEquationsAD.jl/actions/workflows/CI.yml?query=branch%3Amain)

Automatic differentiation support for selected
[`MatrixEquations.jl`](https://github.com/andreasvarga/MatrixEquations.jl)
functions, plus ordered-QZ helpers while those APIs are not available upstream.

```julia
using MatrixEquations
using ForwardDiff
using Enzyme
using MatrixEquationsAD
```

Differentiate through a small discrete Lyapunov solve:

```julia
using MatrixEquations
using ForwardDiff
using MatrixEquationsAD

A = [0.55 0.08; -0.04 0.42]
C = [1.0 0.2; 0.2 0.7]

f(a) = sum(lyapd(reshape(a, 2, 2), C))
ForwardDiff.gradient(f, vec(A))
```

Use the exported ordered-QZ wrapper:

```julia
using MatrixEquationsAD

A = [1.6 0.2 0.1; 0.0 0.35 -0.1; 0.0 0.0 1.9]
B = [1.0 0.1 0.0; 0.0 1.2 0.2; 0.0 0.0 0.8]

F = ordqz(A, B, qzselect_inside_unit)
F.S, F.T, F.Q, F.Z
```
