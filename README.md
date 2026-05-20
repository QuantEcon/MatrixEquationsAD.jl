# MatrixEquationsAD

[![Build Status](https://github.com/QuantEcon/MatrixEquationsAD.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/QuantEcon/MatrixEquationsAD.jl/actions/workflows/CI.yml?query=branch%3Amain)

Automatic differentiation support for selected
[`MatrixEquations.jl`](https://github.com/andreasvarga/MatrixEquations.jl)
functions, plus ordered-QZ helpers while those APIs are not available upstream.
The implemented AD formulas are documented in [DERIVATIONS.md](DERIVATIONS.md).

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

function lyapd_sum(x)
    nA = length(A)
    A_dual = reshape(x[1:nA], size(A))
    C_dual = reshape(x[(nA + 1):end], size(C))
    return sum(lyapd(A_dual, C_dual))
end

ForwardDiff.gradient(lyapd_sum, [vec(A); vec(C)])
```

Reverse-mode differentiate a small generalized Sylvester solve:

```julia
using MatrixEquations
using Enzyme: Active, Const, Duplicated, Reverse, autodiff, make_zero
using LinearAlgebra: dot
using MatrixEquationsAD

A = [4.0 0.1 0.0; -0.2 3.6 0.3; 0.1 0.0 3.8]
B = [3.0 0.2; -0.1 2.7]
C = [0.2 0.0 0.0; 0.0 0.2 0.0; 0.0 0.0 0.2]
D = [0.3 0.0; 0.0 0.3]
E = [1.0 -0.4; 0.3 0.8; -0.2 0.5]
W = [0.7 -0.1; -0.2 0.4; 0.5 0.3]

function gsylv_weighted_sum(A, B, C, D, E, W)
    return dot(W, gsylv(A, B, C, D, E))
end

dA = make_zero(A)
dB = make_zero(B)
dC = make_zero(C)
dD = make_zero(D)
dE = make_zero(E)

autodiff(
    Reverse, gsylv_weighted_sum, Active,
    Duplicated(A, dA),
    Duplicated(B, dB),
    Duplicated(C, dC),
    Duplicated(D, dD),
    Duplicated(E, dE),
    Const(W),
)
```

Use the exported `ordqz` wrapper for a Blanchard-Kahn check. The package
builds on `LinearAlgebra.schur` + `ordschur!` under the hood — there is no
LAPACK-specific shim.

```julia
using MatrixEquationsAD

A = [1.6 0.2 0.1; 0.0 0.35 -0.1; 0.0 0.0 1.9]
B = [1.0 0.1 0.0; 0.0 1.2 0.2; 0.0 0.0 0.8]

eps_BK = 1.0e-6
n_unstable_expected = 2
(; S, T, Q, Z, sdim) = ordqz(A, B, :bk; threshold = eps_BK)
sdim == n_unstable_expected ||
    error("Blanchard-Kahn condition failed")

S, T, Q, Z
```
