# MatrixEquationsAD

[![Build Status](https://github.com/QuantEcon/MatrixEquationsAD.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/QuantEcon/MatrixEquationsAD.jl/actions/workflows/CI.yml?query=branch%3Amain)

Automatic differentiation support for selected
[`MatrixEquations.jl`](https://github.com/andreasvarga/MatrixEquations.jl)
functions, plus Klein policy-map extraction and AD for first-order DSGE gschur inputs.
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

Extract a Klein policy map from a first-order gschur input:

```julia
using LinearAlgebra: I, norm
using MatrixEquationsAD

A = [
    0.00012263591151906127 -0.011623494029190608 0.028377570562199094 0.0 0.0;
    1.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0;
    0.0 1.0 0.0 0.0 0.0;
    -1.0 0.0 0.0 0.0 0.0
]
B = [
    0.0 0.0 -0.028377570562199098 0.0 0.0;
    -0.98 0.0 1.0 -1.0 0.0;
    -0.07263157894736837 -6.884057971014498 0.0 1.0 0.0;
    0.0 -0.2 0.0 0.0 0.0;
    0.98 0.0 0.0 0.0 1.0
]

r = klein_map(A, B; threshold = 1.0e-6)
G = vcat(Matrix{Float64}(I, size(r.h_x, 1), size(r.h_x, 1)), r.g_x)
norm(A * G * r.h_x + B * G)
```
