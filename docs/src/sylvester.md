# Generalised Sylvester

Two `MatrixEquations.jl` solvers for the same equation:

- `gsylv(A, B, C, D, E)` — Schur-based, ``O(n^3)`` per solve.
- `gsylvkr(A, B, C, D, E)` — Kronecker LU on the
  ``n^2 \times n^2`` operator.

Both solve

```math
A\,X\,B \;+\; C\,X\,D \;=\; E.
\tag{S}
```

Define the linear operator

```math
G[X] \;=\; A\,X\,B \;+\; C\,X\,D,
```

so (S) reads ``G[X] = E``. `MatrixEquationsAD` provides ForwardDiff
`Dual` dispatches and Enzyme forward / reverse rules for both
front-ends; the rules share their tangent / adjoint derivations and
differ only in which factorisation is cached.

Implementation pointers:

- `ext/forwarddiff_sylvester.jl`, `ext/enzyme_sylvester.jl` — AD
  front-ends for `gsylv` and `gsylvkr`.

## Primal

`gsylv` uses the generalised-Schur (QZ) decompositions of
``(A, C)`` and ``(B, D)`` from
[`MatrixEquations.jl`](https://andreasvarga.github.io/MatrixEquations.jl/v2.4/sylvester.html#MatrixEquations.gsylv);
`gsylvkr` builds the Kronecker matrix
``B^\top \otimes A + D^\top \otimes C`` directly:

```math
\bigl(B^\top \otimes A + D^\top \otimes C\bigr)\,\operatorname{vec}(X)
\;=\;
\operatorname{vec}(E).
```

The factorisation assumption is that ``G`` is nonsingular —
equivalently, the operator pairs ``(A - \lambda C)`` and
``(D + \lambda B)`` are regular and share no common eigenvalue
(`MatrixEquations.jl` documents this requirement).

## Worked example

```jldoctest gsylv_small
julia> using MatrixEquations: gsylv, gsylvkr

julia> using MatrixEquationsAD

julia> A = [4.0 0.1 0.0; -0.2 3.6 0.3; 0.1 0.0 3.8]
3×3 Matrix{Float64}:
  4.0  0.1  0.0
 -0.2  3.6  0.3
  0.1  0.0  3.8

julia> B = [3.0 0.2; -0.1 2.7]
2×2 Matrix{Float64}:
  3.0  0.2
 -0.1  2.7

julia> C = [0.2 0.0 0.0; 0.0 0.2 0.0; 0.0 0.0 0.2]
3×3 Matrix{Float64}:
 0.2  0.0  0.0
 0.0  0.2  0.0
 0.0  0.0  0.2

julia> D = [0.3 0.0; 0.0 0.3]
2×2 Matrix{Float64}:
 0.3  0.0
 0.0  0.3

julia> E = [1.0 -0.4; 0.3 0.8; -0.2 0.5]
3×2 Matrix{Float64}:
  1.0  -0.4
  0.3   0.8
 -0.2   0.5

julia> X = gsylv(A, B, C, D, E)
3×2 Matrix{Float64}:
  0.0805978  -0.0446488
  0.0362012   0.072903
 -0.017917    0.050781

julia> isapprox(A * X * B + C * X * D - E, zeros(3, 2); atol = 1.0e-12)
true
```

`gsylvkr` returns the same solution up to round-off:

```jldoctest gsylv_small
julia> gsylvkr(A, B, C, D, E) ≈ X
true
```

## ForwardDiff / Enzyme JVP

Differentiating (S) gives

```math
G[d X]
\;=\;
d E
\;-\; d A\,X\,B \;-\; A\,X\,d B
\;-\; d C\,X\,D \;-\; C\,X\,d D.
```

`gsylv` caches its generalised-Schur factors of the pairs ``(A, C)``
and ``(B, D)`` on the value layer and reuses them for every
chunked-`Dual` partial direction or Enzyme `BatchDuplicated` tangent —
each lane is one triangular sweep against the shared Schur factors.
`gsylvkr` caches the LU factorisation of
``B^\top \otimes A + D^\top \otimes C``; chunked tangents are ``N`` LU
back-substitutions against that single factorisation.

## Enzyme VJP

Solve the adjoint generalised Sylvester equation

```math
G^*[Y]
\;=\;
A^\top\,Y\,B^\top \;+\; C^\top\,Y\,D^\top
\;=\;
\bar X.
```

Then

```math
\bar E \;\mathrel{+}=\; Y,
```

and

```math
\begin{aligned}
\bar A &\mathrel{-}= Y\,B^\top\,X^\top, \\
\bar B &\mathrel{-}= X^\top\,A^\top\,Y, \\
\bar C &\mathrel{-}= Y\,D^\top\,X^\top, \\
\bar D &\mathrel{-}= X^\top\,C^\top\,Y.
\end{aligned}
```

The two implementations share these formulas; only the cached
factorisation differs (generalised-Schur factors for `gsylv`,
Kronecker LU for `gsylvkr`). In both cases the augmented primal
stashes the cache on Enzyme's tape so multiple reverse cotangents
reuse it without refactorising.

## Differentiating through `gsylv`

```julia
using Enzyme: Active, Const, Duplicated, Reverse, autodiff, make_zero
using LinearAlgebra: dot
using MatrixEquations: gsylv
using MatrixEquationsAD

A = [4.0 0.1 0.0; -0.2 3.6 0.3; 0.1 0.0 3.8]
B = [3.0  0.2; -0.1 2.7]
C = [0.2 0.0 0.0; 0.0 0.2 0.0; 0.0 0.0 0.2]
D = [0.3 0.0; 0.0 0.3]
E = [1.0 -0.4; 0.3 0.8; -0.2 0.5]
W = [0.7 -0.1; -0.2 0.4; 0.5 0.3]

weighted(A, B, C, D, E, W) = dot(W, gsylv(A, B, C, D, E))

dA = make_zero(A); dB = make_zero(B)
dC = make_zero(C); dD = make_zero(D); dE = make_zero(E)
autodiff(
    Reverse, weighted, Active,
    Duplicated(A, dA), Duplicated(B, dB),
    Duplicated(C, dC), Duplicated(D, dD),
    Duplicated(E, dE), Const(W),
)
# dA, dB, dC, dD, dE now hold the gradients.
```

References:

- MatrixEquations.jl documents `gsylv` as solving ``A X B + C X D = E``
  in its
  [Sylvester solver documentation](https://andreasvarga.github.io/MatrixEquations.jl/v2.4/sylvester.html);
  `gsylvkr` is the Kronecker variant.
- LAPACK's
  [DTGSYL](https://www.netlib.org/lapack/explore-html/d4/d3b/group__tgsyl_ga96eff9d077e7600c68cd18246ca4cdc3.html)
  documents the generalised Sylvester systems used in Schur-based
  solvers and their transposed forms.
- Kao and Hennequin derive AD rules for Sylvester-type equations in
  [arXiv:2011.11430](https://arxiv.org/abs/2011.11430).
