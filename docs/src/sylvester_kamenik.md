# Compact Order-2 Kronecker Sylvester (Kamenik)

A specialised solver for the order-2 Kronecker Sylvester equation

```math
A\,X \;+\; B\,X\,(C \otimes C) \;=\; D,
\tag{KS}
```

with ``A, B \in \mathbb{R}^{n \times n}``, ``C \in \mathbb{R}^{m \times m}``,
``D, X \in \mathbb{R}^{n \times m^2}``. This system arises in the
second-order perturbation step of nonlinear DSGE models around a
saddle-path steady state (Schmitt-Grohé and Uribe, 2004; Andreasen,
Fernández-Villaverde, Rubio-Ramírez, 2018), where ``C`` is the
first-order state transition matrix (``h_x``).

**Scope:** this page documents the allocating
`MatrixEquationsAD.gsylv_kamenik` and the in-place
`MatrixEquationsAD.gsylv_kamenik!`. Both forms ship with an Enzyme
forward rule (Width = 1 and `BatchDuplicated`) and an Enzyme reverse
rule (Width = 1). Both are **order = 2 only** (one Kronecker factor
squared). There is no ForwardDiff `Dual` dispatch and no `order > 2`
implementation.

The algorithm is a compact rewrite of Kamenik (2005), departing from
the
[DynareJulia/GeneralizedSylvesterSolver.jl](https://github.com/DynareJulia/GeneralizedSylvesterSolver.jl)
port: we use the `MatrixEquations.jl` primitives `sylvds!` (the
discrete Sylvester solve on real-Schur factors) and `gsylv` directly,
avoiding the `QuasiTriangularMatrices.jl` / `KroneckerTools.jl`
dependency stack and the ~13-function recursive decomposition that
the original port carries.

Implementation pointers:

- `src/sylvester_kamenik.jl` — primal solver (allocating
  `gsylv_kamenik`, in-place `gsylv_kamenik!`, and a private
  `_gsylv_kamenik_factor` / `_gsylv_kamenik_solve!` split that lets the
  Enzyme forward rule amortise one factorisation across the primal +
  every tangent solve).
- `ext/enzyme_sylvester_kamenik.jl` — Enzyme forward (Width 1 and
  `BatchDuplicated`) and Enzyme reverse (Width = 1) rules for both
  forms.

## Primal algorithm

Define ``K := C \otimes C`` and the linear operator
``\mathcal{L}[X] := A\,X + B\,X\,K``, so (KS) reads
``\mathcal{L}[X] = D``.

The solver proceeds in three stages:

1. **Preprocess.** Compute ``\tilde B := A^{-1} B`` and
   ``\tilde D_0 := A^{-1} D`` via a single LU factorisation of ``A``.
2. **Double real Schur.** Take ``\tilde B = U_B\,T\,U_B^\top`` and
   ``C = U_C\,S\,U_C^\top`` with ``T`` and ``S``
   quasi-upper-triangular. In Schur coordinates the equation collapses
   to

   ```math
   \tilde X \;+\; T\,\tilde X\,(S \otimes S) \;=\; \tilde D,
   ```

   where ``\tilde X := U_B^\top X (U_C \otimes U_C)`` and
   ``\tilde D := U_B^\top \tilde D_0 (U_C \otimes U_C)``.
3. **Column-by-column back-substitution.** View
   ``\tilde X, \tilde D`` as ``(n, m, m)`` tensors with
   ``\tilde X[:, p, o]`` the column ``(o-1)\,m + p`` (column-major
   Kronecker convention). The equation in tensor form is

   ```math
   \tilde X[:, p, o] \;+\; \sum_{o',\,p'} S[o', o]\,S[p', p]\,T\,\tilde X[:, p', o']
   \;=\; \tilde D[:, p, o].
   ```

   Since ``S`` is quasi-upper-triangular, ``S[o', o] = 0`` for
   ``o' > o`` except at the subdiagonal of a 2×2 complex Schur block.
   Iterate ``o = 1, 2, \ldots, m`` forward:

   - **1×1 real block** (``S[o+1, o] = 0`` or ``o = m``): solve

     ```math
     \tilde X[:, :, o] \;+\; (S[o, o]\,T)\,\tilde X[:, :, o]\,S \;=\; \mathrm{RHS}_o
     ```

     for ``\tilde X[:, :, o]`` via `MatrixEquations.sylvds!`, with
     ``\mathrm{RHS}_o`` already in real-Schur form for both
     ``S[o, o]\,T`` and ``S``.
   - **2×2 complex block** at ``(o, o+1)``: stack
     ``Y = [\tilde X[:, :, o]; \tilde X[:, :, o+1]]`` (``2n \times m``)
     and solve a 2-equation coupled system via `MatrixEquations.gsylv`.

   Finally, the inverse Schur transform is
   ``X = U_B\,\tilde X\,(U_C \otimes U_C)^\top``.

Both ``(U_C \otimes U_C)`` and its transpose are applied without
materialising the ``m^2 \times m^2`` Kronecker product: each is one
``n m \times m`` GEMM plus a per-slice loop of ``m`` small GEMMs
against ``U_C`` (column-major reshape).

## Worked example

```julia
using LinearAlgebra: I, kron, norm
using MatrixEquationsAD: gsylv_kamenik

# Toy 4×3 example.
A = [4.0  0.1  0.0  0.0; -0.2  3.6  0.3  0.1;  0.1  0.0  3.8  0.0;  0.0 -0.1  0.0  3.4]
B = 0.1 .* Matrix(I, 4, 4)
C = [0.5  0.1 -0.05;  0.0  0.6  0.1; -0.05  0.0  0.4]
D = reshape(collect(1.0:36.0), 4, 9)

X = gsylv_kamenik(A, B, C, D)
norm(A * X + B * X * kron(C, C) - D) / norm(D)  # ≈ 1e-15
```

## Enzyme JVP (forward)

Differentiating (KS) gives, for tangent inputs
``(\mathrm{d}A, \mathrm{d}B, \mathrm{d}C, \mathrm{d}D)``,

```math
A\,\mathrm{d}X \;+\; B\,\mathrm{d}X\,K
\;=\; \mathrm{d}D
 \;-\; \mathrm{d}A\,X
 \;-\; \mathrm{d}B\,(X\,K)
 \;-\; B\,X\,(\mathrm{d}C \otimes C + C \otimes \mathrm{d}C),
```

via the Kronecker product rule
``\mathrm{d}(C \otimes C) = (\mathrm{d}C \otimes C) + (C \otimes \mathrm{d}C)``.
The JVP equation in ``\mathrm{d}X`` has the **same** coefficient triple
``(A, B, C)`` as the primal — it is again a Kamenik order-2 system, so
the Enzyme forward rule reuses the primal factorisation across all
tangent solves. The terms ``B\,X\,(\mathrm{d}C \otimes C)`` and
``B\,X\,(C \otimes \mathrm{d}C)`` are computed via the two-pass GEMM
trick used by the primal — no ``m^2 \times m^2`` Kronecker is ever
formed. Pre-computed ``B\,X`` and ``X\,K`` are reused across tangents.

**Worked example — `BatchDuplicated(4)`** computes four tangent
directions in one Enzyme call, amortising the factorisation:

```julia
using Enzyme: BatchDuplicated, Const, Forward, autodiff
using MatrixEquationsAD: gsylv_kamenik

# (A, B, C, D) from the example above.
A = [4.0  0.1  0.0  0.0; -0.2  3.6  0.3  0.1;  0.1  0.0  3.8  0.0;  0.0 -0.1  0.0  3.4]
B = 0.1 .* Matrix(1.0I, 4, 4)
C = [0.5  0.1 -0.05;  0.0  0.6  0.1; -0.05  0.0  0.4]
D = reshape(collect(1.0:36.0), 4, 9)

dAs = ntuple(_ -> randn(size(A)...), Val(4))
dBs = ntuple(_ -> randn(size(B)...), Val(4))
dCs = ntuple(_ -> randn(size(C)...), Val(4))
dDs = ntuple(_ -> randn(size(D)...), Val(4))

dXs = autodiff(
    Forward, gsylv_kamenik, BatchDuplicated,
    BatchDuplicated(A, dAs), BatchDuplicated(B, dBs),
    BatchDuplicated(C, dCs), BatchDuplicated(D, dDs),
)[1]
# `dXs[k]` is the JVP in the k-th direction.
```

The in-place form `gsylv_kamenik!` exposes the same forward rule. The
input buffer (`D`) is overwritten with the primal solution, and each
shadow slot in `D.dval` is overwritten with the corresponding tangent
solution:

```julia
Dwork = copy(D)
dD_io = map(copy, dDs)
autodiff(
    Forward, gsylv_kamenik!, Const,
    BatchDuplicated(Dwork, dD_io),
    BatchDuplicated(A, dAs), BatchDuplicated(B, dBs),
    BatchDuplicated(C, dCs),
)
# `Dwork` now holds X, `dD_io[k]` holds the k-th tangent dX.
```

## Enzyme VJP (reverse)

Take cotangent ``\bar X`` on the solution ``X`` and use the Frobenius
pairing ``\langle U, V \rangle = \operatorname{tr}(U^\top V)``.

**Step 1: adjoint Sylvester.** Differentiating (KS) and pairing with
``\Lambda`` against the linear operator ``\mathcal{L}`` gives the
adjoint equation

```math
A^\top\,\Lambda \;+\; B^\top\,\Lambda\,(C^\top \otimes C^\top) \;=\; \bar X.
```

This is structurally identical to the primal (KS) with
``(A, B, C, D) \mapsto (A^\top, B^\top, C^\top, \bar X)`` — solve via
a second call to `gsylv_kamenik` on the transposed arguments.

**Step 2: parameter cotangents.** The simple ones follow directly
from the matrix product rule:

```math
\bar D \mathrel{+}= \Lambda, \qquad
\bar A \mathrel{-}= \Lambda\,X^\top, \qquad
\bar B \mathrel{-}= \Lambda\,(X\,K)^\top.
```

**Step 3: pullback through the Kronecker leg.** The cotangent
``\bar K`` of ``K = C \otimes C`` is

```math
\bar K \;=\; -\,X^\top\,B^\top\,\Lambda
\;\in\; \mathbb{R}^{m^2 \times m^2}.
```

The product rule ``\mathrm{d}(C \otimes C) = (\mathrm{d}C \otimes C) + (C \otimes \mathrm{d}C)``
turns this into a sum of two index contractions. Using the
column-major Kronecker convention
``(C \otimes C)_{(i-1)m+k,\,(j-1)m+l} = C_{i,j}\,C_{k,l}``,

```math
\bar C_{i,j}
\;=\;
\sum_{k,l} \bar K_{(i-1)m+k,\,(j-1)m+l}\,C_{k,l}
\;+\;
\sum_{k,l} \bar K_{(k-1)m+i,\,(l-1)m+j}\,C_{k,l}.
```

The two sums correspond to the two legs of the Kronecker product.
The implementation accumulates these as ``\bar C_{ij} \mathrel{-}= \ldots``
(absorbing the minus sign from ``\bar K = -X^\top B^\top \Lambda``
into the accumulator).

**Step 4: in-place form.** `gsylv_kamenik!` overwrites the input
buffer `D` with the solution `X`. The Enzyme reverse rule accordingly
treats the `Duplicated(D, D̄)` slot as carrying the cotangent of the
post-call value (the solution `X̄`) on entry, and overwrites it with
``\Lambda`` — the cotangent of the pre-call value (the original RHS).
The four parameter pullbacks are otherwise identical to the
allocating form.

### Code path

`ext/enzyme_sylvester_kamenik.jl` implements both rules. The augmented
primal caches ``X``, ``A``, ``B``, ``C`` on the tape (``D`` is not
needed in the reverse formulae). The reverse pass calls
`gsylv_kamenik` once on the transposed inputs to obtain ``\Lambda``,
then accumulates the four pullbacks. Single-cotangent only
(`EnzymeRules.width(config) == 1`) — explicitly asserted.

## Differentiating through `gsylv_kamenik`

```julia
using Enzyme: Active, Const, Duplicated, Reverse, autodiff, make_zero
using LinearAlgebra: dot
using MatrixEquationsAD: gsylv_kamenik

A = [4.0  0.1  0.0  0.0; -0.2  3.6  0.3  0.1;  0.1  0.0  3.8  0.0;  0.0 -0.1  0.0  3.4]
B = 0.1 .* [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
C = [0.5  0.1 -0.05;  0.0  0.6  0.1; -0.05  0.0  0.4]
D = reshape(collect(1.0:36.0), 4, 9)
W = randn(size(D)...)

weighted(A, B, C, D, W) = dot(W, gsylv_kamenik(A, B, C, D))

dA = make_zero(A); dB = make_zero(B); dC = make_zero(C); dD = make_zero(D)
autodiff(
    Reverse, weighted, Active,
    Duplicated(A, dA), Duplicated(B, dB),
    Duplicated(C, dC), Duplicated(D, dD),
    Const(W),
)
# dA, dB, dC, dD now hold the gradients of the scalar loss.
```

References:

- Kamenik, O., *Solving SDGE Models: A New Algorithm for the
  Sylvester Equation*, Computational Economics 25 (2005), 167–187.
- Schmitt-Grohé, S. and Uribe, M., *Solving dynamic general
  equilibrium models using a second-order approximation to the policy
  function*, Journal of Economic Dynamics and Control 28 (2004),
  755–775.
- Andreasen, M., Fernández-Villaverde, J. and Rubio-Ramírez, J.,
  *The Pruned State-Space System for Non-Linear DSGE Models: Theory
  and Empirical Applications*, Review of Economic Studies 85 (2018),
  1–49.
- [DynareJulia/GeneralizedSylvesterSolver.jl](https://github.com/DynareJulia/GeneralizedSylvesterSolver.jl)
  — the original Julia port of Kamenik (2005), which uses
  `QuasiTriangularMatrices.jl` and `KroneckerTools.jl` internally.
- Kao and Hennequin derive AD rules for Sylvester-type equations in
  [arXiv:2011.11430](https://arxiv.org/abs/2011.11430); the present
  derivation specialises to the Kronecker-squared right factor.
