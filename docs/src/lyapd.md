# Discrete Lyapunov (Schur)

`lyapd(A, C)` solves the discrete Lyapunov equation

```math
A X A^{\!\top} - X + C = 0
```

using the Schur-based Bartels–Stewart algorithm from
[`MatrixEquations.jl`](https://github.com/andreasvarga/MatrixEquations.jl):
the upstream solver Schur-factorises `A` once and runs a triangular sweep
(`MatrixEquations.lyapds!`) on the transformed right-hand side. For
non-toy state dimensions this is the right default — the cost is
``O(n^3)`` per solve, vs. ``O(n^6)`` for the [Kronecker-vec
form](lyapdkr.md). `MatrixEquationsAD` provides custom AD rules that wrap
that solver in a cache-aware shadow so a single `schur(A)` is reused
across all tangent / cotangent directions.

Implementation pointers:

- `ext/enzyme_lyapunov.jl` — `LyapDSchurCache`, `lyapdfactor`,
  `lyapdsolve`, `lyapdadjointsolve`, the cache-aware `lyapd(A, C)` shadow,
  and the Enzyme forward / reverse rules.
- `ext/forwarddiff_lyapunov.jl` — the ForwardDiff `Dual` dispatch on
  `lyapd(A, C)`, which uses the same cache primitives.
- `MatrixEquations.lyapds!` — upstream in-place kernel that operates on
  `A` in real or complex Schur form (`adj = false / true` for transpose
  variants).

`MatrixEquationsAD` also exports `lyapd!(X, A, C)`, which writes the
solution into a caller-supplied `X` and returns `nothing`. The in-place
form shares the same Schur cache plumbing and carries an analogous full
set of AD rules — ForwardDiff `Dual` dispatch, Enzyme forward
(Duplicated / BatchDuplicated), and Enzyme reverse (augmented_primal +
reverse). The reverse rule stashes the Schur factors on the tape so the
adjoint pass never re-Schurs `A`. A cache-taking overload
`lyapd!(X, cache::LyapDSchurCache, C)` lets the AD rules share one
`schur(A)` across all forward tangent directions or reverse cotangents.

## Primal

The cache-aware shadow precomputes ``\mathrm{schur}(A) = (T, Z)``,
``A = Z\,T\,Z^{\!\top}``, and stores it in a `LyapDSchurCache`. With that
cache available, the solve transforms the right-hand side, runs the
upstream Schur-form kernel, and untransforms:

```math
\tilde C \;=\; Z^{\!\top}\,C\,Z,
\qquad
X \;=\; Z\,\tilde X\,Z^{\!\top},
```

where ``\tilde X`` satisfies the triangular Sylvester / Lyapunov

```math
T\,\tilde X\,T^{\!\top} \;-\; \tilde X \;+\; \tilde C \;=\; 0.
```

For ``\texttt{C::Symmetric}`` this is exactly one call to
`lyapds!(cache.T, rhs)`; for general dense ``C`` the dispatch routes to
`sylvds!(-cache.T, cache.T, rhs; adjB = true)`, which solves the same
triangular system without enforcing symmetry.

Existence and uniqueness require that no two eigenvalues of ``A``
multiply to one. ``\rho(A) < 1`` is sufficient and is the case of
interest for stationary-covariance applications below.

## ForwardDiff JVP

For one tangent direction ``(dA, dC)``, differentiating
``A X A^{\!\top} - X + C = 0`` and applying the same operator gives

```math
A\,dX\,A^{\!\top} \;-\; dX
\;=\;
-\,dC \;-\; dA\,X\,A^{\!\top} \;-\; A\,X\,dA^{\!\top},
```

i.e. ``dX`` solves another discrete Lyapunov equation against the same
``A``. The ForwardDiff dispatch builds the cache once and calls
`lyapdsolve(cache, rhs_i)` for each partial direction in the chunk —
``N`` `lyapds!` triangular sweeps against the shared Schur factors.

The Enzyme `BatchDuplicated` forward rule is structurally identical: one
`schur(A)` per outer call, then one triangular solve per tangent.

## Enzyme VJP

Let ``\bar X`` be the cotangent on the output. Define ``Y`` by the
adjoint discrete Lyapunov solve

```math
A^{\!\top}\,Y\,A \;-\; Y \;=\; -\,\bar X,
```

implemented as `lyapdadjointsolve(cache, X̄)` (which routes to
`lyapds!(cache.T, rhs; adj = true)` for `Symmetric` cotangents and to
the transposed `sylvds!` variant otherwise). The parameter cotangents
are then

```math
\bar C \;\mathrel{+}=\; Y,
\qquad
\bar A \;\mathrel{+}=\; Y\,A\,X^{\!\top} \;+\; Y^{\!\top}\,A\,X.
```

The augmented primal copies the cached Schur and the primal ``X`` onto
the tape; the reverse pass performs one triangular adjoint solve plus
the two outer products.

When the input ``C`` is wrapped as `Symmetric`, the reverse rule
projects the incoming cotangent onto the symmetric manifold (replacing
``\bar C`` by ``\tfrac{1}{2}(\bar C + \bar C^{\!\top})``) before the
adjoint solve. This matches the primal symmetry contract: a `Symmetric`
parameter is differentiated only against the symmetric perturbations
that lie in its parameter space.

## Example: RBC stationary covariance, end-to-end

The bundled RBC fixture `dp_rbc_first_order_inputs()` returns a
`NamedTuple` `(; A_schur, B_schur, B_shock, g_x, h_x, n_x)`. Pairing
its `h_x` with ``Q = B_{\text{shock}}\,B_{\text{shock}}^{\!\top}`` and
solving ``V = h_x\, V\, h_x^{\!\top} + Q`` gives the stationary state
covariance under the Klein/Sims policy:

```jldoctest
julia> using LinearAlgebra: Symmetric

julia> using MatrixEquations: lyapd

julia> using MatrixEquationsAD

julia> include(joinpath(pkgdir(MatrixEquationsAD), "test", "example_matrices", "rbc.jl"));

julia> (; h_x, B_shock) = RBCExampleMatrices.dp_rbc_first_order_inputs();

julia> Q = Symmetric(B_shock * transpose(B_shock));

julia> V = lyapd(h_x, Q);

julia> V[2, 2] ≈ 0.01^2 / (1 - 0.2^2)               # matches AR(1) closed form
true

julia> round(sqrt(V[2, 2]); sigdigits = 4)          # TFP stationary std. (~1%)
0.01021

julia> round(sqrt(V[1, 1]); sigdigits = 3)          # capital stationary std.
0.265
```

Wrapping `Q` as `Symmetric` routes the AD rules onto the
`lyapds!`-based path; passing a plain `Matrix` would dispatch onto the
`sylvds!`-based path with the same result. Either way `lyapd` is the
right call here — its cost is ``O(n^3)`` in the state dimension, and
`MatrixEquationsAD` adds a single `schur(A)` cache reused across all AD
directions on top.

Differentiating any summary of ``V`` with respect to the RBC parameters
``p`` works straight through the `rbc_first_order_assembly → klein_map →
lyapd` pipeline with
[DifferentiationInterface](index.md#quick-start-parameters-pencil-policy);
the test suite exercises both ForwardDiff and Enzyme reverse against the
same closure.
