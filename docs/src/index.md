# MatrixEquationsAD.jl

Automatic differentiation rules for selected
[`MatrixEquations.jl`](https://github.com/andreasvarga/MatrixEquations.jl)
solvers and a Klein/Sims policy-function map for first-order DSGE pencils.
The package supplies custom ForwardDiff `Dual` dispatches and Enzyme
forward/reverse rules so the solvers integrate cleanly into AD-driven
likelihood, calibration, and gradient-based estimation workflows.

```julia
using MatrixEquations    # primal solvers
using ForwardDiff
using Enzyme
using MatrixEquationsAD  # AD rules loaded on extension load
```

## Guide

- [Klein Policy Map](klein_map.md) — implicit policy equation, JVP, VJP.
- [Kronecker Discrete Lyapunov](lyapdkr.md) — Kronecker-vec discrete
  Lyapunov solve and symmetric JVP/VJP.

The remaining AD rules — for `lyapd`, `gsylv`, `gsylvkr`, `ared` — follow
the standard implicit-function-theorem pattern documented in
[`DERIVATIONS.md`](https://github.com/QuantEcon/MatrixEquationsAD.jl/blob/main/DERIVATIONS.md).

## Quick start: parameters → pencil → policy

The bundled example is a standard RBC model. Its first-order pencil
``(A, B)`` is assembled from the parameter vector

```math
p = [\alpha,\ \beta,\ \rho,\ \delta,\ \sigma,\ \Omega_1]
  = [0.5,\ 0.95,\ 0.2,\ 0.02,\ 0.01,\ 0.01]
```

(capital share, discount factor, TFP persistence, depreciation, TFP
innovation s.d., observation-noise s.d.) by
`RBCExampleMatrices.rbc_first_order_assembly`, defined in
[`test/example_matrices/rbc.jl`](https://github.com/QuantEcon/MatrixEquationsAD.jl/blob/main/test/example_matrices/rbc.jl).
That function is a readable port of
[`DifferentiablePerturbation.jl`'s code-generated `RBC.first_order_assembly!`](https://github.com/HighDimensionalEconLab/DifferentiablePerturbation.jl/blob/main/src/models/RBC_generated/first_order_ip.jl)
with each matrix entry written in natural model variables; ForwardDiff
`Dual` and Enzyme `Duplicated` perturbations of `p` flow through it into
`klein_map`.

```jldoctest rbc_quick
julia> using LinearAlgebra: eigvals

julia> using MatrixEquationsAD

julia> include(joinpath(pkgdir(MatrixEquationsAD), "test", "example_matrices", "rbc.jl"));

julia> using .RBCExampleMatrices: rbc_first_order_assembly

julia> p = [0.5, 0.95, 0.2, 0.02, 0.01, 0.01];

julia> A, B, n_x = rbc_first_order_assembly(p);

julia> r = klein_map(A, B; threshold = 1.0e-6);

julia> size(r.g_x), size(r.h_x)
((3, 2), (2, 2))
```

`r.g_x` is the static policy: each row gives one jump variable
(consumption, output, investment) as a linear function of the
predetermined state `(k, z)`. `r.h_x` is the state transition, and its
spectrum sits strictly inside the unit disc by construction:

```jldoctest rbc_quick
julia> maximum(abs, eigvals(r.h_x)) < 1
true
```

To differentiate any summary of the policy with respect to the model
parameters, compose the assembly with `klein_map` and hand the closure to
[`DifferentiationInterface.jl`](https://github.com/JuliaDiff/DifferentiationInterface.jl).
The backend (ForwardDiff, Enzyme reverse, etc.) is just an argument:

```jldoctest rbc_quick; setup = :(using ForwardDiff; using DifferentiationInterface: AutoForwardDiff, gradient)
julia> function policy_summary(p)
           A, B, _ = rbc_first_order_assembly(p)
           r = klein_map(A, B; threshold = 1.0e-6)
           return sum(r.g_x) + sum(r.h_x)
       end;

julia> ∇p = gradient(policy_summary, AutoForwardDiff(), p);

julia> (∇p[5], ∇p[6])                              # σ and Ω_1 don't enter the pencil
(0.0, 0.0)
```

Swapping to Enzyme reverse mode is a one-line change:

```julia
using Enzyme
using DifferentiationInterface: AutoEnzyme
∇p = gradient(policy_summary, AutoEnzyme(mode = Enzyme.Reverse), p)
```

Both backends return the same gradient to floating-point round-off; the
test suite exercises this directly.

The other DSGE pencils used by the test/benchmark suites have analogous
precomputed fixture bundles but no parametric assembly. Each one returns a
`(; A_schur, B_schur, B_shock, g_x, h_x, n_x)` `NamedTuple` so the same
destructuring works downstream:
`RBCExampleMatrices.dp_rbc_sv_first_order_inputs` (RBC with stochastic
volatility, ``n = 8``), `SGUExampleMatrices.dp_sgu_first_order_inputs`
(``n = 15``), `FVGQExampleMatrices.dp_fvgq_first_order_inputs`
(``n = 38``), and `SW07ExampleMatrices.dp_sw07pfeifer_first_order_inputs`
(``n = 42``).

Differentiate the policy with Enzyme reverse mode:

```julia
using Enzyme

loss(A, B) = (r = klein_map(A, B); sum(r.g_x) + sum(r.h_x))

A_bar = zero(A); B_bar = zero(B)
Enzyme.autodiff(
    Reverse, loss, Active,
    Duplicated(copy(A), A_bar),
    Duplicated(copy(B), B_bar),
)
```

`A_bar` and `B_bar` now contain ``\bar A`` and ``\bar B`` as defined in
[Klein Policy Map](klein_map.md#enzyme-vjp).

## Loaded extensions

The custom rules live in package extensions and load automatically when the
corresponding AD package is in scope:

| Extension | Trigger |
| --- | --- |
| `MatrixEquationsADForwardDiffExt` | `using ForwardDiff` |
| `MatrixEquationsADEnzymeExt` | `using Enzyme` |
| `MatrixEquationsADStaticArraysExt` | `using StaticArrays` |

The StaticArrays extension supplies `SMatrix` dispatches for `klein_map`
(requires explicit `Val(n_x)` for type-stable output sizing) and `lyapdkr`.
