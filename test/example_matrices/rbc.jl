module RBCExampleMatrices

export dp_rbc_first_order_gschur, dp_rbc_sv_first_order_gschur,
    rbc_first_order_assembly

# RBC first-order gschur inputs captured from DifferentiablePerturbation.jl
# canonical parameter vectors. Each function returns (A, B, n_x), where n_x is
# the number of predetermined states.
#
# RBC_P    = [0.5, 0.95, 0.2, 0.02, 0.01, 0.01]                 (DP test/first_order_perturbation.jl:7)
# RBC_SV_P = vcat(RBC_P, [0.9, -4.0, 0.1])                       (DP test/first_order_perturbation.jl:8)
#
# `rbc_first_order_assembly(p)` is a hand-written port of
# `DifferentiablePerturbation.jl`'s code-generated
# `RBC.first_order_assembly!` (src/models/RBC_generated/first_order_ip.jl)
# with the CSE-named locals replaced by their model meaning. It is purely
# functional and ForwardDiff-/Enzyme-compatible, so it provides a
# parameter-to-pencil entry point that AD can pass through into
# `klein_map`. The hard-coded `dp_rbc_first_order_gschur` returns the same
# matrices at `RBC_P` to bit-for-bit precision.

function dp_rbc_first_order_gschur()
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
    return A, B, 2
end

"""
    rbc_first_order_assembly(p) -> (A, B, n_x)

Assemble the first-order RBC pencil `(A, B)` and the predetermined-block
size `n_x = 2` at the parameter vector

    p = [α, β, ρ, δ, σ, Ω_1]

so that the linearised equilibrium satisfies `A·E_t[z_{t+1}] + B·z_t = 0`
with `z = [k, z_proc, c, y, i]` (capital, TFP, consumption, output,
investment). The five rows of the pencil are, in order:

  1. Euler equation,   2. capital budget,   3. production,
  4. TFP AR(1),        5. investment identity.

`σ` and `Ω_1` enter only the shock loading and observation noise, neither
of which is part of the pencil, so they do not show up below. The function
is pure Julia; ForwardDiff `Dual` and Enzyme `Duplicated` perturbations of
`p` flow through to `(A, B)`.

Translated from `DifferentiablePerturbation.jl`'s code-generated
`RBC.first_order_assembly!` (`src/models/RBC_generated/first_order_ip.jl`)
by undoing common-subexpression elimination and naming the locals.
"""
function rbc_first_order_assembly(p)
    α, β, _ρ, δ, _σ, _Ω_1 = p  # σ, Ω_1 unused in the pencil

    # Deterministic steady state from the Euler condition
    #   α · k_ss^(α-1) = 1/β - 1 + δ,
    # plus the resource constraint y_ss = k_ss^α, c_ss = y_ss - δ·k_ss.
    rk = (1 / β - 1 + δ) / α           # ≡ k_ss^(α-1)
    k_ss = rk^(1 / (α - 1))
    y_ss = k_ss^α
    c_ss = y_ss - δ * k_ss

    T = promote_type(typeof(α), typeof(β), typeof(δ), typeof(k_ss))
    A = zeros(T, 5, 5)
    B = zeros(T, 5, 5)

    # Row 1 — Euler equation, 1/c_t = β·E[(1/c_{t+1})·(α·e^{z_{t+1}}·k_{t+1}^(α-1) + 1 - δ)].
    # At SS, β·(α·k_ss^(α-1) + 1 - δ) = 1, so the c_{t+1} coefficient
    # collapses to 1/c_ss^2.
    A[1, 1] = -β * α * (α - 1) * k_ss^(α - 2) / c_ss   # ∂/∂k_{t+1}
    A[1, 2] = -β * α * k_ss^(α - 1) / c_ss             # ∂/∂z_{t+1}
    A[1, 3] = inv(c_ss^2)                                # ∂/∂c_{t+1}
    B[1, 3] = -inv(c_ss^2)                               # ∂/∂c_t

    # Row 2 — capital budget,  k_{t+1} = (1-δ)·k_t + y_t - c_t.
    A[2, 1] = one(T)
    B[2, 1] = -(one(T) - δ)
    B[2, 3] = one(T)
    B[2, 4] = -one(T)

    # Row 3 — production, y_t = e^{z_t}·k_t^α, linearised at SS gives
    #   y_t = α·k_ss^(α-1)·k_t + y_ss·z_t.
    B[3, 1] = -α * k_ss^(α - 1)
    B[3, 2] = -y_ss
    B[3, 4] = one(T)

    # Row 4 — TFP process, z_{t+1} = ρ·z_t.
    A[4, 2] = one(T)
    B[4, 2] = -p[3]  # ρ; named lookup keeps the eltype stable under Dual

    # Row 5 — investment identity, i_t = k_{t+1} - (1-δ)·k_t.
    A[5, 1] = -one(T)
    B[5, 1] = one(T) - δ
    B[5, 5] = one(T)

    return A, B, 2
end

function dp_rbc_sv_first_order_gschur()
    A = [
        0.00012263591151906127 0.0 0.0 0.0 0.028377570562199094 0.0 -0.011623494029190608 0.0;
        1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0;
        0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0;
        -1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    ]
    B = [
        0.0 0.0 0.0 0.0 -0.028377570562199098 0.0 0.0 0.0;
        -0.98 0.0 0.0 0.0 1.0 -1.0 0.0 0.0;
        -0.07263157894736837 0.0 0.0 0.0 0.0 1.0 -6.884057971014498 0.0;
        0.0 -0.2 0.0 -0.01831563888873418 0.0 0.0 1.0 0.0;
        0.0 0.0 -0.9 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 -1.0 0.0;
        0.98 0.0 0.0 0.0 0.0 0.0 0.0 1.0
    ]
    return A, B, 4
end

end
