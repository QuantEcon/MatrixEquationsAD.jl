module RBCExampleMatrices

export rbc_first_order_inputs, rbc_sv_first_order_inputs,
    rbc_first_order_assembly

# RBC / RBC_SV first-order assembly + klein solution bundles for the
# canonical parameter vectors below. Each function returns
# (; A_schur, B_schur, B_shock, g_x, h_x, n_x) so downstream tests can
# pair klein_map (A_schur, B_schur → g_x, h_x) with lyapd (h_x, B_shock).
#
# RBC_P    = [0.5, 0.95, 0.2, 0.02, 0.01, 0.01]
# RBC_SV_P = vcat(RBC_P, [0.9, -4.0, 0.1])
#
# `rbc_first_order_assembly(p)` derives the RBC pencil symbolically from
# the steady-state conditions. It is purely functional and
# ForwardDiff-/Enzyme-compatible, so it provides a parameter-to-pencil
# entry point that AD can pass through into `klein_map`. The hard-coded
# `rbc_first_order_inputs` returns the same matrices at `RBC_P` to
# bit-for-bit precision.

# FO assembly + klein solution bundle for downstream matrix-equation tests.
# klein_map:  (A_schur, B_schur) → (g_x, h_x).
# lyapd:      lyapd(h_x, B_shock * transpose(B_shock)).
function rbc_first_order_inputs()
    A_schur = [
        0.00012263591151906127 -0.011623494029190608 0.028377570562199094 0.0 0.0;
        1.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0;
        0.0 1.0 0.0 0.0 0.0;
        -1.0 0.0 0.0 0.0 0.0
    ]
    B_schur = [
        0.0 0.0 -0.028377570562199098 0.0 0.0;
        -0.98 0.0 1.0 -1.0 0.0;
        -0.07263157894736837 -6.884057971014498 0.0 1.0 0.0;
        0.0 -0.2 0.0 0.0 0.0;
        0.98 0.0 0.0 0.0 1.0
    ]
    B_shock = [
        0.0;
        -0.01;;
    ]
    g_x = [
        0.09579643002421286 0.6746869652588258;
        0.07263157894736855 6.884057971014499;
        -0.023164851076844135 6.209371005755667
    ]
    h_x = [
        0.9568351489231556 6.209371005755667;
        -3.3737787177631822e-18 0.20000000000000004
    ]
    return (; A_schur, B_schur, B_shock, g_x, h_x, n_x = 2)
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

# FO assembly + klein solution bundle for downstream matrix-equation tests.
# klein_map:  (A_schur, B_schur) → (g_x, h_x).
# lyapd:      lyapd(h_x, B_shock * transpose(B_shock)).
function rbc_sv_first_order_inputs()
    A_schur = [
        0.00012263591151906127 0.0 0.0 0.0 0.028377570562199094 0.0 -0.011623494029190608 0.0;
        1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0;
        0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0;
        -1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    ]
    B_schur = [
        0.0 0.0 0.0 0.0 -0.028377570562199098 0.0 0.0 0.0;
        -0.98 0.0 0.0 0.0 1.0 -1.0 0.0 0.0;
        -0.07263157894736837 0.0 0.0 0.0 0.0 1.0 -6.884057971014498 0.0;
        0.0 -0.2 0.0 -0.01831563888873418 0.0 0.0 1.0 0.0;
        0.0 0.0 -0.9 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 0.0 -1.0 0.0;
        0.98 0.0 0.0 0.0 0.0 0.0 0.0 1.0
    ]
    B_shock = [
        0.0 0.0;
        0.0 0.0;
        0.0 -0.1;
        -0.01 0.0
    ]
    g_x = [
        0.09579643002421243 0.1349373930517618 0.0 0.012357322818616293;
        0.07263157894736753 1.3768115942028987 0.0 0.1260859198862136;
        -1.3904557445170683e-16 0.19999999999999987 0.0 0.018315638888734175;
        -0.02316485107684383 1.241874201151137 -0.0 0.11372859706759733
    ]
    h_x = [
        0.9568351489231559 1.2418742011511372 0.0 0.11372859706759736;
        -8.425400481064518e-17 0.19999999999999973 0.0 0.01831563888873415;
        0.0 0.0 0.9 0.0;
        6.204228642701525e-19 8.052454488200187e-19 -0.0 7.373751485042179e-20
    ]
    return (; A_schur, B_schur, B_shock, g_x, h_x, n_x = 4)
end

end
