module RBCExampleMatrices

export dp_rbc_first_order_inputs, dp_rbc_sv_first_order_inputs

# RBC / RBC_SV first-order assembly + klein solution bundles captured from
# DifferentiablePerturbation.jl canonical parameter vectors. Each function
# returns (; A_schur, B_schur, B_shock, g_x, h_x, n_x) so downstream tests can
# pair klein_map (A_schur, B_schur → g_x, h_x) with lyapd (h_x, B_shock).
#
# RBC_P    = [0.5, 0.95, 0.2, 0.02, 0.01, 0.01]                 (DP test/first_order_perturbation.jl:7)
# RBC_SV_P = vcat(RBC_P, [0.9, -4.0, 0.1])                       (DP test/first_order_perturbation.jl:8)

# FO assembly + klein solution bundle for downstream matrix-equation tests.
# klein_map:  (A_schur, B_schur) → (g_x, h_x).
# lyapd:      lyapd(h_x, B_shock * transpose(B_shock)).
function dp_rbc_first_order_inputs()
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

# FO assembly + klein solution bundle for downstream matrix-equation tests.
# klein_map:  (A_schur, B_schur) → (g_x, h_x).
# lyapd:      lyapd(h_x, B_shock * transpose(B_shock)).
function dp_rbc_sv_first_order_inputs()
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
