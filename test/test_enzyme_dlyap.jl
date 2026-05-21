using Enzyme:
    Active, BatchDuplicated, BatchDuplicatedNoNeed, Const, Duplicated, DuplicatedNoNeed
using EnzymeTestUtils: test_forward, test_reverse
using LinearAlgebra: Symmetric, dot
using MatrixEquations
using Test

function lyapd_weighted_sum(A, C, W)::Float64
    X = lyapd(A, C)
    return dot(W, X)
end

function lyapd_symmetric(A, C)
    return lyapd(A, Symmetric(C))
end

function lyapd_symmetric_weighted_sum(A, c, W)::Float64
    X = lyapd(A, Symmetric(unvech_symmetric(c, 2)))
    return dot(W, X)
end

function lyapd_symmetric_factor_weighted_sum(A, C_factor, W)::Float64
    C = C_factor' * C_factor
    X = lyapd(A, Symmetric(C))
    return dot(W, X)
end

@testset "lyapd Enzyme rules" begin
    A = [0.55 0.08; -0.04 0.42]
    C = [1.0 0.2; 0.2 0.7]
    c = vech_symmetric(C)
    W = [0.4 -0.1; -0.1 0.7]
    C_factor = [1.0 0.25; -0.35 0.9]

    test_forward(
        lyapd, DuplicatedNoNeed, (A, Duplicated), (C, Duplicated)
    )
    test_forward(
        lyapd, BatchDuplicatedNoNeed, (A, BatchDuplicated), (C, BatchDuplicated)
    )
    test_reverse(
        lyapd, Duplicated, (A, Duplicated), (C, Duplicated)
    )
    test_reverse(
        lyapd_weighted_sum, Active,
        (A, Duplicated), (C, Duplicated), (W, Const)
    )

    test_forward(
        lyapd_symmetric, DuplicatedNoNeed, (A, Duplicated), (C, Duplicated)
    )
    test_reverse(
        lyapd_symmetric_weighted_sum, Active,
        (A, Duplicated), (c, Duplicated), (W, Const)
    )
    test_reverse(
        lyapd_symmetric_factor_weighted_sum, Active,
        (A, Duplicated), (C_factor, Duplicated), (W, Const)
    )
end
