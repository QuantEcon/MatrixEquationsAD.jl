using Enzyme:
    Active, BatchDuplicated, BatchDuplicatedNoNeed, Const, Duplicated, DuplicatedNoNeed
using EnzymeTestUtils: test_forward, test_reverse
using LinearAlgebra: dot, issymmetric
using MatrixEquations
using MatrixEquationsAD
using StaticArrays: SMatrix
using Test

function lyapdkr_weighted_sum(A, C, W)::Float64
    X = lyapdkr(A, C)
    return dot(W, X)
end

@testset "lyapdkr Enzyme rules" begin
    A = [0.55 0.08; -0.04 0.42]
    C = [1.0 0.2; 0.2 0.7]
    W = [0.3 -0.1; 0.2 0.5]
    X = lyapdkr(A, C)
    As = SMatrix{2, 2, Float64}(A)
    Cs = SMatrix{2, 2, Float64}(C)
    Ws = SMatrix{2, 2, Float64}(W)

    @test X ≈ lyapd(A, C)
    @test issymmetric(X)

    test_forward(
        lyapdkr, DuplicatedNoNeed, (A, Duplicated), (C, Duplicated)
    )
    test_forward(
        lyapdkr, BatchDuplicatedNoNeed, (A, BatchDuplicated), (C, BatchDuplicated)
    )
    test_reverse(
        lyapdkr, Duplicated, (A, Duplicated), (C, Duplicated)
    )
    test_reverse(
        lyapdkr_weighted_sum, Active,
        (A, Duplicated), (C, Duplicated), (W, Const)
    )

    test_forward(
        lyapdkr_weighted_sum, BatchDuplicated,
        (As, BatchDuplicated), (Cs, BatchDuplicated), (Ws, Const)
    )
end
