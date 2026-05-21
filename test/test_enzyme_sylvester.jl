using Enzyme: BatchDuplicated, BatchDuplicatedNoNeed, Const, Duplicated, DuplicatedNoNeed
using EnzymeTestUtils: test_forward, test_reverse
using LinearAlgebra: I
using MatrixEquations
using Test

@testset "gsylv Enzyme rules" begin
    A = [4.0 0.1 0.0; -0.2 3.6 0.3; 0.1 0.0 3.8]
    B = [3.0 0.2; -0.1 2.7]
    C = Matrix(0.2I, 3, 3)
    D = Matrix(0.3I, 2, 2)
    E = [1.0 -0.4; 0.3 0.8; -0.2 0.5]

    test_forward(
        gsylv, DuplicatedNoNeed,
        (A, Duplicated), (B, Duplicated), (C, Duplicated), (D, Duplicated), (E, Duplicated)
    )
    test_forward(
        gsylv, BatchDuplicatedNoNeed,
        (A, BatchDuplicated), (B, BatchDuplicated), (C, BatchDuplicated),
        (D, BatchDuplicated), (E, BatchDuplicated)
    )
    test_reverse(
        gsylv, Duplicated,
        (A, Duplicated), (B, Duplicated), (C, Duplicated), (D, Duplicated), (E, Duplicated)
    )
    test_reverse(
        gsylv, Duplicated,
        (A, Const), (B, Duplicated), (C, Duplicated), (D, Duplicated), (E, Duplicated)
    )
end
