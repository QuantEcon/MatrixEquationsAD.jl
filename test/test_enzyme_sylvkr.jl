using Enzyme: BatchDuplicated, BatchDuplicatedNoNeed, Const, Duplicated, DuplicatedNoNeed
using EnzymeTestUtils: test_forward, test_reverse
using MatrixEquations
using Test

@testset "gsylvkr Enzyme rules" begin
    A = [4.0 0.2 -0.1; -0.3 3.7 0.4; 0.1 -0.2 3.5]
    B = [2.8 -0.3; 0.2 3.1]
    C = [0.5 0.1 -0.2; 0.0 0.7 0.3; -0.1 0.2 0.6]
    D = [0.9 0.2; -0.4 0.8]
    E = [1.0 -0.4; 0.3 0.8; -0.2 0.5]

    @test gsylvkr(A, B, C, D, E) ≈ gsylv(A, B, C, D, E)

    test_forward(
        gsylvkr, DuplicatedNoNeed,
        (A, Duplicated), (B, Duplicated), (C, Duplicated), (D, Duplicated), (E, Duplicated)
    )
    test_forward(
        gsylvkr, BatchDuplicatedNoNeed,
        (A, BatchDuplicated), (B, BatchDuplicated), (C, BatchDuplicated),
        (D, BatchDuplicated), (E, BatchDuplicated)
    )
    test_reverse(
        gsylvkr, Duplicated,
        (A, Duplicated), (B, Duplicated), (C, Duplicated), (D, Duplicated), (E, Duplicated)
    )
    test_reverse(
        gsylvkr, Duplicated,
        (A, Const), (B, Duplicated), (C, Duplicated), (D, Duplicated), (E, Duplicated)
    )
end
