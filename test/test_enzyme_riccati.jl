using Enzyme: Active, BatchDuplicated, Const, Duplicated
using EnzymeTestUtils: test_forward, test_reverse
using FiniteDifferences: central_fdm
using LinearAlgebra: I, dot, issymmetric
using MatrixEquations
using MatrixEquationsAD
using Random: MersenneTwister
using Test

function ared_weighted_sum(A, B, r, q, WX, WF)::Float64
    R = unvech_symmetric(r, 2)
    Q = unvech_symmetric(q, 2)
    X, _, F = ared(A, B, R, Q)
    return dot(WX, X) + dot(WF, F)
end

@testset "ared Enzyme rules" begin
    A = Matrix([0.95 0.0; 0.0 0.95]')
    B = Matrix([0.5 0.0; 0.0 0.5]')
    R = Matrix(0.2I, 2, 2)
    Q = Matrix(0.5I, 2, 2)
    X, _, F = ared(A, B, R, Q)
    @test issymmetric(X)

    r = vech_symmetric(R)
    q = vech_symmetric(Q)
    WX = [0.4 -0.2; -0.2 0.8]
    WF = [0.1 -0.3; 0.5 0.2]

    test_forward(
        ared_weighted_sum, BatchDuplicated,
        (A, BatchDuplicated), (B, BatchDuplicated),
        (r, BatchDuplicated), (q, BatchDuplicated), (WX, Const), (WF, Const);
        rng = MersenneTwister(3344), fdm = central_fdm(5, 1; max_range = 1.0e-4),
    )
    test_reverse(
        ared_weighted_sum, Active,
        (A, Duplicated), (B, Duplicated),
        (r, Duplicated), (q, Duplicated), (WX, Const), (WF, Const);
        rng = MersenneTwister(4433),
    )
end
