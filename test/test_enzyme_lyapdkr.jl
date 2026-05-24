using Enzyme:
    Active, BatchDuplicated, BatchDuplicatedNoNeed, Const, Duplicated, DuplicatedNoNeed
using EnzymeTestUtils: test_forward, test_reverse
using LinearAlgebra: I, dot, issymmetric
using MatrixEquations
using MatrixEquationsAD
using Random: Random
using StaticArrays: SMatrix
using Test

include(joinpath(@__DIR__, "example_matrices", "fvgq.jl"))

function lyapdkr_weighted_sum(A, C, W)::Float64
    X = lyapdkr(A, C)
    return dot(W, X)
end

function lyapdkr_weighted_sum_ws(A, C, W, M_ws)::Float64
    X = lyapdkr(A, C; M_ws)
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

@testset "lyapdkr Enzyme rules — M_ws workspace (FVGQ large)" begin
    Random.seed!(0x1d3f)
    fo = FVGQExampleMatrices.fvgq_first_order_inputs()
    A = fo.h_x
    B = fo.B_shock
    n = size(A, 1)
    C = B * B' + 1.0e-6 * I(n)
    W = randn(n, n)
    M_ws = Matrix{Float64}(undef, n * n, n * n)

    # FVGQ gradients are O(10^3-10^4); FD reference precision caps around
    # 1e-3 relative, so relax accordingly.
    fd_kwargs = (atol = 1.0e-2, rtol = 1.0e-3)

    test_forward(
        lyapdkr_weighted_sum_ws, Duplicated,
        (A, Duplicated), (C, Duplicated), (W, Const), (M_ws, Const);
        fd_kwargs...
    )
    test_forward(
        lyapdkr_weighted_sum_ws, BatchDuplicated,
        (A, BatchDuplicated), (C, BatchDuplicated), (W, Const), (M_ws, Const);
        fd_kwargs...
    )
    test_reverse(
        lyapdkr_weighted_sum_ws, Active,
        (A, Duplicated), (C, Duplicated), (W, Const), (M_ws, Const);
        fd_kwargs...
    )
end

function lyapdkr_inplace_weighted_sum(X, A, C, W)::Float64
    lyapdkr!(X, A, C)
    return dot(W, X)
end

@testset "lyapdkr! Enzyme rules" begin
    A = [0.55 0.08; -0.04 0.42]
    C = [1.0 0.2; 0.2 0.7]
    W = [0.3 -0.1; 0.2 0.5]

    X = similar(A)
    test_forward(
        lyapdkr!, Const,
        (X, Duplicated), (A, Duplicated), (C, Duplicated)
    )
    X = similar(A)
    test_forward(
        lyapdkr!, Const,
        (X, BatchDuplicated), (A, BatchDuplicated), (C, BatchDuplicated)
    )
    X = similar(A)
    test_reverse(
        lyapdkr_inplace_weighted_sum, Active,
        (X, Duplicated), (A, Duplicated), (C, Duplicated), (W, Const)
    )
end
