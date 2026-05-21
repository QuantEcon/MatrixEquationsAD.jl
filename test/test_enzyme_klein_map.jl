using Enzyme: Active, BatchDuplicated, Const, Duplicated
using EnzymeTestUtils: test_forward, test_reverse
using FiniteDifferences: central_fdm
using LinearAlgebra: dot
using MatrixEquationsAD: klein_map, klein_map!
using Random: MersenneTwister, randn
using Test

include(joinpath(@__DIR__, "example_matrices", "rbc.jl"))

function klein_map_pair!(g_x, h_x, A, B)
    klein_map!(g_x, h_x, A, B; threshold = 1.0e-6)
    return (g_x, h_x)
end

function klein_map_weighted(A, B, Wg, Wh)::Float64
    r = klein_map(A, B; threshold = 1.0e-6)
    return dot(Wg, r.g_x) + dot(Wh, r.h_x)
end

function klein_map_inplace_weighted(A, B, Wg, Wh, n_x, n_y)::Float64
    g_x = Matrix{eltype(A)}(undef, n_y, n_x)
    h_x = Matrix{eltype(A)}(undef, n_x, n_x)
    klein_map!(g_x, h_x, A, B; threshold = 1.0e-6)
    return dot(Wg, g_x) + dot(Wh, h_x)
end

@testset "klein_map Enzyme rules" begin
    A, B, n_x = RBCExampleMatrices.dp_rbc_first_order_gschur()
    n_y = size(A, 1) - n_x
    rng = MersenneTwister(13579)
    Wg = 1.0e-5 .* randn(rng, n_y, n_x)
    Wh = 1.0e-5 .* randn(rng, n_x, n_x)

    @testset "OOP BatchDuplicated forward" begin
        test_forward(
            klein_map_weighted, BatchDuplicated,
            (copy(A), BatchDuplicated), (copy(B), BatchDuplicated),
            (Wg, Const), (Wh, Const);
            rng = MersenneTwister(1234), fdm = central_fdm(5, 1; max_range = 1.0e-6),
        )
    end

    @testset "OOP reverse" begin
        test_reverse(
            klein_map_weighted, Active,
            (copy(A), Duplicated), (copy(B), Duplicated), (Wg, Const), (Wh, Const);
            rng = MersenneTwister(2345), fdm = central_fdm(5, 1; max_range = 1.0e-6),
        )
    end

    @testset "in-place BatchDuplicated forward" begin
        test_forward(
            klein_map_pair!, Const,
            (zeros(n_y, n_x), BatchDuplicated),
            (zeros(n_x, n_x), BatchDuplicated),
            (copy(A), BatchDuplicated),
            (copy(B), BatchDuplicated);
            rng = MersenneTwister(3456),
        )
    end

    @testset "in-place reverse" begin
        test_reverse(
            klein_map_inplace_weighted, Active,
            (copy(A), Duplicated), (copy(B), Duplicated),
            (Wg, Const), (Wh, Const), (n_x, Const), (n_y, Const);
            rng = MersenneTwister(4567), fdm = central_fdm(5, 1; max_range = 1.0e-4),
        )
    end
end
