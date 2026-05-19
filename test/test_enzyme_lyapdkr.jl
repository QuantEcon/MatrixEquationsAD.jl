using Enzyme:
    Active, BatchDuplicated, BatchDuplicatedNoNeed, Const, Duplicated,
    DuplicatedNoNeed, Forward, Reverse, autodiff
using EnzymeTestUtils: test_forward, test_reverse
using LinearAlgebra: dot, issymmetric
using MatrixEquations
using MatrixEquationsAD
using Random
using Test

function lyapdkr_enzyme_problem()
    A = [0.55 0.08; -0.04 0.42]
    C = [1.0 0.2; 0.2 0.7]
    return A, C
end

function lyapdkr_weighted_sum(A, C, W)::Float64
    X = lyapdkr(A, C)
    return dot(W, X)
end

function lyapdkr_enzyme_forward_ad(A, C, dA, dC)
    return autodiff(
        Forward, lyapdkr, Duplicated,
        Duplicated(A, dA), Duplicated(C, dC)
    )
end

function lyapdkr_enzyme_reverse_ad(A, C, dA, dC, W)
    return autodiff(
        Reverse, lyapdkr_weighted_sum, Active,
        Duplicated(A, dA), Duplicated(C, dC), Const(W)
    )
end

# using BenchmarkTools
# function bench_lyapdkr_enzyme()
#     A, C = lyapdkr_enzyme_problem()
#     @btime lyapdkr($A, $C)
#     return nothing
# end

@testset "lyapdkr Enzyme rules" begin
    A, C = lyapdkr_enzyme_problem()

    @test lyapdkr(A, C) ≈ lyapd(A, C)

    test_forward(
        lyapdkr, DuplicatedNoNeed, (A, Duplicated), (C, Duplicated)
    )
    test_forward(
        lyapdkr, BatchDuplicatedNoNeed, (A, BatchDuplicated), (C, BatchDuplicated)
    )
    test_reverse(
        lyapdkr, Duplicated, (A, Duplicated), (C, Duplicated)
    )

    Random.seed!(2468)
    N = 2
    dAs = ntuple(_ -> 0.1 .* randn(size(A)), Val(N))
    dCs = ntuple(_ -> 0.1 .* randn(size(C)), Val(N))
    ForwardADReturn = NamedTuple{(Symbol("1"),), Tuple{Matrix{Float64}}}
    ReverseADReturn = Tuple{Tuple{Nothing, Nothing, Nothing}}
    @test (@inferred lyapdkr_enzyme_forward_ad(A, C, dAs[1], dCs[1])) isa
        ForwardADReturn
    W = [0.3 -0.1; 0.2 0.5]
    @test (@inferred lyapdkr_enzyme_reverse_ad(A, C, zero(A), zero(C), W)) isa
        ReverseADReturn

    result = autodiff(
        Forward, lyapdkr, Duplicated,
        BatchDuplicated(A, dAs), BatchDuplicated(C, dCs)
    )
    for dX in result[1]
        @test issymmetric(dX)
    end
end
