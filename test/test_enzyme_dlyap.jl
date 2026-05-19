using Enzyme:
    Active, BatchDuplicated, BatchDuplicatedNoNeed, Const, Duplicated,
    DuplicatedNoNeed, Reverse, autodiff
using EnzymeTestUtils: test_forward, test_reverse
using FiniteDifferences: central_fdm, jvp
using LinearAlgebra: dot, issymmetric
using MatrixEquations
using Random
using Test

function lyapd_enzyme_problem()
    A = [0.55 0.08; -0.04 0.42]
    C = [1.0 0.2; 0.2 0.7]
    return A, C
end

function lyapd_weighted_sum(A, C, W)::Float64
    X = lyapd(A, C)
    return dot(W, X)
end

# using BenchmarkTools
# function bench_lyapd_enzyme()
#     A, C = lyapd_enzyme_problem()
#     @btime lyapd($A, $C)
#     return nothing
# end

@testset "lyapd Enzyme rules" begin
    A, C = lyapd_enzyme_problem()

    test_forward(
        lyapd, DuplicatedNoNeed, (A, Duplicated), (C, Duplicated)
    )
    test_forward(
        lyapd, BatchDuplicatedNoNeed, (A, BatchDuplicated), (C, BatchDuplicated)
    )
    test_reverse(
        lyapd, Duplicated, (A, Duplicated), (C, Duplicated)
    )

    Random.seed!(9753)
    W0 = randn(size(C))
    W = 0.5 .* (W0 + W0')
    dA = zero(A)
    dC = zero(C)
    autodiff(
        Reverse, lyapd_weighted_sum, Active,
        Duplicated(A, dA), Duplicated(C, dC), Const(W)
    )
    @test issymmetric(dC)

    dA_dir = 0.1 .* randn(size(A))
    dC0 = 0.1 .* randn(size(C))
    dC_dir = 0.5 .* (dC0 + dC0')
    fdm = central_fdm(5, 1)
    dfd = jvp(
        fdm, (A_, C_) -> lyapd_weighted_sum(A_, C_, W),
        (A, dA_dir), (C, dC_dir)
    )
    @test dot(dA, dA_dir) + dot(dC, dC_dir) ≈ dfd
end
