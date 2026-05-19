using Enzyme: BatchDuplicated, BatchDuplicatedNoNeed, Duplicated, DuplicatedNoNeed
using Enzyme: Forward, autodiff
using EnzymeTestUtils: test_forward, test_reverse
using LinearAlgebra: issymmetric
using MatrixEquations
using MatrixEquationsAD
using Random
using Test

function lyapdkr_enzyme_problem()
    A = [0.55 0.08; -0.04 0.42]
    C = [1.0 0.2; 0.2 0.7]
    return A, C
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
    result = autodiff(
        Forward, lyapdkr, Duplicated,
        BatchDuplicated(A, dAs), BatchDuplicated(C, dCs)
    )
    for dX in result[1]
        @test issymmetric(dX)
    end
end
