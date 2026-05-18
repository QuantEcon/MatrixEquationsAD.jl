using Enzyme: BatchDuplicated, BatchDuplicatedNoNeed, Duplicated, DuplicatedNoNeed
using EnzymeTestUtils: test_forward, test_reverse
using MatrixEquations
using Test

function lyapd_enzyme_problem()
    A = [0.55 0.08; -0.04 0.42]
    C = [1.0 0.2; 0.2 0.7]
    return A, C
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
end
