using Enzyme: BatchDuplicated, BatchDuplicatedNoNeed, Const, Duplicated, DuplicatedNoNeed
using EnzymeTestUtils: test_forward, test_reverse
using LinearAlgebra: I
using MatrixEquations
using Test

function gsylv_enzyme_problem()
    A = Matrix([4.0 0.1 0.0; -0.2 3.6 0.3; 0.1 0.0 3.8])
    C = Matrix(0.2I, 3, 3)
    B = Matrix([3.0 0.2; -0.1 2.7])
    D = Matrix(0.3I, 2, 2)
    E = [1.0 -0.4; 0.3 0.8; -0.2 0.5]
    return A, B, C, D, E
end

# using BenchmarkTools
# function bench_gsylv_enzyme()
#     A, B, C, D, E = gsylv_enzyme_problem()
#     @btime gsylv($A, $B, $C, $D, $E)
#     return nothing
# end

@testset "gsylv Enzyme rules" begin
    A, B, C, D, E = gsylv_enzyme_problem()

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
