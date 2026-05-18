using Enzyme: Active, BatchDuplicated, Const, Duplicated
using EnzymeTestUtils: test_forward, test_reverse
using FiniteDifferences: central_fdm
using MatrixEquationsAD
using Random
using Test

const ordqz_fdm = central_fdm(5, 1)

function ordqz_enzyme_problem()
    A = [1.6 0.2 0.1; 0.0 0.35 -0.1; 0.0 0.0 1.9]
    B = [1.0 0.1 0.0; 0.0 1.2 0.2; 0.0 0.0 0.8]
    return A, B
end

function ordqz_enzyme_sum!(S, T, Q, Z, A, B)::Float64
    sdim = ordqz!(S, T, Q, Z, A, B, qzselect_inside_unit)
    scale = sdim == 1 ? 1.0 : -1.0
    return scale * (sum(abs2, Q * S * Z') + 0.7 * sum(abs2, Q * T * Z'))
end

function ordqz_reverse_reconstruction_sum(A, B)::Float64
    S = zero(A)
    T = zero(B)
    Q = zero(A)
    Z = zero(A)
    sdim = ordqz!(S, T, Q, Z, A, B, qzselect_inside_unit)
    scale = sdim == 1 ? 1.0 : -1.0
    return scale * (sum(abs2, Q * S * Z') + 0.7 * sum(abs2, Q * T * Z'))
end

@testset "ordqz Enzyme rules" begin
    A, B = ordqz_enzyme_problem()

    test_forward(
        ordqz_enzyme_sum!, Const,
        (zero(A), Duplicated), (zero(B), Duplicated), (zero(A), Duplicated),
        (zero(A), Duplicated), (copy(A), Duplicated), (copy(B), Duplicated);
        rng = Random.MersenneTwister(1234), fdm = ordqz_fdm
    )
    test_forward(
        ordqz_enzyme_sum!, Const,
        (zero(A), BatchDuplicated), (zero(B), BatchDuplicated), (zero(A), BatchDuplicated),
        (zero(A), BatchDuplicated), (copy(A), BatchDuplicated), (copy(B), BatchDuplicated);
        rng = Random.MersenneTwister(1234), fdm = ordqz_fdm
    )
    test_reverse(
        ordqz_reverse_reconstruction_sum, Active,
        (copy(A), Duplicated), (copy(B), Duplicated);
        rng = Random.MersenneTwister(1234), fdm = ordqz_fdm
    )
    test_reverse(
        ordqz_reverse_reconstruction_sum, Active,
        (copy(A), Duplicated), (copy(B), Const);
        rng = Random.MersenneTwister(1234), fdm = ordqz_fdm
    )
end

# using BenchmarkTools
# function bench_ordqz_enzyme_forward()
#     A, B = ordqz_enzyme_problem()
#     S = zero(A)
#     T = zero(B)
#     Q = zero(A)
#     Z = zero(A)
#     @btime ordqz_enzyme_sum!($S, $T, $Q, $Z, $A, $B)
#     return nothing
# end
