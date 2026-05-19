using Enzyme: Active, BatchDuplicated, Const, Duplicated, Forward, Reverse, autodiff
using FiniteDifferences: central_fdm, jvp
using LinearAlgebra: dot, issymmetric
using MatrixEquations
using MatrixEquationsAD
using Random
using Test

function ared_weighted_sum(A, B, R, Q, WX, WF)::Float64
    X, _, F = ared(A, B, R, Q)
    return dot(WX, X) + dot(WF, F)
end

# using BenchmarkTools
# function bench_ared_enzyme_forward()
#     A, B, R, Q = quantecon_kalman_ared_problem()
#     dA = 0.1 .* randn(size(A))
#     dB = 0.1 .* randn(size(B))
#     dR = zero(R)
#     dQ = zero(Q)
#     WX = zero(Q)
#     WF = zero(B')
#     @btime autodiff(
#         Forward, ared_weighted_sum,
#         Duplicated($A, $dA), Duplicated($B, $dB),
#         Duplicated($R, $dR), Duplicated($Q, $dQ), Const($WX), Const($WF)
#     )
#     return nothing
# end

@testset "ared Enzyme rules" begin
    rng = Random.MersenneTwister(3344)
    A, B, R, Q = quantecon_kalman_ared_problem()
    X, _, F = ared(A, B, R, Q)
    @test issymmetric(X)
    @test @inferred(ared(A, B, R, Q, zero(B))) isa Tuple

    WX = random_symmetric_direction(rng, X, 1.0)
    WF = randn(rng, size(F))
    dA_dir = 0.1 .* randn(rng, size(A))
    dB_dir = 0.1 .* randn(rng, size(B))
    dR_dir = random_symmetric_direction(rng, R)
    dQ_dir = random_symmetric_direction(rng, Q)
    fdm = central_fdm(5, 1; max_range = 1.0e-4)
    fd = jvp(
        fdm, (A_, B_, R_, Q_) -> ared_weighted_sum(A_, B_, R_, Q_, WX, WF),
        (A, dA_dir), (B, dB_dir), (R, dR_dir), (Q, dQ_dir)
    )

    fwd = autodiff(
        Forward, ared_weighted_sum,
        Duplicated(A, dA_dir), Duplicated(B, dB_dir),
        Duplicated(R, dR_dir), Duplicated(Q, dQ_dir), Const(WX), Const(WF)
    )
    @test only(fwd) ≈ fd

    dirs = (
        0.1 .* randn(rng, size(A)),
        0.1 .* randn(rng, size(A)),
    )
    batch = autodiff(
        Forward, ared_weighted_sum,
        BatchDuplicated(A, dirs), BatchDuplicated(B, (dB_dir, dB_dir)),
        BatchDuplicated(R, (dR_dir, dR_dir)),
        BatchDuplicated(Q, (dQ_dir, dQ_dir)), Const(WX), Const(WF)
    )
    batch_vals = only(batch)
    @test length(batch_vals) == 2
    for i in 1:2
        fd_batch = jvp(
            fdm, (A_, B_, R_, Q_) -> ared_weighted_sum(A_, B_, R_, Q_, WX, WF),
            (A, dirs[i]), (B, dB_dir), (R, dR_dir), (Q, dQ_dir)
        )
        @test batch_vals[i] ≈ fd_batch
    end

    dA = zero(A)
    dB = zero(B)
    dR = zero(R)
    dQ = zero(Q)
    autodiff(
        Reverse, ared_weighted_sum, Active,
        Duplicated(A, dA), Duplicated(B, dB),
        Duplicated(R, dR), Duplicated(Q, dQ), Const(WX), Const(WF)
    )
    @test dot(dA, dA_dir) + dot(dB, dB_dir) + dot(dR, dR_dir) + dot(dQ, dQ_dir) ≈ fd
end
