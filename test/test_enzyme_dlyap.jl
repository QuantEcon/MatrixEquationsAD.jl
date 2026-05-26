using Enzyme:
    Active, BatchDuplicated, BatchDuplicatedNoNeed, Const, Duplicated,
    DuplicatedNoNeed, Reverse, autodiff
using EnzymeTestUtils: test_forward, test_reverse
using FiniteDifferences: central_fdm, grad
using LinearAlgebra: Symmetric, dot, norm
using MatrixEquations
using Test

# Test tier flag — see test/runtests.jl.
const _RUN_SLOW_TESTS = get(ENV, "RUN_SLOW_TESTS", "false") == "true"

function lyapd_weighted_sum(A, C, W)::Float64
    X = lyapd(A, C)
    return dot(W, X)
end

function lyapd_symmetric(A, C)
    return lyapd(A, Symmetric(C))
end

function lyapd_symmetric_weighted_sum(A, c, W)::Float64
    X = lyapd(A, Symmetric(unvech_symmetric(c, 2)))
    return dot(W, X)
end

function lyapd_symmetric_factor_weighted_sum(A, C_factor, W)::Float64
    C = C_factor' * C_factor
    X = lyapd(A, Symmetric(C))
    return dot(W, X)
end

# Manual Enzyme reverse vs FD anchor — fast (~5s). The full EnzymeTestUtils
# sweep below is gated under `RUN_SLOW_TESTS`.
@testset "lyapd Enzyme reverse vs FD — n=2 anchor" begin
    A = [0.55 0.08; -0.04 0.42]
    C = [1.0 0.2; 0.2 0.7]
    W = [0.4 -0.1; -0.1 0.7]

    A_bar = zero(A); C_bar = zero(C)
    autodiff(
        Reverse, lyapd_weighted_sum, Active,
        Duplicated(copy(A), A_bar), Duplicated(copy(C), C_bar), Const(W),
    )

    fdm = central_fdm(5, 1)
    A_fd = reshape(grad(fdm, v -> lyapd_weighted_sum(reshape(v, 2, 2), C, W), vec(A))[1], 2, 2)
    C_fd = reshape(grad(fdm, v -> lyapd_weighted_sum(A, reshape(v, 2, 2), W), vec(C))[1], 2, 2)
    @test A_bar ≈ A_fd rtol = 1.0e-8
    @test C_bar ≈ C_fd rtol = 1.0e-8
end

if _RUN_SLOW_TESTS
    # Full EnzymeTestUtils sweep — ~3 min on cold CI. Covers Symmetric-C
    # path, BatchDuplicated, vech-encoded C, factored C variants.
    @testset "lyapd Enzyme rules" begin
        A = [0.55 0.08; -0.04 0.42]
        C = [1.0 0.2; 0.2 0.7]
        c = vech_symmetric(C)
        W = [0.4 -0.1; -0.1 0.7]
        C_factor = [1.0 0.25; -0.35 0.9]

        test_forward(
            lyapd, DuplicatedNoNeed, (A, Duplicated), (C, Duplicated)
        )
        test_forward(
            lyapd, BatchDuplicatedNoNeed, (A, BatchDuplicated), (C, BatchDuplicated)
        )
        test_reverse(
            lyapd, Duplicated, (A, Duplicated), (C, Duplicated)
        )
        test_reverse(
            lyapd_weighted_sum, Active,
            (A, Duplicated), (C, Duplicated), (W, Const)
        )

        test_forward(
            lyapd_symmetric, DuplicatedNoNeed, (A, Duplicated), (C, Duplicated)
        )
        test_reverse(
            lyapd_symmetric_weighted_sum, Active,
            (A, Duplicated), (c, Duplicated), (W, Const)
        )
        test_reverse(
            lyapd_symmetric_factor_weighted_sum, Active,
            (A, Duplicated), (C_factor, Duplicated), (W, Const)
        )
    end
end
