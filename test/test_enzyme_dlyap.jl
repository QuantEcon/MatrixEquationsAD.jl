using Enzyme:
    Active, BatchDuplicated, BatchDuplicatedNoNeed, Const, Duplicated,
    DuplicatedNoNeed, Forward, Reverse, autodiff
using EnzymeTestUtils: test_forward, test_reverse
using FiniteDifferences: central_fdm, jvp
using LinearAlgebra: Symmetric, dot, issymmetric
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

function lyapd_symmetric(A, C)
    return lyapd(A, Symmetric(C))
end

function lyapd_symmetric_weighted_sum(A, C, W)::Float64
    X = lyapd_symmetric(A, C)
    return dot(W, X)
end

function lyapd_symmetric_wrappers(A, C)
    return lyapd(Symmetric(A), Symmetric(C))
end

function lyapd_symmetric_wrappers_weighted_sum(A, C, W)::Float64
    X = lyapd_symmetric_wrappers(A, C)
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

    dX = autodiff(
        Forward, lyapd_symmetric, Duplicated,
        Duplicated(A, dA_dir), Duplicated(C, dC_dir)
    )[1]
    dX_fd = jvp(fdm, lyapd_symmetric, (A, dA_dir), (C, dC_dir))
    @test issymmetric(dX)
    @test dX ≈ dX_fd

    fill!(dA, 0)
    fill!(dC, 0)
    autodiff(
        Reverse, lyapd_symmetric_weighted_sum, Active,
        Duplicated(A, dA), Duplicated(C, dC), Const(W)
    )
    @test issymmetric(dC)
    dfd_symmetric = jvp(
        fdm, (A_, C_) -> lyapd_symmetric_weighted_sum(A_, C_, W),
        (A, dA_dir), (C, dC_dir)
    )
    @test dot(dA, dA_dir) + dot(dC, dC_dir) ≈ dfd_symmetric

    A_sym = [0.45 0.08; 0.08 0.35]
    @test @inferred(lyapd(Symmetric(A_sym), Symmetric(C))) ≈ lyapd(A_sym, C)
    dA0 = 0.1 .* randn(size(A_sym))
    dA_sym_dir = 0.5 .* (dA0 + dA0')
    dX_sym_wrappers = autodiff(
        Forward, lyapd_symmetric_wrappers, Duplicated,
        Duplicated(A_sym, dA_sym_dir), Duplicated(C, dC_dir)
    )[1]
    dX_sym_wrappers_fd = jvp(
        fdm, lyapd_symmetric_wrappers,
        (A_sym, dA_sym_dir), (C, dC_dir)
    )
    @test issymmetric(dX_sym_wrappers)
    @test dX_sym_wrappers ≈ dX_sym_wrappers_fd

    dA_wrapper = zero(A_sym)
    fill!(dC, 0)
    autodiff(
        Reverse, lyapd_symmetric_wrappers_weighted_sum, Active,
        Duplicated(A_sym, dA_wrapper), Duplicated(C, dC), Const(W)
    )
    @test issymmetric(dA_wrapper)
    @test issymmetric(dC)
    dfd_sym_wrappers = jvp(
        fdm, (A_, C_) -> lyapd_symmetric_wrappers_weighted_sum(A_, C_, W),
        (A_sym, dA_sym_dir), (C, dC_dir)
    )
    @test dot(dA_wrapper, dA_sym_dir) + dot(dC, dC_dir) ≈ dfd_sym_wrappers
end
