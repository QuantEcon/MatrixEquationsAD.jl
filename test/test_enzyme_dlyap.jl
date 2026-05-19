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

function lyapd_symmetric_factor_weighted_sum(A, C_factor, W)::Float64
    C = C_factor' * C_factor
    X = lyapd(A, Symmetric(C))
    return dot(W, X)
end

function lyapd_enzyme_forward_ad(A, C, dA, dC)
    return autodiff(
        Forward, lyapd, Duplicated,
        Duplicated(A, dA), Duplicated(C, dC)
    )
end

function lyapd_symmetric_enzyme_forward_ad(A, C, dA, dC)
    return autodiff(
        Forward, lyapd_symmetric, Duplicated,
        Duplicated(A, dA), Duplicated(C, dC)
    )
end

function lyapd_enzyme_reverse_ad(A, C, dA, dC, W)
    return autodiff(
        Reverse, lyapd_weighted_sum, Active,
        Duplicated(A, dA), Duplicated(C, dC), Const(W)
    )
end

function lyapd_symmetric_enzyme_reverse_ad(A, C, dA, dC, W)
    return autodiff(
        Reverse, lyapd_symmetric_weighted_sum, Active,
        Duplicated(A, dA), Duplicated(C, dC), Const(W)
    )
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
    ForwardADReturn = NamedTuple{(Symbol("1"),), Tuple{Matrix{Float64}}}
    ReverseADReturn = Tuple{Tuple{Nothing, Nothing, Nothing}}
    @test (@inferred lyapd_enzyme_forward_ad(A, C, dA_dir, dC_dir)) isa
        ForwardADReturn
    @test (@inferred lyapd_symmetric_enzyme_forward_ad(A, C, dA_dir, dC_dir)) isa
        ForwardADReturn
    dA_inferred = zero(A)
    dC_inferred = zero(C)
    @test (@inferred lyapd_enzyme_reverse_ad(A, C, dA_inferred, dC_inferred, W)) isa
        ReverseADReturn
    fill!(dA_inferred, 0)
    fill!(dC_inferred, 0)
    @test (
        @inferred lyapd_symmetric_enzyme_reverse_ad(
            A, C, dA_inferred, dC_inferred, W
        )
    ) isa ReverseADReturn

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

    C_factor = [1.0 0.25; -0.35 0.9]
    dC_factor = zero(C_factor)
    fill!(dA, 0)
    autodiff(
        Reverse, lyapd_symmetric_factor_weighted_sum, Active,
        Duplicated(A, dA), Duplicated(C_factor, dC_factor), Const(W)
    )

    dC_factor_dir = 0.1 .* randn(size(C_factor))
    dfd_factor = jvp(
        fdm, (A_, C_factor_) -> lyapd_symmetric_factor_weighted_sum(A_, C_factor_, W),
        (A, dA_dir), (C_factor, dC_factor_dir)
    )
    @test dot(dA, dA_dir) + dot(dC_factor, dC_factor_dir) ≈ dfd_factor
end
