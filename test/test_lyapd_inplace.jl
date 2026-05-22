using Enzyme:
    Active, BatchDuplicated, BatchDuplicatedNoNeed, Const, Duplicated,
    DuplicatedNoNeed, Enzyme, Forward, Reverse, autodiff
using EnzymeTestUtils: test_forward, test_reverse
using ForwardDiff: ForwardDiff, Dual, Partials
using LinearAlgebra: Symmetric, dot, opnorm
using MatrixEquations
using MatrixEquationsAD
using Random
using Test

function lyapd_inplace_weighted!(X, A, C, W)::Float64
    lyapd!(X, A, C)
    return dot(W, X)
end

@testset "lyapd! primal" begin
    rng = MersenneTwister(11)
    for n in (2, 4, 7)
        A = 0.3 .* randn(rng, n, n)
        A ./= 1.2 * opnorm(A)
        M = randn(rng, n, n)
        C = 0.5 .* (M + M')

        X_ref = lyapd(A, C)
        X = zeros(n, n)
        ret = lyapd!(X, A, C)
        @test ret === nothing
        @test X ≈ X_ref atol = 1.0e-12 rtol = 1.0e-12

        Xs = zeros(n, n)
        @test lyapd!(Xs, A, Symmetric(C)) === nothing
        @test Xs ≈ X_ref atol = 1.0e-12 rtol = 1.0e-12

        # Dimension mismatch
        @test_throws DimensionMismatch lyapd!(zeros(n + 1, n + 1), A, C)
    end
end

@testset "lyapd! Enzyme rules" begin
    rng = MersenneTwister(13)
    n = 3
    A = 0.3 .* randn(rng, n, n)
    A ./= 1.2 * opnorm(A)
    M = randn(rng, n, n)
    C = 0.5 .* (M + M')
    W = randn(rng, n, n)
    X0 = zeros(n, n)

    test_forward(
        lyapd!, Const,
        (zeros(n, n), Duplicated),
        (copy(A), Duplicated),
        (copy(C), Duplicated),
    )
    test_forward(
        lyapd!, Const,
        (zeros(n, n), BatchDuplicated),
        (copy(A), BatchDuplicated),
        (copy(C), BatchDuplicated),
    )

    test_reverse(
        lyapd_inplace_weighted!, Active,
        (zeros(n, n), Duplicated),
        (copy(A), Duplicated),
        (copy(C), Duplicated),
        (W, Const),
    )

    # Reverse-on-`Symmetric` C: EnzymeTestUtils' FD path treats the wrapped
    # matrix as a generic StridedMatrix, perturbing upper and lower triangles
    # independently, while the AD rule projects the cotangent onto the
    # symmetric manifold. The vech-parametrised version is exercised by
    # `test_enzyme_dlyap.jl` for the OOP `lyapd` and verified manually to
    # produce the same shadows as the OOP reverse on the same loss.
end

@testset "lyapd! ForwardDiff Dual chunk" begin
    rng = MersenneTwister(17)
    n = 4
    A = 0.3 .* randn(rng, n, n)
    A ./= 1.2 * opnorm(A)
    M = randn(rng, n, n)
    C = 0.5 .* (M + M')

    NCH = 4
    dA_lanes = ntuple(_ -> randn(rng, n, n), Val(NCH))
    dC_lanes = ntuple(NCH) do _
        N = randn(rng, n, n)
        0.5 .* (N + N')
    end

    A_dual = map(A, dA_lanes...) do a, ds...
        Dual{Nothing}(a, ds...)
    end
    C_dual = map(C, dC_lanes...) do c, ds...
        Dual{Nothing}(c, ds...)
    end

    X_dual = similar(A_dual)
    @test lyapd!(X_dual, A_dual, C_dual) === nothing

    # Value layer matches OOP primal
    X_ref = lyapd(A, C)
    @test ForwardDiff.value.(X_dual) ≈ X_ref atol = 1.0e-12

    # Each partial lane matches the per-lane OOP Dual round-trip
    for i in 1:NCH
        A_i = map(A, dA_lanes[i]) do a, d
            Dual{Nothing}(a, d)
        end
        C_i = map(C, dC_lanes[i]) do c, d
            Dual{Nothing}(c, d)
        end
        ref = ForwardDiff.partials.(lyapd(A_i, C_i), 1)
        @test map(x -> ForwardDiff.partials(x, i), X_dual) ≈ ref atol = 1.0e-12
    end
end
