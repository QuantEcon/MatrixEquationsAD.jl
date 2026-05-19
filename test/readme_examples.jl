using Enzyme: Active, Const, Duplicated, Reverse, autodiff, make_zero
using ForwardDiff
using LinearAlgebra: dot
using MatrixEquations
using MatrixEquationsAD
using Test

function readme_gsylv_weighted_sum(A, B, C, D, E, W)
    return dot(W, gsylv(A, B, C, D, E))
end

@testset "README examples" begin
    @testset "ForwardDiff lyapd" begin
        A = [0.55 0.08; -0.04 0.42]
        C = [1.0 0.2; 0.2 0.7]

        function lyapd_sum(x)
            nA = length(A)
            A_dual = reshape(x[1:nA], size(A))
            C_dual = reshape(x[(nA + 1):end], size(C))
            return sum(lyapd(A_dual, C_dual))
        end

        grad = ForwardDiff.gradient(lyapd_sum, [vec(A); vec(C)])

        @test length(grad) == length(A) + length(C)
        @test all(isfinite, grad)
    end

    @testset "Enzyme reverse gsylv" begin
        A = [4.0 0.1 0.0; -0.2 3.6 0.3; 0.1 0.0 3.8]
        B = [3.0 0.2; -0.1 2.7]
        C = [0.2 0.0 0.0; 0.0 0.2 0.0; 0.0 0.0 0.2]
        D = [0.3 0.0; 0.0 0.3]
        E = [1.0 -0.4; 0.3 0.8; -0.2 0.5]
        W = [0.7 -0.1; -0.2 0.4; 0.5 0.3]

        dA = make_zero(A)
        dB = make_zero(B)
        dC = make_zero(C)
        dD = make_zero(D)
        dE = make_zero(E)

        autodiff(
            Reverse, readme_gsylv_weighted_sum, Active,
            Duplicated(A, dA),
            Duplicated(B, dB),
            Duplicated(C, dC),
            Duplicated(D, dD),
            Duplicated(E, dE),
            Const(W),
        )

        @test any(!iszero, dA)
        @test any(!iszero, dB)
        @test any(!iszero, dC)
        @test any(!iszero, dD)
        @test any(!iszero, dE)
    end

    @testset "ordqz wrapper" begin
        A = [1.6 0.2 0.1; 0.0 0.35 -0.1; 0.0 0.0 1.9]
        B = [1.0 0.1 0.0; 0.0 1.2 0.2; 0.0 0.0 0.8]

        n_unstable_expected = 2
        F, n_unstable = ordqz(A, B, :bk; threshold = 1.0e-6)
        n_unstable == n_unstable_expected ||
            error("Blanchard-Kahn condition failed")

        @test n_unstable == 2
        @test A ≈ F.Q * F.S * F.Z'
        @test B ≈ F.Q * F.T * F.Z'
    end
end
