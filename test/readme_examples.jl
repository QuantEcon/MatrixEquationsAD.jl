using Enzyme: Active, Const, Duplicated, Reverse, autodiff
using ForwardDiff
using LinearAlgebra: I, dot
using MatrixEquations
using MatrixEquationsAD
using Test

function readme_gsylv_weighted_sum(A, B, C, D, E, W)::Float64
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
        A = Matrix([4.0 0.1 0.0; -0.2 3.6 0.3; 0.1 0.0 3.8])
        B = Matrix([3.0 0.2; -0.1 2.7])
        C = Matrix(0.2I, 3, 3)
        D = Matrix(0.3I, 2, 2)
        E = [1.0 -0.4; 0.3 0.8; -0.2 0.5]
        W = [0.7 -0.1; -0.2 0.4; 0.5 0.3]

        dA = zeros(size(A))
        dB = zeros(size(B))
        dC = zeros(size(C))
        dD = zeros(size(D))
        dE = zeros(size(E))

        autodiff(
            Reverse, readme_gsylv_weighted_sum, Active,
            Duplicated(copy(A), dA),
            Duplicated(copy(B), dB),
            Duplicated(copy(C), dC),
            Duplicated(copy(D), dD),
            Duplicated(copy(E), dE),
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

        F = ordqz(A, B, qzselect_inside_unit)

        @test A ≈ F.Q * F.S * F.Z'
        @test B ≈ F.Q * F.T * F.Z'
    end
end
