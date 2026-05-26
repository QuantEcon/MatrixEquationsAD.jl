using Enzyme: Active, Const, Duplicated, Reverse, autodiff, make_zero
using ForwardDiff
using LinearAlgebra: I, dot, norm
using MatrixEquations
using MatrixEquationsAD
using Test

# Test tier flag — see test/runtests.jl.
const _RUN_SLOW_TESTS = get(ENV, "RUN_SLOW_TESTS", "false") == "true"

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

    if _RUN_SLOW_TESTS
        # Enzyme rule for `gsylv` is a non-production wrapper (not
        # exported); gated to the slow tier.
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
    end

    @testset "klein_map" begin
        A = [
            0.00012263591151906127 -0.011623494029190608 0.028377570562199094 0.0 0.0;
            1.0 0.0 0.0 0.0 0.0;
            0.0 0.0 0.0 0.0 0.0;
            0.0 1.0 0.0 0.0 0.0;
            -1.0 0.0 0.0 0.0 0.0
        ]
        B = [
            0.0 0.0 -0.028377570562199098 0.0 0.0;
            -0.98 0.0 1.0 -1.0 0.0;
            -0.07263157894736837 -6.884057971014498 0.0 1.0 0.0;
            0.0 -0.2 0.0 0.0 0.0;
            0.98 0.0 0.0 0.0 1.0
        ]

        r = klein_map(A, B; threshold = 1.0e-6)
        G = vcat(Matrix{Float64}(I, size(r.h_x, 1), size(r.h_x, 1)), r.g_x)
        @test size(r.h_x) == (2, 2)
        @test size(r.g_x) == (3, 2)
        @test norm(A * G * r.h_x + B * G) <= 1.0e-8
    end
end
