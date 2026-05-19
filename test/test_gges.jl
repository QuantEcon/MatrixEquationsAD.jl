using LinearAlgebra: I
using MatrixEquationsAD
using Test

function gges_problem()
    A = [1.6 0.2 0.1; 0.0 0.35 -0.1; 0.0 0.0 1.9]
    B = [1.0 0.1 0.0; 0.0 1.2 0.2; 0.0 0.0 0.8]
    return A, B
end

@testset "gges wrapper" begin
    A, B = gges_problem()
    A0 = copy(A)
    B0 = copy(B)
    threshold = 1.0e-6
    criterium = (1 - threshold)^2

    F, expected = ordqz(A, B, :bk; threshold)
    result = gges(A, B; select = :ed, criterium)

    @test expected == 2
    @test result.n_explosive == expected
    @test A == A0
    @test B == B0
    @test result.Q' * result.Q ≈ I
    @test result.Z' * result.Z ≈ I
    @test A ≈ result.Q * result.S * result.Z'
    @test B ≈ result.Q * result.T * result.Z'
    @test abs.(result.S) ≈ abs.(F.S)
    @test abs.(result.T) ≈ abs.(F.T)

    S = zero(A)
    T = zero(B)
    Q = zero(A)
    Z = zero(A)
    inplace = gges!(S, T, Q, Z, A, B; select = :ed, criterium)
    @test inplace.S === S
    @test inplace.T === T
    @test inplace.Q === Q
    @test inplace.Z === Z
    @test inplace.n_explosive == expected
    @test A == A0
    @test B == B0
    @test S ≈ result.S
    @test T ≈ result.T
    @test Q ≈ result.Q
    @test Z ≈ result.Z

    @test_throws ArgumentError gges(A, B; select = :id, criterium)
    @test_throws ArgumentError gges(A, B; select = :ed, criterium = -1.0)
    @test_throws ArgumentError gges(
        ones(Int, 3, 3), Matrix{Int}(I, 3, 3); select = :ed, criterium
    )
    @test_throws DimensionMismatch gges(A[1:2, :], B; select = :ed, criterium)
    @test_throws DimensionMismatch gges!(zeros(2, 2), T, Q, Z, A, B; select = :ed, criterium)
end

@testset "gges Float32 wrapper" begin
    A, B = gges_problem()
    A32 = Float32.(A)
    B32 = Float32.(B)
    result = gges(A32, B32; select = :ed, criterium = Float32((1 - 1.0e-6)^2))

    @test eltype(result.S) === Float32
    @test result.n_explosive == 2
    @test result.Q' * result.Q ≈ I atol = 1.0e-5
    @test result.Z' * result.Z ≈ I atol = 1.0e-5
    @test A32 ≈ result.Q * result.S * result.Z' atol = 1.0e-5
    @test B32 ≈ result.Q * result.T * result.Z' atol = 1.0e-5
end

@testset "gges DifferentiablePerturbation fixtures" begin
    for (A, B, expected) in (dp_rbc_ordqz_problem(), dp_rbc_sv_ordqz_problem())
        threshold = dp_ordqz_threshold
        criterium = (1 - threshold)^2
        result = gges(A, B; select = :ed, criterium)

        @test result.n_explosive == expected
        @test result.Q' * result.Q ≈ I
        @test result.Z' * result.Z ≈ I
        @test A ≈ result.Q * result.S * result.Z'
        @test B ≈ result.Q * result.T * result.Z'
    end
end
