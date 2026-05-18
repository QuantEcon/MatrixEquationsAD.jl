using LinearAlgebra: I, ordschur!, schur
using MatrixEquationsAD
using Test

function ordqz_problem()
    A = [1.6 0.2 0.1; 0.0 0.35 -0.1; 0.0 0.0 1.9]
    B = [1.0 0.1 0.0; 0.0 1.2 0.2; 0.0 0.0 0.8]
    return A, B
end

@testset "ordered QZ wrapper" begin
    A, B = ordqz_problem()
    F = ordqz(A, B, qzselect_inside_unit)
    F_ref = schur(A, B)
    ordschur!(F_ref, abs.(F_ref.α) .< abs.(F_ref.β))

    @test F.S ≈ F_ref.S
    @test F.T ≈ F_ref.T
    @test F.Q ≈ F_ref.Q
    @test F.Z ≈ F_ref.Z
    @test A ≈ F.Q * F.S * F.Z'
    @test B ≈ F.Q * F.T * F.Z'
    @test F.Q' * F.Q ≈ I
    @test F.Z' * F.Z ≈ I

    S = zero(A)
    T = zero(B)
    Q = zero(A)
    Z = zero(A)
    @test ordqz!(S, T, Q, Z, A, B, qzselect_inside_unit) == 1
    @test S ≈ F.S
    @test T ≈ F.T
    @test Q ≈ F.Q
    @test Z ≈ F.Z
end

@testset "ordered QZ DifferentiablePerturbation fixtures" begin
    for (A, B, expected) in (dp_rbc_ordqz_problem(), dp_rbc_sv_ordqz_problem())
        F = ordqz(A, B, dp_ordqz_select)
        F_ref = schur(A, B)
        select = dp_ordqz_select.(F_ref.α, F_ref.β)
        @test count(select) == expected
        ordschur!(F_ref, select)

        @test F.S ≈ F_ref.S
        @test F.T ≈ F_ref.T
        @test F.Q ≈ F_ref.Q
        @test F.Z ≈ F_ref.Z
        @test A ≈ F.Q * F.S * F.Z'
        @test B ≈ F.Q * F.T * F.Z'

        S = zero(A)
        T = zero(B)
        Q = zero(A)
        Z = zero(A)
        @test ordqz!(S, T, Q, Z, A, B, dp_ordqz_select) == expected
        @test S ≈ F.S
        @test T ≈ F.T
        @test Q ≈ F.Q
        @test Z ≈ F.Z
    end
end
