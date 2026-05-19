using LinearAlgebra: issymmetric, norm
using MatrixEquations
using MatrixEquationsAD
using Test

function lyapdkr_problem()
    A = [0.55 0.08; -0.04 0.42]
    C = [1.0 0.2; 0.2 0.7]
    return A, C
end

@testset "lyapdkr primal" begin
    A, C = lyapdkr_problem()
    X = @inferred lyapdkr(A, C)

    @test X ≈ lyapd(A, C)
    @test issymmetric(X)
    @test norm(A * X * A' - X + C) < 1.0e-12

    C_general = [1.0 -0.3; 0.5 0.7]
    C_symmetric_part = 0.5 .* (C_general + C_general')
    X_general = lyapdkr(A, C_general)
    @test X_general ≈ lyapd(A, C_symmetric_part)
    @test issymmetric(X_general)
    @test norm(A * X_general * A' - X_general + C_symmetric_part) < 1.0e-12
end
