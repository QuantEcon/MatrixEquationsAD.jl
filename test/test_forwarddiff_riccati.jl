using FiniteDifferences: central_fdm, jvp
using ForwardDiff
using LinearAlgebra: I, issymmetric
using MatrixEquations
using MatrixEquationsAD
using Test

@testset "ared ForwardDiff rules" begin
    A = Matrix([0.95 0.0; 0.0 0.95]')
    B = Matrix([0.5 0.0; 0.0 0.5]')
    R = Matrix(0.2I, 2, 2)
    Q = Matrix(0.5I, 2, 2)
    x = [vec(A); vec(B); vech_symmetric(R); vech_symmetric(Q)]
    fdm = central_fdm(5, 1; max_range = 1.0e-4)

    function ared_vec(x)
        A_x = reshape(x[1:4], 2, 2)
        B_x = reshape(x[5:8], 2, 2)
        R_x = unvech_symmetric(x[9:11], 2)
        Q_x = unvech_symmetric(x[12:14], 2)
        X, _, F = ared(A_x, B_x, R_x, Q_x)
        return [vec(X); vec(F)]
    end

    X, _, F = ared(A, B, R, Q)
    J = ForwardDiff.jacobian(ared_vec, x)
    @test ared_vec(x) ≈ [vec(X); vec(F)]

    for dx in (
            0.01 .* sin.(1:length(x)),
            0.01 .* cos.(2.0 .* collect(1:length(x))),
        )
        ad = J * dx
        fd = jvp(fdm, ared_vec, (x, dx))
        @test issymmetric(reshape(ad[1:4], 2, 2))
        @test ad ≈ fd atol = 1.0e-7 rtol = 1.0e-7
    end
end
