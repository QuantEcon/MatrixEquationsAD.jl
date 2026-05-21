using FiniteDifferences: central_fdm, jvp
using ForwardDiff
using LinearAlgebra: issymmetric
using MatrixEquations
using MatrixEquationsAD
using Test

@testset "lyapdkr ForwardDiff rules" begin
    A = [0.55 0.08; -0.04 0.42]
    C = [1.0 0.2; 0.2 0.7]
    x = [vec(A); vec(C)]
    fdm = central_fdm(5, 1)

    function lyapdkr_vec(x)
        A_x = reshape(x[1:4], 2, 2)
        C_x = reshape(x[5:8], 2, 2)
        return vec(lyapdkr(A_x, C_x))
    end

    J = ForwardDiff.jacobian(lyapdkr_vec, x)
    @test lyapdkr_vec(x) ≈ vec(lyapdkr(A, C))

    for dx in (
            0.01 .* sin.(1:length(x)),
            0.01 .* cos.(2.0 .* collect(1:length(x))),
        )
        ad = reshape(J * dx, 2, 2)
        fd = reshape(jvp(fdm, lyapdkr_vec, (x, dx)), 2, 2)
        @test issymmetric(ad)
        @test ad ≈ fd
    end
end
