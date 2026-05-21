using FiniteDifferences: central_fdm, jvp
using ForwardDiff
using LinearAlgebra: I
using MatrixEquations
using Test

@testset "gsylv ForwardDiff rules" begin
    A = [4.0 0.1 0.0; -0.2 3.6 0.3; 0.1 0.0 3.8]
    B = [3.0 0.2; -0.1 2.7]
    C = Matrix(0.2I, 3, 3)
    D = Matrix(0.3I, 2, 2)
    E = [1.0 -0.4; 0.3 0.8; -0.2 0.5]
    x = [vec(A); vec(B); vec(C); vec(D); vec(E)]
    fdm = central_fdm(5, 1)

    function gsylv_vec(x)
        A_x = reshape(x[1:9], 3, 3)
        B_x = reshape(x[10:13], 2, 2)
        C_x = reshape(x[14:22], 3, 3)
        D_x = reshape(x[23:26], 2, 2)
        E_x = reshape(x[27:32], 3, 2)
        return vec(gsylv(A_x, B_x, C_x, D_x, E_x))
    end

    J = ForwardDiff.jacobian(gsylv_vec, x)
    @test gsylv_vec(x) ≈ vec(gsylv(A, B, C, D, E))

    for dx in (
            0.01 .* sin.(1:length(x)),
            0.01 .* cos.(2.0 .* collect(1:length(x))),
        )
        @test J * dx ≈ jvp(fdm, gsylv_vec, (x, dx))
    end
end
