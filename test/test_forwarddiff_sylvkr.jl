using FiniteDifferences: central_fdm, jvp
using ForwardDiff
using MatrixEquations
using Test

@testset "gsylvkr ForwardDiff rules" begin
    A = [4.0 0.2 -0.1; -0.3 3.7 0.4; 0.1 -0.2 3.5]
    B = [2.8 -0.3; 0.2 3.1]
    C = [0.5 0.1 -0.2; 0.0 0.7 0.3; -0.1 0.2 0.6]
    D = [0.9 0.2; -0.4 0.8]
    E = [1.0 -0.4; 0.3 0.8; -0.2 0.5]
    x = [vec(A); vec(B); vec(C); vec(D); vec(E)]
    fdm = central_fdm(5, 1)

    function gsylvkr_vec(x)
        A_x = reshape(x[1:9], 3, 3)
        B_x = reshape(x[10:13], 2, 2)
        C_x = reshape(x[14:22], 3, 3)
        D_x = reshape(x[23:26], 2, 2)
        E_x = reshape(x[27:32], 3, 2)
        return vec(gsylvkr(A_x, B_x, C_x, D_x, E_x))
    end

    J = ForwardDiff.jacobian(gsylvkr_vec, x)
    @test gsylvkr_vec(x) ≈ vec(gsylvkr(A, B, C, D, E))

    for dx in (
            0.01 .* sin.(1:length(x)),
            0.01 .* cos.(2.0 .* collect(1:length(x))),
        )
        @test J * dx ≈ jvp(fdm, gsylvkr_vec, (x, dx))
    end
end
