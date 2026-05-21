using FiniteDifferences: central_fdm, jvp
using ForwardDiff
using ForwardDiff: Dual
using LinearAlgebra: issymmetric
using MatrixEquations
using MatrixEquationsAD
using StaticArrays: SMatrix
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

    function lyapdkr_static_vec(x)
        A_x = SMatrix{2, 2, eltype(x)}(reshape(x[1:4], 2, 2))
        C_x = SMatrix{2, 2, eltype(x)}(reshape(x[5:8], 2, 2))
        return vec(lyapdkr(A_x, C_x))
    end

    J_static = ForwardDiff.jacobian(lyapdkr_static_vec, x)
    @test lyapdkr_static_vec(x) ≈ vec(lyapdkr(A, C))

    x_dual = map(v -> Dual{Nothing}(v, one(v)), x)
    A_dual = SMatrix{2, 2, eltype(x_dual)}(reshape(x_dual[1:4], 2, 2))
    C_dual = SMatrix{2, 2, eltype(x_dual)}(reshape(x_dual[5:8], 2, 2))
    X_dual = @inferred lyapdkr(A_dual, C_dual)
    @test X_dual isa SMatrix{2, 2, eltype(x_dual)}

    for dx in (
            0.01 .* sin.(3.0 .* collect(1:length(x))),
            0.01 .* cos.(4.0 .* collect(1:length(x))),
        )
        ad = reshape(J_static * dx, 2, 2)
        fd = reshape(jvp(fdm, lyapdkr_static_vec, (x, dx)), 2, 2)
        @test issymmetric(ad)
        @test ad ≈ fd
    end
end
