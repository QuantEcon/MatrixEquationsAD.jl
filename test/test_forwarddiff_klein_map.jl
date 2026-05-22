using FiniteDifferences: central_fdm, jvp
using ForwardDiff
using MatrixEquationsAD: klein_map, klein_map!
using StaticArrays: SMatrix
using Test

include(joinpath(@__DIR__, "example_matrices", "rbc.jl"))
include(joinpath(@__DIR__, "example_matrices", "sgu.jl"))

@testset "klein_map ForwardDiff rules" begin
    @testset "RBC heap OOP" begin
        (; A_schur, B_schur, g_x, h_x) = RBCExampleMatrices.dp_rbc_first_order_inputs()
        A = A_schur
        B = B_schur
        n = size(A, 1)
        n_g = length(g_x)
        x = [vec(A); vec(B)]
        fdm = central_fdm(5, 1; max_range = 1.0e-4)

        function klein_oop_vec(x)
            A_x = reshape(x[1:(n * n)], n, n)
            B_x = reshape(x[((n * n) + 1):end], n, n)
            r = klein_map(A_x, B_x; threshold = 1.0e-6)
            return [vec(r.g_x); vec(r.h_x)]
        end

        y = klein_oop_vec(x)
        J = ForwardDiff.jacobian(klein_oop_vec, x)
        @test reshape(y[1:n_g], size(g_x)) ≈ g_x
        @test reshape(y[(n_g + 1):end], size(h_x)) ≈ h_x

        for dx in (
                0.01 .* sin.(1:length(x)),
                0.01 .* cos.(2.0 .* collect(1:length(x))),
            )
            @test J * dx ≈ jvp(fdm, klein_oop_vec, (x, dx)) atol = 1.0e-7 rtol = 1.0e-7
        end
    end

    @testset "RBC static OOP" begin
        (; A_schur, B_schur, g_x, h_x, n_x) =
            RBCExampleMatrices.dp_rbc_first_order_inputs()
        A = A_schur
        B = B_schur
        n = size(A, 1)
        n_g = length(g_x)
        x = [vec(A); vec(B)]
        fdm = central_fdm(5, 1; max_range = 1.0e-4)

        function klein_static_vec(x)
            A_x = SMatrix{n, n, eltype(x)}(reshape(x[1:(n * n)], n, n))
            B_x = SMatrix{n, n, eltype(x)}(reshape(x[((n * n) + 1):end], n, n))
            r = klein_map(A_x, B_x, Val(n_x); threshold = 1.0e-6)
            return [vec(r.g_x); vec(r.h_x)]
        end

        y = klein_static_vec(x)
        J = ForwardDiff.jacobian(klein_static_vec, x)
        @test reshape(y[1:n_g], size(g_x)) ≈ g_x
        @test reshape(y[(n_g + 1):end], size(h_x)) ≈ h_x

        for dx in (
                0.01 .* sin.(3.0 .* collect(1:length(x))),
                0.01 .* cos.(4.0 .* collect(1:length(x))),
            )
            @test J * dx ≈ jvp(fdm, klein_static_vec, (x, dx)) atol = 1.0e-7 rtol = 1.0e-7
        end
    end

    @testset "SGU heap OOP" begin
        (; A_schur, B_schur, g_x, h_x) = SGUExampleMatrices.dp_sgu_first_order_inputs()
        A = A_schur
        B = B_schur
        n = size(A, 1)
        n_g = length(g_x)
        x = [vec(A); vec(B)]
        fdm = central_fdm(5, 1; max_range = 1.0e-4)

        function klein_oop_vec(x)
            A_x = reshape(x[1:(n * n)], n, n)
            B_x = reshape(x[((n * n) + 1):end], n, n)
            r = klein_map(A_x, B_x; threshold = 1.0e-6)
            return [vec(r.g_x); vec(r.h_x)]
        end

        y = klein_oop_vec(x)
        J = ForwardDiff.jacobian(klein_oop_vec, x)
        @test reshape(y[1:n_g], size(g_x)) ≈ g_x
        @test reshape(y[(n_g + 1):end], size(h_x)) ≈ h_x

        for dx in (
                0.01 .* sin.(1:length(x)),
                0.01 .* cos.(2.0 .* collect(1:length(x))),
            )
            @test J * dx ≈ jvp(fdm, klein_oop_vec, (x, dx)) atol = 1.0e-7 rtol = 1.0e-7
        end
    end

    @testset "SGU heap in-place" begin
        (; A_schur, B_schur, g_x, h_x) = SGUExampleMatrices.dp_sgu_first_order_inputs()
        A = A_schur
        B = B_schur
        n = size(A, 1)
        n_g = length(g_x)
        g_size = size(g_x)
        h_size = size(h_x)
        x = [vec(A); vec(B)]
        fdm = central_fdm(5, 1; max_range = 1.0e-4)

        function klein_inplace_vec(x)
            A_x = reshape(x[1:(n * n)], n, n)
            B_x = reshape(x[((n * n) + 1):end], n, n)
            g_x_ip = Matrix{eltype(x)}(undef, g_size)
            h_x_ip = Matrix{eltype(x)}(undef, h_size)
            klein_map!(g_x_ip, h_x_ip, A_x, B_x; threshold = 1.0e-6)
            return [vec(g_x_ip); vec(h_x_ip)]
        end

        y = klein_inplace_vec(x)
        J = ForwardDiff.jacobian(klein_inplace_vec, x)
        @test reshape(y[1:n_g], size(g_x)) ≈ g_x
        @test reshape(y[(n_g + 1):end], size(h_x)) ≈ h_x

        for dx in (
                0.01 .* sin.(3.0 .* collect(1:length(x))),
                0.01 .* cos.(4.0 .* collect(1:length(x))),
            )
            @test J * dx ≈ jvp(fdm, klein_inplace_vec, (x, dx)) atol = 1.0e-7 rtol = 1.0e-7
        end
    end
end
