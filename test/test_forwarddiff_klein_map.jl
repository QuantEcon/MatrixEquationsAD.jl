using FiniteDifferences: central_fdm, jvp
using ForwardDiff
using MatrixEquationsAD: klein_map, klein_map!
using StaticArrays: SMatrix
using Test

include(joinpath(@__DIR__, "example_matrices", "rbc.jl"))
include(joinpath(@__DIR__, "example_matrices", "sgu.jl"))
include(joinpath(@__DIR__, "klein_map_fixtures.jl"))

@testset "klein_map ForwardDiff rules" begin
    @testset "RBC heap OOP" begin
        A, B, _ = RBCExampleMatrices.dp_rbc_first_order_gschur()
        F = KleinMapFixtures.KLEIN_RBC
        n = size(A, 1)
        n_g = length(F.g_x)
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
        @test reshape(y[1:n_g], size(F.g_x)) ≈ F.g_x
        @test reshape(y[(n_g + 1):end], size(F.h_x)) ≈ F.h_x

        for dx in (
                0.01 .* sin.(1:length(x)),
                0.01 .* cos.(2.0 .* collect(1:length(x))),
            )
            @test J * dx ≈ jvp(fdm, klein_oop_vec, (x, dx)) atol = 1.0e-7 rtol = 1.0e-7
        end
    end

    @testset "RBC static OOP" begin
        A, B, _ = RBCExampleMatrices.dp_rbc_first_order_gschur()
        F = KleinMapFixtures.KLEIN_RBC
        n = size(A, 1)
        n_g = length(F.g_x)
        x = [vec(A); vec(B)]
        fdm = central_fdm(5, 1; max_range = 1.0e-4)

        function klein_static_vec(x)
            A_x = SMatrix{n, n, eltype(x)}(reshape(x[1:(n * n)], n, n))
            B_x = SMatrix{n, n, eltype(x)}(reshape(x[((n * n) + 1):end], n, n))
            r = klein_map(A_x, B_x; threshold = 1.0e-6)
            return [vec(r.g_x); vec(r.h_x)]
        end

        y = klein_static_vec(x)
        J = ForwardDiff.jacobian(klein_static_vec, x)
        @test reshape(y[1:n_g], size(F.g_x)) ≈ F.g_x
        @test reshape(y[(n_g + 1):end], size(F.h_x)) ≈ F.h_x

        for dx in (
                0.01 .* sin.(3.0 .* collect(1:length(x))),
                0.01 .* cos.(4.0 .* collect(1:length(x))),
            )
            @test J * dx ≈ jvp(fdm, klein_static_vec, (x, dx)) atol = 1.0e-7 rtol = 1.0e-7
        end
    end

    @testset "SGU heap OOP" begin
        A, B, _ = SGUExampleMatrices.dp_sgu_first_order_gschur()
        F = KleinMapFixtures.KLEIN_SGU
        n = size(A, 1)
        n_g = length(F.g_x)
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
        @test reshape(y[1:n_g], size(F.g_x)) ≈ F.g_x
        @test reshape(y[(n_g + 1):end], size(F.h_x)) ≈ F.h_x

        for dx in (
                0.01 .* sin.(1:length(x)),
                0.01 .* cos.(2.0 .* collect(1:length(x))),
            )
            @test J * dx ≈ jvp(fdm, klein_oop_vec, (x, dx)) atol = 1.0e-7 rtol = 1.0e-7
        end
    end

    @testset "SGU heap in-place" begin
        A, B, _ = SGUExampleMatrices.dp_sgu_first_order_gschur()
        F = KleinMapFixtures.KLEIN_SGU
        n = size(A, 1)
        n_g = length(F.g_x)
        g_size = size(F.g_x)
        h_size = size(F.h_x)
        x = [vec(A); vec(B)]
        fdm = central_fdm(5, 1; max_range = 1.0e-4)

        function klein_inplace_vec(x)
            A_x = reshape(x[1:(n * n)], n, n)
            B_x = reshape(x[((n * n) + 1):end], n, n)
            g_x = Matrix{eltype(x)}(undef, g_size)
            h_x = Matrix{eltype(x)}(undef, h_size)
            klein_map!(g_x, h_x, A_x, B_x; threshold = 1.0e-6)
            return [vec(g_x); vec(h_x)]
        end

        y = klein_inplace_vec(x)
        J = ForwardDiff.jacobian(klein_inplace_vec, x)
        @test reshape(y[1:n_g], size(F.g_x)) ≈ F.g_x
        @test reshape(y[(n_g + 1):end], size(F.h_x)) ≈ F.h_x

        for dx in (
                0.01 .* sin.(3.0 .* collect(1:length(x))),
                0.01 .* cos.(4.0 .* collect(1:length(x))),
            )
            @test J * dx ≈ jvp(fdm, klein_inplace_vec, (x, dx)) atol = 1.0e-7 rtol = 1.0e-7
        end
    end
end
