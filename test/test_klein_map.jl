# Value-comparison and AD tests for klein_map / klein_map! against the
# ground-truth (g_x, h_x) matrices bundled inside each model's
# `dp_<model>_first_order_inputs()` fixture
# (test/example_matrices/<model>.jl). The fixture function is internally
# consistent:
#   klein_map(A_schur, B_schur; threshold = 1.0e-6) reproduces (g_x, h_x).
#
# Test convention: full-matrix ≈ at atol = rtol = 1.0e-10. Heap, in-place,
# SMatrix heap fallback, and explicit Val-sized SMatrix output are checked
# per model; SMatrix is skipped for n > 15 where it stops being
# meaningful. AD coverage (ForwardDiff Jacobian + Enzyme forward/reverse
# rules) runs at the RBC pencil (n = 5) — small enough that the dense
# big-K factorisation is cheap and `EnzymeTestUtils` finite-difference
# probes stay inside the linearity ball — plus an SGU heap in-place
# ForwardDiff Jacobian round-trip at n = 15.

using Enzyme: Active, BatchDuplicated, Const, Duplicated
using EnzymeTestUtils: test_forward, test_reverse
using FiniteDifferences: central_fdm, jvp
using ForwardDiff
using LinearAlgebra: I, dot, norm
using MatrixEquationsAD: klein_map, klein_map!
using Random: MersenneTwister
using StaticArrays: SMatrix
using Test

include(joinpath(@__DIR__, "example_matrices", "rbc.jl"))
include(joinpath(@__DIR__, "example_matrices", "sgu.jl"))
include(joinpath(@__DIR__, "example_matrices", "fvgq.jl"))
include(joinpath(@__DIR__, "example_matrices", "sw07.jl"))

@testset "klein_map" begin
    fixtures = (
        ("rbc",         RBCExampleMatrices.dp_rbc_first_order_inputs()),
        ("rbc_sv",      RBCExampleMatrices.dp_rbc_sv_first_order_inputs()),
        ("sgu",         SGUExampleMatrices.dp_sgu_first_order_inputs()),
        ("fvgq",        FVGQExampleMatrices.dp_fvgq_first_order_inputs()),
        ("sw07pfeifer", SW07ExampleMatrices.dp_sw07pfeifer_first_order_inputs()),
    )
    for (name, fo) in fixtures
        (; A_schur, B_schur, g_x, h_x, n_x) = fo
        A = A_schur
        B = B_schur
        @testset "$name" begin
            r = klein_map(A, B; threshold = 1.0e-6)
            @test r.g_x ≈ g_x atol = 1.0e-10 rtol = 1.0e-10
            @test r.h_x ≈ h_x atol = 1.0e-10 rtol = 1.0e-10
            G = vcat(Matrix{Float64}(I, size(r.h_x, 1), size(r.h_x, 1)), r.g_x)
            @test norm(A * G * r.h_x + B * G) <= 1.0e-8

            g_x_ip = similar(g_x)
            h_x_ip = similar(h_x)
            klein_map!(g_x_ip, h_x_ip, A, B; threshold = 1.0e-6)
            @test g_x_ip ≈ g_x atol = 1.0e-10 rtol = 1.0e-10
            @test h_x_ip ≈ h_x atol = 1.0e-10 rtol = 1.0e-10
            G .= vcat(Matrix{Float64}(I, size(h_x_ip, 1), size(h_x_ip, 1)), g_x_ip)
            @test norm(A * G * h_x_ip + B * G) <= 1.0e-8

            if size(A, 1) <= 15
                n = size(A, 1)
                As = SMatrix{n, n, Float64}(A)
                Bs = SMatrix{n, n, Float64}(B)
                rs = klein_map(As, Bs; threshold = 1.0e-6)
                @test rs.g_x isa Matrix{Float64}
                @test rs.h_x isa Matrix{Float64}
                @test rs.g_x ≈ g_x atol = 1.0e-10 rtol = 1.0e-10
                @test rs.h_x ≈ h_x atol = 1.0e-10 rtol = 1.0e-10

                n_y = size(g_x, 1)
                rs_static = @inferred klein_map(As, Bs, Val(n_x); threshold = 1.0e-6)
                @test rs_static.g_x isa SMatrix{n_y, n_x, Float64}
                @test rs_static.h_x isa SMatrix{n_x, n_x, Float64}
                @test Matrix(rs_static.g_x) ≈ g_x atol = 1.0e-10 rtol = 1.0e-10
                @test Matrix(rs_static.h_x) ≈ h_x atol = 1.0e-10 rtol = 1.0e-10
                @test_throws DimensionMismatch klein_map(
                    As, Bs, Val(n + 1); threshold = 1.0e-6,
                )
                @test_throws DimensionMismatch klein_map(
                    As, Bs, Val(n_x + 1); threshold = 1.0e-6,
                )
            end
        end
    end
end

# Scalar-loss helpers reused across the Enzyme reverse / static-size lanes.
function klein_map_pair!(g_x, h_x, A, B)
    klein_map!(g_x, h_x, A, B; threshold = 1.0e-6)
    return (g_x, h_x)
end

function klein_map_weighted(A, B, Wg, Wh)::Float64
    r = klein_map(A, B; threshold = 1.0e-6)
    return dot(Wg, r.g_x) + dot(Wh, r.h_x)
end

function klein_map_static_weighted(A, B, Wg, Wh, ::Val{n_x})::Float64 where {n_x}
    r = klein_map(A, B, Val(n_x); threshold = 1.0e-6)
    return dot(Wg, r.g_x) + dot(Wh, r.h_x)
end

function klein_map_inplace_weighted(A, B, Wg, Wh, n_x, n_y)::Float64
    g_x = Matrix{eltype(A)}(undef, n_y, n_x)
    h_x = Matrix{eltype(A)}(undef, n_x, n_x)
    klein_map!(g_x, h_x, A, B; threshold = 1.0e-6)
    return dot(Wg, g_x) + dot(Wh, h_x)
end

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

@testset "klein_map Enzyme rules" begin
    (; A_schur, B_schur, n_x) = RBCExampleMatrices.dp_rbc_first_order_inputs()
    A = A_schur
    B = B_schur
    n_y = size(A, 1) - n_x
    rng = MersenneTwister(13579)
    Wg = 1.0e-5 .* randn(rng, n_y, n_x)
    Wh = 1.0e-5 .* randn(rng, n_x, n_x)
    n = size(A, 1)
    As = SMatrix{n, n, Float64}(A)
    Bs = SMatrix{n, n, Float64}(B)
    Wgs = SMatrix{n_y, n_x, Float64}(Wg)
    Whs = SMatrix{n_x, n_x, Float64}(Wh)

    @testset "OOP BatchDuplicated forward" begin
        test_forward(
            klein_map_weighted, BatchDuplicated,
            (copy(A), BatchDuplicated), (copy(B), BatchDuplicated),
            (Wg, Const), (Wh, Const);
            rng = MersenneTwister(1234), fdm = central_fdm(5, 1; max_range = 1.0e-6),
        )
    end

    @testset "static OOP BatchDuplicated forward" begin
        test_forward(
            klein_map_static_weighted, BatchDuplicated,
            (As, BatchDuplicated), (Bs, BatchDuplicated),
            (Wgs, Const), (Whs, Const), (Val(n_x), Const);
            rng = MersenneTwister(5678), fdm = central_fdm(5, 1; max_range = 1.0e-6),
        )
    end

    @testset "OOP reverse" begin
        test_reverse(
            klein_map_weighted, Active,
            (copy(A), Duplicated), (copy(B), Duplicated), (Wg, Const), (Wh, Const);
            rng = MersenneTwister(2345), fdm = central_fdm(5, 1; max_range = 1.0e-6),
        )
    end

    @testset "in-place BatchDuplicated forward" begin
        test_forward(
            klein_map_pair!, Const,
            (zeros(n_y, n_x), BatchDuplicated),
            (zeros(n_x, n_x), BatchDuplicated),
            (copy(A), BatchDuplicated),
            (copy(B), BatchDuplicated);
            rng = MersenneTwister(3456),
        )
    end

    @testset "in-place reverse" begin
        test_reverse(
            klein_map_inplace_weighted, Active,
            (copy(A), Duplicated), (copy(B), Duplicated),
            (Wg, Const), (Wh, Const), (n_x, Const), (n_y, Const);
            rng = MersenneTwister(4567), fdm = central_fdm(5, 1; max_range = 1.0e-4),
        )
    end
end
