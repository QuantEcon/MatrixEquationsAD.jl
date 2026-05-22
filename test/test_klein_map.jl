# Value-comparison tests for klein_map / klein_map! against the ground-truth
# (g_x, h_x) matrices bundled inside each model's
# `dp_<model>_first_order_inputs()` fixture (test/example_matrices/<model>.jl).
# The fixture function is internally consistent:
#   klein_map(A_schur, B_schur; threshold = 1.0e-6) reproduces (g_x, h_x).
#
# Test convention: full-matrix ≈ at atol = rtol = 1.0e-10. Heap, in-place,
# SMatrix heap fallback, and explicit Val-sized SMatrix output are checked per
# model; SMatrix is skipped for n > 15 where it stops being meaningful.

using MatrixEquationsAD: klein_map, klein_map!
using LinearAlgebra: I, norm
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
