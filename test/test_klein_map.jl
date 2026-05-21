# Value-comparison tests for klein_map / klein_map! against ground-truth
# (g_x, h_x) matrices committed in test/klein_map_fixtures.jl. The fixture
# file is regenerated (when needed) by test/extract_klein_map_fixtures.jl,
# which runs DifferentiablePerturbation.jl's first_order_perturbation! and
# cross-checks against DP-published spot anchors (RBC_SV + SW07).
#
# Test convention: full-matrix ≈ at atol = rtol = 1.0e-10. Heap, in-place,
# SMatrix heap fallback, and explicit Val-sized SMatrix output are checked per
# gschur input; SMatrix is skipped for n > 15 where it stops being meaningful.

using MatrixEquationsAD: klein_map, klein_map!
using LinearAlgebra: I, norm
using StaticArrays: SMatrix
using Test

include(joinpath(@__DIR__, "example_matrices", "rbc.jl"))
include(joinpath(@__DIR__, "example_matrices", "sgu.jl"))
include(joinpath(@__DIR__, "example_matrices", "fvgq.jl"))
include(joinpath(@__DIR__, "example_matrices", "sw07.jl"))
include(joinpath(@__DIR__, "klein_map_fixtures.jl"))

@testset "klein_map" begin
    rbc_A, rbc_B, _ = RBCExampleMatrices.dp_rbc_first_order_gschur()
    rbc_sv_A, rbc_sv_B, _ = RBCExampleMatrices.dp_rbc_sv_first_order_gschur()
    sgu_A, sgu_B, _ = SGUExampleMatrices.dp_sgu_first_order_gschur()
    fvgq_A = FVGQExampleMatrices.fvgq_klein_gschur_A()
    fvgq_B = FVGQExampleMatrices.fvgq_klein_gschur_B()
    sw07_A, sw07_B, _ = SW07ExampleMatrices.dp_sw07pfeifer_first_order_gschur()

    for (name, A, B, F) in (
            ("rbc", rbc_A, rbc_B, KleinMapFixtures.KLEIN_RBC),
            ("rbc_sv", rbc_sv_A, rbc_sv_B, KleinMapFixtures.KLEIN_RBC_SV),
            ("sgu", sgu_A, sgu_B, KleinMapFixtures.KLEIN_SGU),
            ("fvgq", fvgq_A, fvgq_B, KleinMapFixtures.KLEIN_FVGQ),
            ("sw07pfeifer", sw07_A, sw07_B, KleinMapFixtures.KLEIN_SW07PFEIFER),
        )
        @testset "$name" begin
            r = klein_map(A, B; threshold = 1.0e-6)
            @test r.g_x ≈ F.g_x atol = 1.0e-10 rtol = 1.0e-10
            @test r.h_x ≈ F.h_x atol = 1.0e-10 rtol = 1.0e-10
            G = vcat(Matrix{Float64}(I, size(r.h_x, 1), size(r.h_x, 1)), r.g_x)
            @test norm(A * G * r.h_x + B * G) <= 1.0e-8

            g_x = similar(F.g_x)
            h_x = similar(F.h_x)
            klein_map!(g_x, h_x, A, B; threshold = 1.0e-6)
            @test g_x ≈ F.g_x atol = 1.0e-10 rtol = 1.0e-10
            @test h_x ≈ F.h_x atol = 1.0e-10 rtol = 1.0e-10
            G .= vcat(Matrix{Float64}(I, size(h_x, 1), size(h_x, 1)), g_x)
            @test norm(A * G * h_x + B * G) <= 1.0e-8

            if size(A, 1) <= 15
                n = size(A, 1)
                As = SMatrix{n, n, Float64}(A)
                Bs = SMatrix{n, n, Float64}(B)
                rs = klein_map(As, Bs; threshold = 1.0e-6)
                @test rs.g_x isa Matrix{Float64}
                @test rs.h_x isa Matrix{Float64}
                @test rs.g_x ≈ F.g_x atol = 1.0e-10 rtol = 1.0e-10
                @test rs.h_x ≈ F.h_x atol = 1.0e-10 rtol = 1.0e-10

                n_x = size(F.h_x, 1)
                n_y = size(F.g_x, 1)
                rs_static = @inferred klein_map(As, Bs, Val(n_x); threshold = 1.0e-6)
                @test rs_static.g_x isa SMatrix{n_y, n_x, Float64}
                @test rs_static.h_x isa SMatrix{n_x, n_x, Float64}
                @test Matrix(rs_static.g_x) ≈ F.g_x atol = 1.0e-10 rtol = 1.0e-10
                @test Matrix(rs_static.h_x) ≈ F.h_x atol = 1.0e-10 rtol = 1.0e-10
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
