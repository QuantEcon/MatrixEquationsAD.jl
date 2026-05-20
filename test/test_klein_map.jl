# Value-comparison tests for klein_map / klein_map! against ground-truth
# (g_x, h_x) matrices committed in test/klein_map_fixtures.jl. The fixture
# file is regenerated (when needed) by test/extract_klein_map_fixtures.jl,
# which runs DifferentiablePerturbation.jl's first_order_perturbation! and
# cross-checks against DP-published spot anchors (RBC_SV + SW07).
#
# Test convention: full-matrix ≈ at atol = rtol = 1.0e-10. Heap, in-place,
# and SMatrix dispatches checked per pencil; SMatrix is skipped for n > 15
# where it stops being meaningful.

using MatrixEquationsAD: klein_map, klein_map!
using StaticArrays: SMatrix
using Test

include(joinpath(@__DIR__, "dsge_qz_fixtures.jl"))
include(joinpath(@__DIR__, "fvgq_ordqz_fixture.jl"))
include(joinpath(@__DIR__, "klein_map_fixtures.jl"))

const _KLEIN_ATOL = 1.0e-10
const _KLEIN_RTOL = 1.0e-10
const _KLEIN_THRESHOLD = 1.0e-6

function _check_klein(name::AbstractString, A::AbstractMatrix, B::AbstractMatrix, F)
    @testset "$name" begin
        # Heap OOP
        r = klein_map(A, B; threshold = _KLEIN_THRESHOLD)
        @test r.g_x ≈ F.g_x atol = _KLEIN_ATOL rtol = _KLEIN_RTOL
        @test r.h_x ≈ F.h_x atol = _KLEIN_ATOL rtol = _KLEIN_RTOL

        # In-place
        g_x = similar(F.g_x); h_x = similar(F.h_x)
        klein_map!(g_x, h_x, A, B; threshold = _KLEIN_THRESHOLD)
        @test g_x ≈ F.g_x atol = _KLEIN_ATOL rtol = _KLEIN_RTOL
        @test h_x ≈ F.h_x atol = _KLEIN_ATOL rtol = _KLEIN_RTOL

        # SMatrix (only for small problems)
        if size(A, 1) <= 15
            n = size(A, 1)
            As = SMatrix{n, n, Float64}(A)
            Bs = SMatrix{n, n, Float64}(B)
            rs = klein_map(As, Bs; threshold = _KLEIN_THRESHOLD)
            @test Matrix(rs.g_x) ≈ F.g_x atol = _KLEIN_ATOL rtol = _KLEIN_RTOL
            @test Matrix(rs.h_x) ≈ F.h_x atol = _KLEIN_ATOL rtol = _KLEIN_RTOL
        end
    end
    return nothing
end

@testset "klein_map" begin
    let
        A, B, _ = dp_rbc_first_order_pencil()
        _check_klein("rbc", A, B, KLEIN_RBC)
    end
    let
        A, B, _ = dp_rbc_sv_first_order_pencil()
        _check_klein("rbc_sv", A, B, KLEIN_RBC_SV)
    end
    let
        A, B, _ = dp_sgu_first_order_pencil()
        _check_klein("sgu", A, B, KLEIN_SGU)
    end
    let
        A = fvgq_ordqz_problem_A()
        B = fvgq_ordqz_problem_B()
        _check_klein("fvgq", A, B, KLEIN_FVGQ)
    end
    let
        A, B, _ = dp_sw07pfeifer_first_order_pencil()
        _check_klein("sw07pfeifer", A, B, KLEIN_SW07PFEIFER)
    end
end
