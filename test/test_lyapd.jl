# Primal value tests for MatrixEquations.lyapd over the DSGE fixture
# matrices bundled in dp_<model>_first_order_inputs(). Each fixture exposes
# (h_x, B_shock) so the stationary-covariance Lyapunov equation
#   P = h_x * P * h_x' + B_shock * B_shock'
# can be solved without any side computation. AD-side coverage lives in
# test_forwarddiff_dlyap.jl / test_enzyme_dlyap.jl; this file is the primal
# value gate against realistic Schur pencils.

using LinearAlgebra: I, issymmetric, norm
using MatrixEquations: lyapd
using Test

include(joinpath(@__DIR__, "example_matrices", "rbc.jl"))
include(joinpath(@__DIR__, "example_matrices", "sgu.jl"))
include(joinpath(@__DIR__, "example_matrices", "fvgq.jl"))
include(joinpath(@__DIR__, "example_matrices", "sw07.jl"))

@testset "lyapd primal — DSGE fixtures" begin
    fixtures = (
        ("rbc",         RBCExampleMatrices.dp_rbc_first_order_inputs()),
        ("rbc_sv",      RBCExampleMatrices.dp_rbc_sv_first_order_inputs()),
        ("sgu",         SGUExampleMatrices.dp_sgu_first_order_inputs()),
        ("fvgq",        FVGQExampleMatrices.dp_fvgq_first_order_inputs()),
        ("sw07pfeifer", SW07ExampleMatrices.dp_sw07pfeifer_first_order_inputs()),
    )
    for (name, fo) in fixtures
        (; h_x, B_shock, n_x) = fo
        @testset "$name" begin
            BBT = B_shock * transpose(B_shock)
            P = lyapd(h_x, BBT)
            @test size(P) == (n_x, n_x)
            @test all(isfinite, P)
            # Discrete Lyapunov residual: h_x * P * h_x' - P + B_shock B_shock' = 0.
            @test norm(h_x * P * transpose(h_x) - P + BBT) <=
                1.0e-8 * max(1.0, norm(P))
            # Stationary covariance must be (numerically) symmetric.
            @test norm(P - transpose(P)) <= 1.0e-8 * max(1.0, norm(P))
        end
    end
end
