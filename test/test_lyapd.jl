# Primal + AD tests for MatrixEquations.lyapd / MatrixEquationsAD.lyapd!.
#
# Primal coverage runs over the unified DSGE fixtures
# `dp_<model>_first_order_inputs()`, exercising both the OOP and the
# in-place form. The AD coverage runs at small / medium pencil sizes —
# small enough that `EnzymeTestUtils.test_forward / test_reverse` and the
# ForwardDiff chunked-`Dual` round-trip stay inside numerical tolerance,
# but big enough to exercise the multi-RHS Schur reuse path.
#
# Single-`A` AD coverage at `n = 2` (random, non-fixture) lives in
# `test_forwarddiff_lyapd_small.jl` and `test_enzyme_lyapd_small.jl`; this file
# focuses on the fixture-driven medium-size pencils.

using Enzyme:
    Active, BatchDuplicated, Const, Duplicated, Enzyme, Forward, Reverse, autodiff
using EnzymeTestUtils: test_forward, test_reverse
using FiniteDifferences: central_fdm
using ForwardDiff: ForwardDiff, Dual, Partials
using LinearAlgebra: I, Symmetric, dot, issymmetric, norm, opnorm
using MatrixEquations: lyapd
using MatrixEquationsAD
using Random: MersenneTwister
using Test

include(joinpath(@__DIR__, "example_matrices", "rbc.jl"))
include(joinpath(@__DIR__, "example_matrices", "sgu.jl"))
include(joinpath(@__DIR__, "example_matrices", "fvgq.jl"))
include(joinpath(@__DIR__, "example_matrices", "sw07.jl"))

# Test tier flag — see test/runtests.jl.
const _RUN_SLOW_TESTS = get(ENV, "RUN_SLOW_TESTS", "false") == "true"

# Scalar-loss wrappers reused across the AD lanes (Enzyme reverse on a
# matrix-valued output isn't well-defined; test_reverse here takes a
# weighted scalar functional).
function lyapd_weighted(A, C, W)::Float64
    return dot(W, lyapd(A, C))
end

function lyapd_inplace_weighted!(X, A, C, W)::Float64
    lyapd!(X, A, C)
    return dot(W, X)
end

@testset "lyapd / lyapd! primal — DSGE fixtures" begin
    fixtures = (
        ("rbc", RBCExampleMatrices.rbc_first_order_inputs()),
        ("rbc_sv", RBCExampleMatrices.rbc_sv_first_order_inputs()),
        ("sgu", SGUExampleMatrices.sgu_first_order_inputs()),
        ("fvgq", FVGQExampleMatrices.fvgq_first_order_inputs()),
        ("sw07pfeifer", SW07ExampleMatrices.sw07pfeifer_first_order_inputs()),
    )
    for (name, fo) in fixtures
        (; h_x, B_shock, n_x) = fo
        @testset "$name" begin
            BBT = B_shock * transpose(B_shock)

            # OOP
            P = lyapd(h_x, BBT)
            @test size(P) == (n_x, n_x)
            @test all(isfinite, P)
            @test norm(h_x * P * transpose(h_x) - P + BBT) <=
                1.0e-8 * max(1.0, norm(P))
            @test norm(P - transpose(P)) <= 1.0e-8 * max(1.0, norm(P))

            # In-place writes into a caller-supplied buffer; result must
            # match the OOP solve through the same Schur path.
            Xip = zeros(n_x, n_x)
            @test lyapd!(Xip, h_x, BBT) === nothing
            @test Xip ≈ P atol = 1.0e-10 rtol = 1.0e-10

            # Symmetric RHS exercises the lyapds!-based dispatch.
            Xip_sym = zeros(n_x, n_x)
            @test lyapd!(Xip_sym, h_x, Symmetric(BBT)) === nothing
            @test Xip_sym ≈ P atol = 1.0e-10 rtol = 1.0e-10

            @test_throws DimensionMismatch lyapd!(
                zeros(n_x + 1, n_x + 1), h_x, BBT,
            )
        end
    end
end

@testset "lyapd! primal — small random pencils" begin
    rng = MersenneTwister(11)
    for n in (2, 4, 7)
        A = 0.3 .* randn(rng, n, n)
        A ./= 1.2 * opnorm(A)
        M = randn(rng, n, n)
        C = 0.5 .* (M + M')

        X_ref = lyapd(A, C)
        X = zeros(n, n)
        @test lyapd!(X, A, C) === nothing
        @test X ≈ X_ref atol = 1.0e-12 rtol = 1.0e-12

        Xs = zeros(n, n)
        @test lyapd!(Xs, A, Symmetric(C)) === nothing
        @test Xs ≈ X_ref atol = 1.0e-12 rtol = 1.0e-12

        @test_throws DimensionMismatch lyapd!(zeros(n + 1, n + 1), A, C)
    end
end

# Random-matrix Enzyme-rule sanity at n = 3 (the multi-pencil AD lanes
# below are the more thorough coverage gate; this one keeps the small
# test_forward / test_reverse signal in case fixture sizes change).
# Gated to the slow tier; the default-tier anchor for `lyapd!` reverse
# lives in test_enzyme_lyapd_small.jl.
if _RUN_SLOW_TESTS
@testset "lyapd! Enzyme rules (random n=3)" begin
    rng = MersenneTwister(13)
    n = 3
    A = 0.3 .* randn(rng, n, n)
    A ./= 1.2 * opnorm(A)
    M = randn(rng, n, n)
    C = 0.5 .* (M + M')
    W = randn(rng, n, n)

    test_forward(
        lyapd!, Const,
        (zeros(n, n), Duplicated),
        (copy(A), Duplicated),
        (copy(C), Duplicated),
    )
    test_forward(
        lyapd!, Const,
        (zeros(n, n), BatchDuplicated),
        (copy(A), BatchDuplicated),
        (copy(C), BatchDuplicated),
    )

    test_reverse(
        lyapd_inplace_weighted!, Active,
        (zeros(n, n), Duplicated),
        (copy(A), Duplicated),
        (copy(C), Duplicated),
        (W, Const),
    )
end
end  # if _RUN_SLOW_TESTS

@testset "lyapd! ForwardDiff Dual chunk (random n=4)" begin
    rng = MersenneTwister(17)
    n = 4
    A = 0.3 .* randn(rng, n, n)
    A ./= 1.2 * opnorm(A)
    M = randn(rng, n, n)
    C = 0.5 .* (M + M')

    NCH = 4
    dA_lanes = ntuple(_ -> randn(rng, n, n), Val(NCH))
    dC_lanes = ntuple(NCH) do _
        N = randn(rng, n, n)
        0.5 .* (N + N')
    end

    A_dual = map(A, dA_lanes...) do a, ds...
        Dual{Nothing}(a, ds...)
    end
    C_dual = map(C, dC_lanes...) do c, ds...
        Dual{Nothing}(c, ds...)
    end

    X_dual = similar(A_dual)
    @test lyapd!(X_dual, A_dual, C_dual) === nothing

    X_ref = lyapd(A, C)
    @test ForwardDiff.value.(X_dual) ≈ X_ref atol = 1.0e-12

    for i in 1:NCH
        A_i = map(A, dA_lanes[i]) do a, d
            Dual{Nothing}(a, d)
        end
        C_i = map(C, dC_lanes[i]) do c, d
            Dual{Nothing}(c, d)
        end
        ref = ForwardDiff.partials.(lyapd(A_i, C_i), 1)
        @test map(x -> ForwardDiff.partials(x, i), X_dual) ≈ ref atol = 1.0e-12
    end
end

# Multi-pencil AD coverage. For each fixture below we differentiate both
# the OOP `lyapd` and the in-place `lyapd!` w.r.t. (h_x, C); all three
# backends (FD chunked `Dual`, Enzyme forward, Enzyme reverse) hit the
# same pencil. `max_range = 1.0e-5` keeps the FD perturbation small
# enough to stay inside the Schur-stability ball even at n = 7.
# ~1m EnzymeTestUtils compilation across 3 fixtures × 3 sweeps — gated.
if _RUN_SLOW_TESTS
@testset "lyapd / lyapd! AD coverage — fixtures" begin
    rng_seeds = Dict(
        "rbc" => 1001,
        "rbc_sv" => 1002,
        "sgu" => 1003,
    )
    fixtures = (
        ("rbc", RBCExampleMatrices.rbc_first_order_inputs()),
        ("rbc_sv", RBCExampleMatrices.rbc_sv_first_order_inputs()),
        ("sgu", SGUExampleMatrices.sgu_first_order_inputs()),
    )
    for (name, fo) in fixtures
        (; h_x, B_shock, n_x) = fo
        seed = rng_seeds[name]
        rng = MersenneTwister(seed)
        C = B_shock * transpose(B_shock)
        # Symmetrise so the rules' projection is a no-op on the primal.
        C = 0.5 .* (C + transpose(C))
        W = randn(rng, n_x, n_x)
        fdm = central_fdm(5, 1; max_range = 1.0e-5)

        @testset "$name — ForwardDiff Dual chunk-4" begin
            NCH = 4
            dA_lanes = ntuple(_ -> 1.0e-3 .* randn(rng, n_x, n_x), Val(NCH))
            dC_lanes = ntuple(NCH) do _
                M = 1.0e-3 .* randn(rng, n_x, n_x)
                0.5 .* (M + M')
            end

            A_dual = map(h_x, dA_lanes...) do a, ds...
                Dual{Nothing}(a, ds...)
            end
            C_dual = map(C, dC_lanes...) do c, ds...
                Dual{Nothing}(c, ds...)
            end

            X_oop = lyapd(A_dual, C_dual)
            X_ref = lyapd(h_x, C)
            @test ForwardDiff.value.(X_oop) ≈ X_ref atol = 1.0e-10

            X_ip = similar(A_dual)
            @test lyapd!(X_ip, A_dual, C_dual) === nothing
            @test ForwardDiff.value.(X_ip) ≈ X_ref atol = 1.0e-10
            for i in 1:NCH
                @test map(x -> ForwardDiff.partials(x, i), X_ip) ≈
                    map(x -> ForwardDiff.partials(x, i), X_oop) atol = 1.0e-10
            end
        end

        @testset "$name — Enzyme forward" begin
            test_forward(
                lyapd, Const,
                (copy(h_x), Duplicated), (copy(C), Duplicated);
                rng = MersenneTwister(seed + 1), fdm,
            )
            test_forward(
                lyapd, Const,
                (copy(h_x), BatchDuplicated), (copy(C), BatchDuplicated);
                rng = MersenneTwister(seed + 2), fdm,
            )
            test_forward(
                lyapd!, Const,
                (zeros(n_x, n_x), Duplicated),
                (copy(h_x), Duplicated), (copy(C), Duplicated);
                rng = MersenneTwister(seed + 3), fdm,
            )
            test_forward(
                lyapd!, Const,
                (zeros(n_x, n_x), BatchDuplicated),
                (copy(h_x), BatchDuplicated), (copy(C), BatchDuplicated);
                rng = MersenneTwister(seed + 4), fdm,
            )
        end

        @testset "$name — Enzyme reverse on scalar loss" begin
            # The output buffer of `lyapd!` is fully overwritten, so its
            # cotangent w.r.t. the initial value is exactly zero; FD-side
            # estimates pick up O(1e-9) noise at the n = 7 fixture. atol
            # absorbs that without weakening the (h_x, C) checks.
            test_reverse(
                lyapd_weighted, Active,
                (copy(h_x), Duplicated), (copy(C), Duplicated), (W, Const);
                rng = MersenneTwister(seed + 5), fdm,
                atol = 1.0e-7, rtol = 1.0e-7,
            )
            test_reverse(
                lyapd_inplace_weighted!, Active,
                (zeros(n_x, n_x), Duplicated),
                (copy(h_x), Duplicated), (copy(C), Duplicated), (W, Const);
                rng = MersenneTwister(seed + 6), fdm,
                atol = 1.0e-7, rtol = 1.0e-7,
            )
        end
    end
end
end  # if _RUN_SLOW_TESTS
