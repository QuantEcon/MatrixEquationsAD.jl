using Enzyme: Active, BatchDuplicated, Const, Duplicated, Enzyme, Forward, Reverse, autodiff
using EnzymeTestUtils: test_forward, test_reverse
using FiniteDifferences: central_fdm, grad
using ForwardDiff: ForwardDiff, Dual
using LinearAlgebra: I
using MatrixEquationsAD
using Printf
using Random
using Test

include(joinpath(@__DIR__, "dsge_qz_fixtures.jl"))

# Loss used throughout: scalar reconstruction objective exercising both
# (Q, S, Z) and (Q, T, Z). Mirrors the loss in test_enzyme_ordqz.jl so the
# AD code paths match what's covered there. `sdim_expected` is taken as an
# extra positional argument (passed as `Const` to Enzyme) — never closed over.
function ordqz_reconstruction_sum(A, B, sdim_expected)
    S = zero(A)
    T = zero(B)
    Q = zero(A)
    Z = zero(A)
    sdim = ordqz!(S, T, Q, Z, A, B, :bk; threshold = 1.0e-6)
    scale = sdim == sdim_expected ? one(eltype(A)) : -one(eltype(A))
    return scale * (sum(abs2, Q * S * Z') + 0.7 * sum(abs2, Q * T * Z'))
end

# EnzymeTestUtils.test_forward / test_reverse pass workspace buffers
# positionally; these mutating wrappers match the (S, T, Q, Z, A, B) shape.
function ordqz_mutating_sum!(S, T, Q, Z, A, B, sdim_expected)
    sdim = ordqz!(S, T, Q, Z, A, B, :bk; threshold = 1.0e-6)
    scale = sdim == sdim_expected ? one(eltype(A)) : -one(eltype(A))
    return scale * (sum(abs2, Q * S * Z') + 0.7 * sum(abs2, Q * T * Z'))
end

# FD helpers: closures are fine here — they're consumed by FiniteDifferences,
# not by Enzyme.
function fd_perturbation_in_A(loss, x_vec, A_size, B, sdim_expected)
    return loss(reshape(x_vec, A_size), B, sdim_expected)
end

function fd_perturbation_in_B(loss, A, x_vec, B_size, sdim_expected)
    return loss(A, reshape(x_vec, B_size), sdim_expected)
end

function fd_grad_AB(loss, A, B, sdim_expected, fdm)
    A_size = size(A); B_size = size(B)
    gA = grad(fdm, x -> fd_perturbation_in_A(loss, x, A_size, B, sdim_expected), vec(A))[1]
    gB = grad(fdm, x -> fd_perturbation_in_B(loss, A, x, B_size, sdim_expected), vec(B))[1]
    return reshape(gA, A_size), reshape(gB, B_size)
end

function enzyme_reverse_AB(loss, A, B, sdim_expected)
    dA = zero(A)
    dB = zero(B)
    autodiff(
        Reverse, loss, Active,
        Duplicated(copy(A), dA), Duplicated(copy(B), dB), Const(sdim_expected),
    )
    return dA, dB
end

# Manual ForwardDiff Dual round-trip: confirm value(Q*S*Z') ≈ A so the primal
# path remains exact on these matrices (orthogonal to gradient correctness).
struct DsgeTag end

function promote_to_dsge_dual(M, dM)
    out = Matrix{Dual{DsgeTag, Float64, 1}}(undef, size(M))
    for i in eachindex(M)
        out[i] = Dual{DsgeTag}(M[i], ForwardDiff.Partials((dM[i],)))
    end
    return out
end

function report_gap(label, dA_ad, dA_fd, dB_ad, dB_fd)
    gap_A = maximum(abs.(dA_ad .- dA_fd))
    gap_B = maximum(abs.(dB_ad .- dB_fd))
    scale_A = max(maximum(abs.(dA_fd)), 1.0e-300)
    scale_B = max(maximum(abs.(dB_fd)), 1.0e-300)
    @printf "  %-22s  maxabs(dA-FD)=%.3e  rel=%.3e  maxabs(dB-FD)=%.3e  rel=%.3e\n" label gap_A (gap_A / scale_A) gap_B (gap_B / scale_B)
    return gap_A, gap_B
end

function run_dsge_qz_ad_fixture(name, A, B, sdim_expected)
    fdm = central_fdm(5, 1; max_range = 1.0e-3)
    rng = Random.MersenneTwister(1234)

    @testset "$name" begin
        # --- Primal sanity ---
        S = zero(A); T = zero(A); Q = zero(A); Z = zero(A)
        sdim = ordqz!(S, T, Q, Z, A, B, :bk; threshold = 1.0e-6)
        @test sdim == sdim_expected
        @test Q * S * Z' ≈ A  atol = 1.0e-8
        @test Q * T * Z' ≈ B  atol = 1.0e-8

        # --- ForwardDiff Dual round-trip on identity perturbation ---
        dA_dir = Matrix{Float64}(I, size(A))
        dB_dir = zeros(size(B))
        A_d = promote_to_dsge_dual(A, dA_dir)
        B_d = promote_to_dsge_dual(B, dB_dir)
        r_d = ordqz(A_d, B_d, :bk; threshold = 1.0e-6)
        @test ForwardDiff.value.(r_d.Q * r_d.S * r_d.Z') ≈ A  atol = 1.0e-7
        @test ForwardDiff.value.(r_d.Q * r_d.T * r_d.Z') ≈ B  atol = 1.0e-7

        # --- Enzyme reverse vs FD element-wise ---
        println("[$name] element-wise Enzyme reverse vs FD:")
        dA_ad, dB_ad = enzyme_reverse_AB(ordqz_reconstruction_sum, A, B, sdim_expected)
        dA_fd, dB_fd = fd_grad_AB(ordqz_reconstruction_sum, A, B, sdim_expected, fdm)
        gap_A_o, gap_B_o = report_gap("ordqz reverse", dA_ad, dA_fd, dB_ad, dB_fd)

        # Soft assertion: Phase 2 is a measurement, not a pass/fail gate.
        @test isfinite(gap_A_o)
        @test isfinite(gap_B_o)

        # --- EnzymeTestUtils.test_forward / test_reverse on ordqz ---
        # sdim_expected threaded through as Const, never closed over.
        test_forward(
            ordqz_mutating_sum!, Const,
            (zero(A), Duplicated), (zero(B), Duplicated),
            (zero(A), Duplicated), (zero(A), Duplicated),
            (copy(A), Duplicated), (copy(B), Duplicated),
            (sdim_expected, Const);
            rng, fdm,
        )
        test_forward(
            ordqz_mutating_sum!, Const,
            (zero(A), BatchDuplicated), (zero(B), BatchDuplicated),
            (zero(A), BatchDuplicated), (zero(A), BatchDuplicated),
            (copy(A), BatchDuplicated), (copy(B), BatchDuplicated),
            (sdim_expected, Const);
            rng, fdm,
        )
        test_reverse(
            ordqz_reconstruction_sum, Active,
            (copy(A), Duplicated), (copy(B), Duplicated), (sdim_expected, Const);
            rng, fdm,
        )
    end
    return nothing
end

function run_dsge_qz_ad_phase2()
    @testset "DSGE QZ AD verification (Phase 2)" begin
        let (A, B, n_x) = dp_rbc_first_order_pencil()
            run_dsge_qz_ad_fixture("RBC", A, B, n_x)
        end
        let (A, B, n_x) = dp_rbc_sv_first_order_pencil()
            run_dsge_qz_ad_fixture("RBC_SV", A, B, n_x)
        end
        let (A, B, n_x) = dp_sgu_first_order_pencil()
            run_dsge_qz_ad_fixture("SGU", A, B, n_x)
        end
    end
    return nothing
end

run_dsge_qz_ad_phase2()
