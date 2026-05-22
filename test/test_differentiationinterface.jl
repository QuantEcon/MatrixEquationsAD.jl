using DifferentiationInterface: AutoEnzyme, AutoForwardDiff, gradient
using Enzyme
using ForwardDiff
using MatrixEquationsAD
using Test

include(joinpath(@__DIR__, "example_matrices", "rbc.jl"))
using .RBCExampleMatrices: rbc_first_order_assembly

const RBC_P = [0.5, 0.95, 0.2, 0.02, 0.01, 0.01]

# Pipeline: parameters → pencil → Klein policy → scalar summary. Used as the
# common loss for cross-backend AD checks below.
function policy_summary(p)
    A, B, _ = rbc_first_order_assembly(p)
    r = klein_map(A, B; threshold = 1.0e-6)
    return sum(r.g_x) + sum(r.h_x)
end

# Pipeline: parameters → pencil → policy → stationary covariance (lyapdkr).
# The TFP shock is the only innovation, so Q = [0 0; 0 σ²].
function stationary_capital_variance(p)
    A, B, _ = rbc_first_order_assembly(p)
    r = klein_map(A, B; threshold = 1.0e-6)
    Q = [0.0 0.0; 0.0 p[5]^2]
    V = lyapdkr(r.h_x, Q)
    return V[1, 1]
end

@testset "DifferentiationInterface integration" begin
    @testset "Klein policy summary" begin
        fd = AutoForwardDiff()
        enz = AutoEnzyme(mode = Enzyme.Reverse)

        ∇fd = gradient(policy_summary, fd, RBC_P)
        ∇enz = gradient(policy_summary, enz, RBC_P)

        @test length(∇fd) == length(RBC_P)
        @test ∇fd ≈ ∇enz atol = 1.0e-9 rtol = 1.0e-9
        # σ (p[5]) and Ω_1 (p[6]) do not enter the pencil
        @test ∇fd[5] == 0.0
        @test ∇fd[6] == 0.0
    end

    @testset "Stationary capital variance (klein_map ∘ lyapdkr)" begin
        fd = AutoForwardDiff()
        enz = AutoEnzyme(mode = Enzyme.Reverse)

        ∇fd = gradient(stationary_capital_variance, fd, RBC_P)
        ∇enz = gradient(stationary_capital_variance, enz, RBC_P)

        @test ∇fd ≈ ∇enz atol = 1.0e-9 rtol = 1.0e-9
        @test ∇fd[6] == 0.0                 # Ω_1 still does not enter
        @test ∇fd[5] > 0                    # Var(k) increases with σ
        @test ∇fd[3] > 0                    # Var(k) increases with TFP persistence ρ
    end
end
