using FiniteDifferences: central_fdm, jvp
using ForwardDiff
using LinearAlgebra: dot, issymmetric
using MatrixEquations
using MatrixEquationsAD
using Random
using Test

function ared_vec(A, B, R, Q)
    X, _, F = ared(A, B, R, Q)
    return [vec(X); vec(F)]
end

@testset "ared ForwardDiff rules" begin
    rng = Random.MersenneTwister(1122)
    A, B, R, Q = quantecon_kalman_ared_problem()
    N = 3
    dAs = ntuple(_ -> 0.1 .* randn(rng, size(A)), Val(N))
    dBs = ntuple(_ -> 0.1 .* randn(rng, size(B)), Val(N))
    dRs = ntuple(_ -> random_symmetric_direction(rng, R), Val(N))
    dQs = ntuple(_ -> random_symmetric_direction(rng, Q), Val(N))

    dual_A = map(A, dAs...) do a, ds...
        ForwardDiff.Dual{Nothing}(a, ds...)
    end
    dual_B = map(B, dBs...) do b, ds...
        ForwardDiff.Dual{Nothing}(b, ds...)
    end
    dual_R = map(R, dRs...) do r, ds...
        ForwardDiff.Dual{Nothing}(r, ds...)
    end
    dual_Q = map(Q, dQs...) do q, ds...
        ForwardDiff.Dual{Nothing}(q, ds...)
    end

    X, evals, F = ared(dual_A, dual_B, dual_R, dual_Q)
    X0, evals0, F0 = ared(A, B, R, Q)
    @test map(ForwardDiff.value, X) ≈ X0
    @test evals == evals0
    @test map(ForwardDiff.value, F) ≈ F0

    fdm = central_fdm(5, 1; max_range = 1.0e-4)
    for i in 1:N
        dX = map(x -> ForwardDiff.partials(x, i), X)
        dF = map(x -> ForwardDiff.partials(x, i), F)
        @test dX ≈ dX'
        fd = jvp(
            fdm, ared_vec, (A, dAs[i]), (B, dBs[i]), (R, dRs[i]), (Q, dQs[i])
        )
        @test [vec(dX); vec(dF)] ≈ fd
    end
end
