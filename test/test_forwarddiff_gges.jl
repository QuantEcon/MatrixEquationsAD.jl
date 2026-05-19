using FiniteDifferences: central_fdm, grad
using ForwardDiff
using LinearAlgebra: dot
using MatrixEquationsAD
using Random
using Test

const gges_forwarddiff_fdm = central_fdm(5, 1)

function gges_forwarddiff_problem()
    A = [1.6 0.2 0.1; 0.0 0.35 -0.1; 0.0 0.0 1.9]
    B = [1.0 0.1 0.0; 0.0 1.2 0.2; 0.0 0.0 0.8]
    return A, B
end

function gges_forwarddiff_sum(A, B, expected; criterium = (1 - 1.0e-6)^2)::eltype(A)
    S = zero(A)
    T = zero(B)
    Q = zero(A)
    Z = zero(A)
    result = gges!(S, T, Q, Z, A, B; select = :ed, criterium)
    scale = result.n_explosive == expected ? one(eltype(A)) : -one(eltype(A))
    return scale * (sum(abs2, Q * S * Z') + 0.7 * sum(abs2, Q * T * Z'))
end

function gges_forwarddiff_directional(A, B, dA, dB, expected; criterium = (1 - 1.0e-6)^2)
    gA = grad(
        gges_forwarddiff_fdm,
        x -> gges_forwarddiff_sum(reshape(x, size(A)), B, expected; criterium),
        vec(A)
    )[1]
    gB = grad(
        gges_forwarddiff_fdm,
        x -> gges_forwarddiff_sum(A, reshape(x, size(B)), expected; criterium),
        vec(B)
    )[1]
    return dot(gA, vec(dA)) + dot(gB, vec(dB))
end

@testset "gges ForwardDiff rules" begin
    Random.seed!(5678)
    N = 3
    A, B = gges_forwarddiff_problem()
    criterium = (1 - 1.0e-6)^2
    dAs = ntuple(_ -> 0.1 .* randn(size(A)), Val(N))
    dBs = ntuple(_ -> 0.1 .* randn(size(B)), Val(N))
    A_dual = map(A, dAs...) do a, da...
        ForwardDiff.Dual{Nothing}(a, da...)
    end
    B_dual = map(B, dBs...) do b, db...
        ForwardDiff.Dual{Nothing}(b, db...)
    end

    y = gges_forwarddiff_sum(A_dual, B_dual, 2; criterium)

    @test ForwardDiff.value(y) ≈ gges_forwarddiff_sum(A, B, 2; criterium)
    for i in 1:N
        @test ForwardDiff.partials(y, i) ≈
            gges_forwarddiff_directional(A, B, dAs[i], dBs[i], 2; criterium)
    end

    A_dp, B_dp, expected = dp_rbc_ordqz_problem()
    criterium_dp = (1 - dp_ordqz_threshold)^2
    dAs_dp = ntuple(_ -> 0.1 .* randn(size(A_dp)), Val(N))
    dBs_dp = ntuple(_ -> 0.1 .* randn(size(B_dp)), Val(N))
    A_dp_dual = map(A_dp, dAs_dp...) do a, da...
        ForwardDiff.Dual{Nothing}(a, da...)
    end
    B_dp_dual = map(B_dp, dBs_dp...) do b, db...
        ForwardDiff.Dual{Nothing}(b, db...)
    end
    result_dp = gges!(
        zero(A_dp_dual), zero(B_dp_dual), zero(A_dp_dual), zero(A_dp_dual),
        A_dp_dual, B_dp_dual; select = :ed, criterium = criterium_dp
    )
    @test result_dp.n_explosive == expected
end
