using FiniteDifferences: central_fdm, grad
using ForwardDiff
using LinearAlgebra: dot
using MatrixEquationsAD
using Random
using Test

const ordqz_forwarddiff_fdm = central_fdm(5, 1)

function ordqz_forwarddiff_problem()
    A = [1.6 0.2 0.1; 0.0 0.35 -0.1; 0.0 0.0 1.9]
    B = [1.0 0.1 0.0; 0.0 1.2 0.2; 0.0 0.0 0.8]
    return A, B
end

function ordqz_forwarddiff_sum(A, B, select, expected)::eltype(A)
    S = zero(A)
    T = zero(B)
    Q = zero(A)
    Z = zero(A)
    sdim = ordqz!(S, T, Q, Z, A, B, select)
    scale = sdim == expected ? one(eltype(A)) : -one(eltype(A))
    return scale * (sum(abs2, Q * S * Z') + 0.7 * sum(abs2, Q * T * Z'))
end

function ordqz_forwarddiff_directional(A, B, dA, dB, select, expected)
    gA = grad(
        ordqz_forwarddiff_fdm,
        x -> ordqz_forwarddiff_sum(reshape(x, size(A)), B, select, expected),
        vec(A)
    )[1]
    gB = grad(
        ordqz_forwarddiff_fdm,
        x -> ordqz_forwarddiff_sum(A, reshape(x, size(B)), select, expected),
        vec(B)
    )[1]
    return dot(gA, vec(dA)) + dot(gB, vec(dB))
end

@testset "ordqz ForwardDiff rules" begin
    Random.seed!(5678)
    N = 3
    A, B = ordqz_forwarddiff_problem()
    dAs = ntuple(_ -> 0.1 .* randn(size(A)), Val(N))
    dBs = ntuple(_ -> 0.1 .* randn(size(B)), Val(N))
    A_dual = map(A, dAs...) do a, da...
        ForwardDiff.Dual{Nothing}(a, da...)
    end
    B_dual = map(B, dBs...) do b, db...
        ForwardDiff.Dual{Nothing}(b, db...)
    end

    y = ordqz_forwarddiff_sum(
        A_dual, B_dual, qzselect_inside_unit, 1
    )

    @test ForwardDiff.value(y) ≈ ordqz_forwarddiff_sum(A, B, qzselect_inside_unit, 1)
    for i in 1:N
        @test ForwardDiff.partials(y, i) ≈
            ordqz_forwarddiff_directional(A, B, dAs[i], dBs[i], qzselect_inside_unit, 1)
    end

    A_dp, B_dp, expected = dp_rbc_ordqz_problem()
    dAs_dp = ntuple(_ -> 0.1 .* randn(size(A_dp)), Val(N))
    dBs_dp = ntuple(_ -> 0.1 .* randn(size(B_dp)), Val(N))
    A_dp_dual = map(A_dp, dAs_dp...) do a, da...
        ForwardDiff.Dual{Nothing}(a, da...)
    end
    B_dp_dual = map(B_dp, dBs_dp...) do b, db...
        ForwardDiff.Dual{Nothing}(b, db...)
    end
    @test ordqz!(
        zero(A_dp_dual), zero(B_dp_dual), zero(A_dp_dual), zero(A_dp_dual),
        A_dp_dual, B_dp_dual, dp_ordqz_select
    ) == expected
end
