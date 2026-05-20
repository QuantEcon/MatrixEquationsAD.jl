using Enzyme: Active, Const, Duplicated, Enzyme, Forward, Reverse, autodiff
using ForwardDiff: ForwardDiff, Dual
using LinearAlgebra: I
using MatrixEquationsAD
using Test

# Regression for https://github.com/QuantEcon/MatrixEquationsAD.jl/issues/2.
#
# The FVGQ20-style 38×38 pencil has ~28 diagonal blocks whose generalized
# eigenvalue is exactly zero (DSGE static-equation rows). Off-diagonal block
# pairs that share λ produce a singular per-block Sylvester matrix M inside
# `ordqz_tangent!`, which used to throw `SingularException`. We now fall back
# to a pseudoinverse: the per-block Ω entries are not uniquely defined at
# coincident generalized eigenvalues — there is no smooth tangent there — but
# the pseudoinverse returns a minimum-norm finite value rather than throwing,
# which is what downstream HMC/optimisation callers want.

include(joinpath(@__DIR__, "fvgq_ordqz_fixture.jl"))

struct _FvgqTag end
const _FvgqDual = Dual{_FvgqTag, Float64, 1}

function _promote_to_dual(M, dM)
    out = Matrix{_FvgqDual}(undef, size(M))
    for i in eachindex(M)
        out[i] = Dual{_FvgqTag}(M[i], ForwardDiff.Partials((dM[i],)))
    end
    return out
end

@testset "FVGQ 38x38 ordqz_tangent! regression (issue #2)" begin
    A = fvgq_ordqz_problem_A()
    B = fvgq_ordqz_problem_B()
    @test size(A) == (38, 38)
    @test size(B) == (38, 38)

    # Primal: gges! and ordqz! both succeed on the same inputs.
    S = similar(A); T = similar(A); Q = similar(A); Z = similar(A)
    r = MatrixEquationsAD.gges!(
        S, T, Q, Z, A, B; select = :ed, criterium = (1 - 1.0e-6)^2,
    )
    @test r.sdim == 14
    @test A ≈ Q * S * Z'
    @test B ≈ Q * T * Z'

    Sp = similar(A); Tp = similar(A); Qp = similar(A); Zp = similar(A)
    sdim = MatrixEquationsAD.ordqz!(Sp, Tp, Qp, Zp, A, B, :bk; threshold = 1.0e-6)
    @test sdim == 14

    # Dual path: must not throw and must produce finite Dual outputs. The
    # primal recombination is preserved exactly (any consistent Schur form
    # gives Q*S*Z' = A); only the tangent recombination is lossy because
    # ordqz_tangent! zeroes the strict-subdiagonal entries of dS/dT (a
    # constraint required to keep the Dual Schur form upper-quasi-triangular).
    for direction_label in ("identity dA", "structured dA")
        dA = direction_label == "identity dA" ? Matrix{Float64}(I, size(A)) : begin
            d = zeros(size(A))
            d[1, 1] = 0.7; d[3, 5] = -0.4; d[10, 12] = 0.9; d[20, 25] = 0.3
            d
        end
        dB = zeros(size(B))
        A_d = _promote_to_dual(A, dA)
        B_d = _promote_to_dual(B, dB)

        @testset "gges! (Dual, $direction_label)" begin
            Sd = similar(A_d); Td = similar(A_d); Qd = similar(A_d); Zd = similar(A_d)
            @test_nowarn MatrixEquationsAD.gges!(
                Sd, Td, Qd, Zd, A_d, B_d; select = :ed, criterium = (1 - 1.0e-6)^2,
            )
            for X in (Sd, Td, Qd, Zd)
                @test all(isfinite, ForwardDiff.value.(X))
                @test all(isfinite, [ForwardDiff.partials(x, 1) for x in X])
            end
            # Primal recombination unchanged.
            @test ForwardDiff.value.(Qd * Sd * Zd') ≈ A  atol = 1.0e-8
            @test ForwardDiff.value.(Qd * Td * Zd') ≈ B  atol = 1.0e-8
        end

        @testset "ordqz! (Dual, $direction_label)" begin
            Sd = similar(A_d); Td = similar(A_d); Qd = similar(A_d); Zd = similar(A_d)
            @test_nowarn MatrixEquationsAD.ordqz!(
                Sd, Td, Qd, Zd, A_d, B_d, :bk; threshold = 1.0e-6,
            )
            for X in (Sd, Td, Qd, Zd)
                @test all(isfinite, ForwardDiff.value.(X))
                @test all(isfinite, [ForwardDiff.partials(x, 1) for x in X])
            end
            @test ForwardDiff.value.(Qd * Sd * Zd') ≈ A  atol = 1.0e-8
            @test ForwardDiff.value.(Qd * Td * Zd') ≈ B  atol = 1.0e-8
        end
    end

end
