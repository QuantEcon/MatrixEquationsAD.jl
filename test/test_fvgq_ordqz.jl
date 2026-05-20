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
# `ordqz_tangent!`. The package's escape hatch is the `regularize_A` kwarg
# on `ordqz` / `ordqz!` — passing `δ > 0` factors `(A + δI, B)` instead of
# `(A, B)`, breaking eigenvalue coincidence at the problem level.
#
# Important caveat: a uniform diagonal shift only separates eigenvalues when
# the corresponding T_ii entries differ across blocks. The FVGQ pencil's
# duplicate-zero blocks all have T_ii ≈ 1, so they shift in unison and the
# tangent stays non-smooth even with `regularize_A > 0`. The kwarg works
# (the primal is the Schur form of the perturbed pencil), but does not by
# itself rescue FVGQ-style fully-degenerate cases. Callers that need finite
# tangents on those problems must use a non-uniform perturbation (e.g.
# `A + δ * randn(n, n)`) outside the package.

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

@testset "FVGQ 38x38 ordqz regression (issue #2)" begin
    A = fvgq_ordqz_problem_A()
    B = fvgq_ordqz_problem_B()
    @test size(A) == (38, 38)
    @test size(B) == (38, 38)

    # Primal: succeeds on the unregularized pencil.
    Sp = similar(A); Tp = similar(A); Qp = similar(A); Zp = similar(A)
    sdim = MatrixEquationsAD.ordqz!(Sp, Tp, Qp, Zp, A, B, :bk; threshold = 1.0e-6)
    @test sdim == 14
    @test A ≈ Qp * Sp * Zp'
    @test B ≈ Qp * Tp * Zp'

    # Without regularize_A, the Dual tangent path can produce non-finite
    # values at coincident eigenvalues (no smooth derivative exists). The
    # primal recombination is still exact.
    let
        dA = Matrix{Float64}(I, size(A))
        dB = zeros(size(B))
        A_d = _promote_to_dual(A, dA); B_d = _promote_to_dual(B, dB)
        Sd = similar(A_d); Td = similar(A_d); Qd = similar(A_d); Zd = similar(A_d)
        MatrixEquationsAD.ordqz!(Sd, Td, Qd, Zd, A_d, B_d, :bk; threshold = 1.0e-6)
        @test ForwardDiff.value.(Qd * Sd * Zd') ≈ A  atol = 1.0e-8
        partials_S = [ForwardDiff.partials(x, 1) for x in Sd]
        @test any(!isfinite, partials_S)
    end

    # regularize_A semantics: primal must equal Schur of the perturbed
    # pencil (A + δI, B), regardless of whether δ is large enough to break
    # all coincidences (it isn't, for FVGQ -- see file header).
    @testset "regularize_A primal == Schur(A + δI, B)" begin
        for δ in (1.0e-6, 1.0e-3, 0.1)
            r_kw = MatrixEquationsAD.ordqz(
                A, B, :bk; threshold = 1.0e-6, regularize_A = δ,
            )
            r_mn = MatrixEquationsAD.ordqz(
                A + δ * Matrix{Float64}(I, size(A)), B, :bk; threshold = 1.0e-6,
            )
            @test r_kw.S ≈ r_mn.S
            @test r_kw.T ≈ r_mn.T
            @test r_kw.Q ≈ r_mn.Q
            @test r_kw.Z ≈ r_mn.Z
            @test r_kw.sdim == r_mn.sdim
            @test r_kw.Q * r_kw.S * r_kw.Z' ≈ A + δ * Matrix{Float64}(I, size(A))
        end
    end

    # regularize_A also threads through the Dual specialization: the primal
    # equals Schur of the perturbed pencil at the Dual level.
    @testset "regularize_A through ForwardDiff Dual" begin
        δ = 1.0e-4
        dA = Matrix{Float64}(I, size(A))
        dB = zeros(size(B))
        A_d = _promote_to_dual(A, dA); B_d = _promote_to_dual(B, dB)
        r_d = MatrixEquationsAD.ordqz(
            A_d, B_d, :bk; threshold = 1.0e-6, regularize_A = δ,
        )
        expected_primal = A + δ * Matrix{Float64}(I, size(A))
        @test ForwardDiff.value.(r_d.Q * r_d.S * r_d.Z') ≈ expected_primal  atol = 1.0e-6
    end
end
