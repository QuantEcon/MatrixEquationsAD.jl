using Enzyme: BatchDuplicated, Duplicated, Forward, autodiff
using FiniteDifferences: central_fdm, jvp
using ForwardDiff
using LinearAlgebra: issymmetric
using MatrixEquations
using MatrixEquationsAD
using Random
using Test

@testset "lyapdkr ForwardDiff rules" begin
    Random.seed!(1357)
    N = 3
    A = [0.55 0.08; -0.04 0.42]
    C = [1.0 0.2; 0.2 0.7]
    dAs = ntuple(_ -> 0.1 .* randn(size(A)), Val(N))
    dCs = ntuple(Val(N)) do _
        M = 0.1 .* randn(size(C))
        0.5 .* (M + M')
    end

    dual_A = map(A, dAs...) do a, ds...
        ForwardDiff.Dual{Nothing}(a, ds...)
    end
    dual_C = map(C, dCs...) do c, ds...
        ForwardDiff.Dual{Nothing}(c, ds...)
    end
    X = lyapdkr(dual_A, dual_C)
    result = autodiff(
        Forward, lyapdkr, Duplicated,
        BatchDuplicated(A, dAs), BatchDuplicated(C, dCs)
    )
    dXs = ntuple(i -> result[1][i], Val(N))

    @test map(ForwardDiff.value, X) ≈ lyapdkr(A, C)
    fdm = central_fdm(5, 1)
    for i in 1:N
        dX = map(x -> ForwardDiff.partials(x, i), X)
        @test issymmetric(dX)
        @test dX ≈ dXs[i]
        dX_fdm = jvp(fdm, lyapdkr, (A, dAs[i]), (C, dCs[i]))
        @test dX ≈ dX_fdm
    end
end
