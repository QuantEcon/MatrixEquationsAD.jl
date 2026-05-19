using Enzyme: BatchDuplicated, Duplicated, Forward, autodiff
using ForwardDiff
using LinearAlgebra: issymmetric
using MatrixEquations
using Random
using Test

@testset "lyapd ForwardDiff rules" begin
    Random.seed!(1234)
    N = 3
    A = [0.55 0.08; -0.04 0.42]
    C = [1.0 0.2; 0.2 0.7]
    dAs = ntuple(_ -> 0.1 .* randn(size(A)), Val(N))
    dCs = ntuple(_ -> 0.1 .* randn(size(C)), Val(N))

    dual_A = map(A, dAs...) do a, ds...
        ForwardDiff.Dual{Nothing}(a, ds...)
    end
    dual_C = map(C, dCs...) do c, ds...
        ForwardDiff.Dual{Nothing}(c, ds...)
    end
    X = lyapd(dual_A, dual_C)
    result = autodiff(
        Forward, lyapd, Duplicated,
        BatchDuplicated(A, dAs), BatchDuplicated(C, dCs)
    )
    dXs = ntuple(i -> result[1][i], Val(N))

    @test map(ForwardDiff.value, X) ≈ lyapd(A, C)
    for i in 1:N
        @test map(x -> ForwardDiff.partials(x, i), X) ≈ dXs[i]
    end

    Random.seed!(24601)
    dAs_sym = ntuple(_ -> 0.1 .* randn(size(A)), Val(N))
    dCs_sym = ntuple(Val(N)) do _
        M = 0.1 .* randn(size(C))
        0.5 .* (M + M')
    end
    dual_A_sym = map(A, dAs_sym...) do a, ds...
        ForwardDiff.Dual{Nothing}(a, ds...)
    end
    dual_C_sym = map(C, dCs_sym...) do c, ds...
        ForwardDiff.Dual{Nothing}(c, ds...)
    end
    X_sym = lyapd(dual_A_sym, dual_C_sym)
    result_sym = autodiff(
        Forward, lyapd, Duplicated,
        BatchDuplicated(A, dAs_sym), BatchDuplicated(C, dCs_sym)
    )
    dXs_sym = ntuple(i -> result_sym[1][i], Val(N))

    @test map(ForwardDiff.value, X_sym) ≈ lyapd(A, C)
    for i in 1:N
        dX = map(x -> ForwardDiff.partials(x, i), X_sym)
        @test dX ≈ dX'
        @test dX ≈ dXs_sym[i]
    end
end
