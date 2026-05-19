using Enzyme: BatchDuplicated, Duplicated, Forward, autodiff
using ForwardDiff
using LinearAlgebra: Symmetric, issymmetric
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

    dual_C_wrapper = Symmetric(dual_C_sym)
    X_wrapper = lyapd(dual_A_sym, dual_C_wrapper)
    result_wrapper = autodiff(
        Forward, (A_, C_) -> lyapd(A_, Symmetric(C_)), Duplicated,
        BatchDuplicated(A, dAs_sym), BatchDuplicated(C, dCs_sym)
    )
    dXs_wrapper = ntuple(i -> result_wrapper[1][i], Val(N))

    @test map(ForwardDiff.value, X_wrapper) ≈ lyapd(A, Symmetric(C))
    for i in 1:N
        dX = map(x -> ForwardDiff.partials(x, i), X_wrapper)
        @test dX ≈ dX'
        @test dX ≈ dXs_wrapper[i]
    end

    A_wrapper = [0.45 0.08; 0.08 0.35]
    dAs_wrapper = ntuple(Val(N)) do _
        M = 0.1 .* randn(size(A_wrapper))
        0.5 .* (M + M')
    end
    dual_A_wrapper = map(A_wrapper, dAs_wrapper...) do a, ds...
        ForwardDiff.Dual{Nothing}(a, ds...)
    end
    X_both_wrappers = lyapd(Symmetric(dual_A_wrapper), Symmetric(dual_C_sym))
    result_both_wrappers = autodiff(
        Forward, (A_, C_) -> lyapd(Symmetric(A_), Symmetric(C_)), Duplicated,
        BatchDuplicated(A_wrapper, dAs_wrapper), BatchDuplicated(C, dCs_sym)
    )
    dXs_both_wrappers = ntuple(i -> result_both_wrappers[1][i], Val(N))

    @test map(ForwardDiff.value, X_both_wrappers) ≈
        lyapd(Symmetric(A_wrapper), Symmetric(C))
    for i in 1:N
        dX = map(x -> ForwardDiff.partials(x, i), X_both_wrappers)
        @test dX ≈ dX'
        @test dX ≈ dXs_both_wrappers[i]
    end
end
