using Enzyme: BatchDuplicated, Duplicated, Forward, autodiff
using ForwardDiff
using LinearAlgebra: I
using MatrixEquations
using Random
using Test

@testset "gsylv ForwardDiff rules" begin
    Random.seed!(4321)
    N = 3
    A = Matrix([4.0 0.1 0.0; -0.2 3.6 0.3; 0.1 0.0 3.8])
    C = Matrix(0.2I, 3, 3)
    B = Matrix([3.0 0.2; -0.1 2.7])
    D = Matrix(0.3I, 2, 2)
    E = [1.0 -0.4; 0.3 0.8; -0.2 0.5]

    dAs = ntuple(_ -> 0.1 .* randn(size(A)), Val(N))
    dBs = ntuple(_ -> 0.1 .* randn(size(B)), Val(N))
    dCs = ntuple(_ -> 0.1 .* randn(size(C)), Val(N))
    dDs = ntuple(_ -> 0.1 .* randn(size(D)), Val(N))
    dEs = ntuple(_ -> 0.1 .* randn(size(E)), Val(N))

    dual_A = map(A, dAs...) do a, ds...
        ForwardDiff.Dual{Nothing}(a, ds...)
    end
    dual_B = map(B, dBs...) do b, ds...
        ForwardDiff.Dual{Nothing}(b, ds...)
    end
    dual_C = map(C, dCs...) do c, ds...
        ForwardDiff.Dual{Nothing}(c, ds...)
    end
    dual_D = map(D, dDs...) do d, ds...
        ForwardDiff.Dual{Nothing}(d, ds...)
    end
    dual_E = map(E, dEs...) do e, ds...
        ForwardDiff.Dual{Nothing}(e, ds...)
    end
    X = gsylv(
        dual_A, dual_B, dual_C, dual_D, dual_E
    )
    result = autodiff(
        Forward, gsylv, Duplicated,
        BatchDuplicated(A, dAs), BatchDuplicated(B, dBs),
        BatchDuplicated(C, dCs), BatchDuplicated(D, dDs),
        BatchDuplicated(E, dEs)
    )
    dXs = ntuple(i -> result[1][i], Val(N))

    @test map(ForwardDiff.value, X) ≈ gsylv(A, B, C, D, E)
    for i in 1:N
        @test map(x -> ForwardDiff.partials(x, i), X) ≈ dXs[i]
    end
end
