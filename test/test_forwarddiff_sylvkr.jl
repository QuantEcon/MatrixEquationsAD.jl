using Enzyme: BatchDuplicated, Duplicated, Forward, autodiff
using FiniteDifferences: central_fdm, jvp
using ForwardDiff
using MatrixEquations
using Random
using Test

@testset "gsylvkr ForwardDiff rules" begin
    Random.seed!(8765)
    N = 3
    A = [4.0 0.2 -0.1; -0.3 3.7 0.4; 0.1 -0.2 3.5]
    B = [2.8 -0.3; 0.2 3.1]
    C = [0.5 0.1 -0.2; 0.0 0.7 0.3; -0.1 0.2 0.6]
    D = [0.9 0.2; -0.4 0.8]
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
    X = gsylvkr(
        dual_A, dual_B, dual_C, dual_D, dual_E
    )
    result = autodiff(
        Forward, gsylvkr, Duplicated,
        BatchDuplicated(A, dAs), BatchDuplicated(B, dBs),
        BatchDuplicated(C, dCs), BatchDuplicated(D, dDs),
        BatchDuplicated(E, dEs)
    )
    dXs = ntuple(i -> result[1][i], Val(N))

    @test map(ForwardDiff.value, X) ≈ gsylvkr(A, B, C, D, E)
    fdm = central_fdm(5, 1)
    for i in 1:N
        @test map(x -> ForwardDiff.partials(x, i), X) ≈ dXs[i]
        dX_fdm = jvp(
            fdm, gsylvkr,
            (A, dAs[i]), (B, dBs[i]), (C, dCs[i]), (D, dDs[i]), (E, dEs[i])
        )
        @test map(x -> ForwardDiff.partials(x, i), X) ≈ dX_fdm
    end
end
