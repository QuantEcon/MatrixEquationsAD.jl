using BenchmarkTools
using Enzyme: Active, BatchDuplicated, Const, Duplicated, Enzyme, Forward, Reverse,
    make_zero
using ForwardDiff: Dual
using LinearAlgebra: I, dot
using MatrixEquations: ared
using MatrixEquationsAD
using Random: randn

include(joinpath(pkgdir(MatrixEquationsAD), "test", "example_matrices", "sgu.jl"))

function ared_dual_matrix(A, tangents::NTuple{N}) where {N}
    return map(A, tangents...) do a, ds...
        Dual{Nothing}(a, ds...)
    end
end

function ared_small_problem(n_tangents)
    A = Matrix([0.95 0.0; 0.0 0.95]')
    B = Matrix([0.5 0.0; 0.0 0.5]')
    R = Matrix(0.2I, 2, 2)
    Q = Matrix(0.5I, 2, 2)

    M = randn(size(Q)...)
    WX = 0.5 .* (M + M')
    WF = randn(size(B, 2), size(A, 1))
    A_tangents = ntuple(_ -> randn(size(A)...), n_tangents)
    B_tangents = ntuple(_ -> randn(size(B)...), n_tangents)
    R_tangents = ntuple(n_tangents) do _
        M = 0.1 .* randn(size(R)...)
        0.5 .* (M + M')
    end
    Q_tangents = ntuple(n_tangents) do _
        M = 0.1 .* randn(size(Q)...)
        0.5 .* (M + M')
    end
    return (; A, B, R, Q, WX, WF, A_tangents, B_tangents, R_tangents, Q_tangents)
end

function ared_medium_problem(n_tangents)
    n_x = 7
    n_u = 3
    A = 0.9 .* Matrix{Float64}(I, n_x, n_x) .+
        0.02 .* randn(n_x, n_x)
    B = SGUExampleMatrices.sgu_B_shock()
    R = Matrix(0.2I, n_u, n_u)
    Q = Matrix(0.5I, n_x, n_x)

    M = randn(size(Q)...)
    WX = 0.5 .* (M + M')
    WF = randn(n_u, n_x)
    A_tangents = ntuple(_ -> randn(size(A)...), n_tangents)
    B_tangents = ntuple(_ -> randn(size(B)...), n_tangents)
    R_tangents = ntuple(n_tangents) do _
        M = 0.1 .* randn(size(R)...)
        0.5 .* (M + M')
    end
    Q_tangents = ntuple(n_tangents) do _
        M = 0.1 .* randn(size(Q)...)
        0.5 .* (M + M')
    end
    return (; A, B, R, Q, WX, WF, A_tangents, B_tangents, R_tangents, Q_tangents)
end

function ared_loss(A, B, R, Q, WX, WF)
    X, _, F = ared(A, B, R, Q)
    return dot(WX, X) + dot(WF, F)
end

function ared_group(problem)
    g = BenchmarkGroup()

    g["primal"] = @benchmarkable ared_loss(A, B, R, Q, WX, WF) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        R = copy($(problem.R))
        Q = copy($(problem.Q))
        WX = $(problem.WX)
        WF = $(problem.WF)
    end evals = 4

    g["enzyme_reverse"] = @benchmarkable Enzyme.autodiff(
        Reverse, ared_loss, Active,
        Duplicated(A, A_bar),
        Duplicated(B, B_bar),
        Duplicated(R, R_bar),
        Duplicated(Q, Q_bar),
        Const(WX),
        Const(WF),
    ) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        R = copy($(problem.R))
        Q = copy($(problem.Q))
        A_bar = make_zero(A)
        B_bar = make_zero(B)
        R_bar = make_zero(R)
        Q_bar = make_zero(Q)
        WX = $(problem.WX)
        WF = $(problem.WF)
    end evals = 4

    g["enzyme_batch_forward"] = @benchmarkable Enzyme.autodiff(
        Forward, ared_loss, BatchDuplicated,
        BatchDuplicated(A, A_tangents),
        BatchDuplicated(B, B_tangents),
        BatchDuplicated(R, R_tangents),
        BatchDuplicated(Q, Q_tangents),
        Const(WX),
        Const(WF),
    ) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        R = copy($(problem.R))
        Q = copy($(problem.Q))
        A_tangents = ntuple(Val($(length(problem.A_tangents)))) do i
            tangent = make_zero(A)
            copyto!(tangent, $(problem.A_tangents)[i])
            tangent
        end
        B_tangents = ntuple(Val($(length(problem.B_tangents)))) do i
            tangent = make_zero(B)
            copyto!(tangent, $(problem.B_tangents)[i])
            tangent
        end
        R_tangents = ntuple(Val($(length(problem.R_tangents)))) do i
            tangent = make_zero(R)
            copyto!(tangent, $(problem.R_tangents)[i])
            tangent
        end
        Q_tangents = ntuple(Val($(length(problem.Q_tangents)))) do i
            tangent = make_zero(Q)
            copyto!(tangent, $(problem.Q_tangents)[i])
            tangent
        end
        WX = $(problem.WX)
        WF = $(problem.WF)
    end evals = 4

    g["forwarddiff_chunked"] = @benchmarkable ared_loss(
        A_dual, B_dual, R_dual, Q_dual, WX, WF,
    ) setup = begin
        A_dual = ared_dual_matrix($(problem.A), $(problem.A_tangents))
        B_dual = ared_dual_matrix($(problem.B), $(problem.B_tangents))
        R_dual = ared_dual_matrix($(problem.R), $(problem.R_tangents))
        Q_dual = ared_dual_matrix($(problem.Q), $(problem.Q_tangents))
        WX = $(problem.WX)
        WF = $(problem.WF)
    end evals = 4

    return g
end

ARED_SUITE = BenchmarkGroup()
ARED_SUITE["small"] = ared_group(ared_small_problem(Val(4)))
ARED_SUITE["medium"] = ared_group(ared_medium_problem(Val(4)))
ARED_SUITE
