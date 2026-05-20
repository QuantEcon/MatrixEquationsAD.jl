using BenchmarkTools
using Enzyme: Active, BatchDuplicated, Const, Duplicated, Enzyme, Forward, Reverse
using LinearAlgebra: dot
using MatrixEquations: ared
using MatrixEquationsAD

const ARED_ENZ_LANES = 4

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
    end evals = 1

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
        A_bar = zeros(size(A))
        B_bar = zeros(size(B))
        R_bar = zeros(size(R))
        Q_bar = zeros(size(Q))
        WX = $(problem.WX)
        WF = $(problem.WF)
    end evals = 1

    g["enzyme_batch_forward"] = @benchmarkable Enzyme.autodiff(
        Forward, ared_loss, BatchDuplicated,
        BatchDuplicated(A, A_tans),
        BatchDuplicated(B, B_tans),
        BatchDuplicated(R, R_tans),
        BatchDuplicated(Q, Q_tans),
        Const(WX),
        Const(WF),
    ) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        R = copy($(problem.R))
        Q = copy($(problem.Q))
        A_tans = ntuple(i -> copy($(problem.dA_lanes)[i]), Val($ARED_ENZ_LANES))
        B_tans = ntuple(i -> copy($(problem.dB_lanes)[i]), Val($ARED_ENZ_LANES))
        R_tans = ntuple(i -> copy($(problem.dR_lanes)[i]), Val($ARED_ENZ_LANES))
        Q_tans = ntuple(i -> copy($(problem.dQ_lanes)[i]), Val($ARED_ENZ_LANES))
        WX = $(problem.WX)
        WF = $(problem.WF)
    end evals = 1

    g["forwarddiff_jvp"] = @benchmarkable ared_loss(A_dual, B_dual, R_dual, Q_dual, WX, WF) setup = begin
        A_dual = dual_matrix($(problem.A), $(problem.dA_lanes))
        B_dual = dual_matrix($(problem.B), $(problem.dB_lanes))
        R_dual = dual_matrix($(problem.R), $(problem.dR_lanes))
        Q_dual = dual_matrix($(problem.Q), $(problem.dQ_lanes))
        WX = $(problem.WX)
        WF = $(problem.WF)
    end evals = 1

    return g
end

g = BenchmarkGroup()
g["small"]  = ared_group(ared_small_problem())
g["medium"] = ared_group(ared_medium_problem())
g
