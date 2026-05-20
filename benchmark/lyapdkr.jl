using BenchmarkTools
using Enzyme: Active, BatchDuplicated, Const, Duplicated, Enzyme, Forward, Reverse
using LinearAlgebra: dot
using MatrixEquationsAD

const LYAPDKR_ENZ_LANES = 4

lyapdkr_loss(A, C, W) = dot(W, lyapdkr(A, C))

function lyapdkr_group(problem)
    g = BenchmarkGroup()

    g["primal"] = @benchmarkable lyapdkr_loss(A, C, W) setup = begin
        A = copy($(problem.A))
        C = copy($(problem.C))
        W = $(problem.W)
    end evals = 1

    g["enzyme_reverse"] = @benchmarkable Enzyme.autodiff(
        Reverse, lyapdkr_loss, Active,
        Duplicated(A, A_bar),
        Duplicated(C, C_bar),
        Const(W),
    ) setup = begin
        A = copy($(problem.A))
        C = copy($(problem.C))
        A_bar = zeros(size(A))
        C_bar = zeros(size(C))
        W = $(problem.W)
    end evals = 1

    g["enzyme_batch_forward"] = @benchmarkable Enzyme.autodiff(
        Forward, lyapdkr_loss, BatchDuplicated,
        BatchDuplicated(A, A_tans),
        BatchDuplicated(C, C_tans),
        Const(W),
    ) setup = begin
        A = copy($(problem.A))
        C = copy($(problem.C))
        A_tans = ntuple(i -> copy($(problem.dA_lanes)[i]), Val($LYAPDKR_ENZ_LANES))
        C_tans = ntuple(i -> copy($(problem.dC_lanes)[i]), Val($LYAPDKR_ENZ_LANES))
        W = $(problem.W)
    end evals = 1

    g["forwarddiff_jvp"] = @benchmarkable lyapdkr_loss(A_dual, C_dual, W) setup = begin
        A_dual = dual_matrix($(problem.A), $(problem.dA_lanes))
        C_dual = dual_matrix($(problem.C), $(problem.dC_lanes))
        W = $(problem.W)
    end evals = 1

    return g
end

g = BenchmarkGroup()
g["small"]  = lyapdkr_group(lyapd_small_problem())
g["medium"] = lyapdkr_group(lyapd_medium_problem())
g
