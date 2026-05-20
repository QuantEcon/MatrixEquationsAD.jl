using BenchmarkTools
using Enzyme: Active, BatchDuplicated, Const, Duplicated, Enzyme, Forward, Reverse
using LinearAlgebra: dot
using MatrixEquations: gsylvkr
using MatrixEquationsAD

const GSYLVKR_ENZ_LANES = 4

gsylvkr_loss(A, B, C, D, E, W) = dot(W, gsylvkr(A, B, C, D, E))

function gsylvkr_group(problem)
    g = BenchmarkGroup()

    g["primal"] = @benchmarkable gsylvkr_loss(A, B, C, D, E, W) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        C = copy($(problem.C))
        D = copy($(problem.D))
        E = copy($(problem.E))
        W = $(problem.W)
    end evals = 1

    g["enzyme_reverse"] = @benchmarkable Enzyme.autodiff(
        Reverse, gsylvkr_loss, Active,
        Duplicated(A, A_bar),
        Duplicated(B, B_bar),
        Duplicated(C, C_bar),
        Duplicated(D, D_bar),
        Duplicated(E, E_bar),
        Const(W),
    ) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        C = copy($(problem.C))
        D = copy($(problem.D))
        E = copy($(problem.E))
        A_bar = zeros(size(A))
        B_bar = zeros(size(B))
        C_bar = zeros(size(C))
        D_bar = zeros(size(D))
        E_bar = zeros(size(E))
        W = $(problem.W)
    end evals = 1

    g["enzyme_batch_forward"] = @benchmarkable Enzyme.autodiff(
        Forward, gsylvkr_loss, BatchDuplicated,
        BatchDuplicated(A, A_tans),
        BatchDuplicated(B, B_tans),
        BatchDuplicated(C, C_tans),
        BatchDuplicated(D, D_tans),
        BatchDuplicated(E, E_tans),
        Const(W),
    ) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        C = copy($(problem.C))
        D = copy($(problem.D))
        E = copy($(problem.E))
        A_tans = ntuple(i -> copy($(problem.dA_lanes)[i]), Val($GSYLVKR_ENZ_LANES))
        B_tans = ntuple(i -> copy($(problem.dB_lanes)[i]), Val($GSYLVKR_ENZ_LANES))
        C_tans = ntuple(i -> copy($(problem.dC_lanes)[i]), Val($GSYLVKR_ENZ_LANES))
        D_tans = ntuple(i -> copy($(problem.dD_lanes)[i]), Val($GSYLVKR_ENZ_LANES))
        E_tans = ntuple(i -> copy($(problem.dE_lanes)[i]), Val($GSYLVKR_ENZ_LANES))
        W = $(problem.W)
    end evals = 1

    g["forwarddiff_jvp"] = @benchmarkable gsylvkr_loss(A_dual, B_dual, C_dual, D_dual, E_dual, W) setup = begin
        A_dual = dual_matrix($(problem.A), $(problem.dA_lanes))
        B_dual = dual_matrix($(problem.B), $(problem.dB_lanes))
        C_dual = dual_matrix($(problem.C), $(problem.dC_lanes))
        D_dual = dual_matrix($(problem.D), $(problem.dD_lanes))
        E_dual = dual_matrix($(problem.E), $(problem.dE_lanes))
        W = $(problem.W)
    end evals = 1

    return g
end

g = BenchmarkGroup()
g["small"]  = gsylvkr_group(gsylv_small_problem())
g["medium"] = gsylvkr_group(gsylv_medium_problem())
g
