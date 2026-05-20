using BenchmarkTools
using Enzyme: Active, BatchDuplicated, Const, Duplicated, Enzyme, Forward, Reverse
using MatrixEquationsAD
using StaticArrays: SMatrix

function ordqz_oop_loss(A, B, threshold)
    r = ordqz(A, B, :bk; threshold)
    return sum(abs2, r.Q * r.S * r.Z')
end

function ordqz_oop_heap_group(problem)
    g = BenchmarkGroup()

    g["primal"] = @benchmarkable ordqz_oop_loss(A, B, threshold) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        threshold = $(problem.threshold)
    end evals = 1

    g["enzyme_reverse"] = @benchmarkable Enzyme.autodiff(
        Reverse, ordqz_oop_loss, Active,
        Duplicated(A, A_bar),
        Duplicated(B, B_bar),
        Const(threshold),
    ) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        A_bar = zeros(size(A))
        B_bar = zeros(size(B))
        threshold = $(problem.threshold)
    end evals = 1

    g["enzyme_batch_forward"] = @benchmarkable Enzyme.autodiff(
        Forward, ordqz_oop_loss, BatchDuplicated,
        BatchDuplicated(A, A_tans),
        BatchDuplicated(B, B_tans),
        Const(threshold),
    ) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        A_tans = ntuple(i -> copy($(problem.dA_lanes)[i]), Val(4))
        B_tans = ntuple(i -> copy($(problem.dB_lanes)[i]), Val(4))
        threshold = $(problem.threshold)
    end evals = 1

    g["forwarddiff_jvp"] = @benchmarkable ordqz_oop_loss(A_dual, B_dual, threshold) setup = begin
        A_dual = dual_matrix($(problem.A), $(problem.dA_lanes))
        B_dual = dual_matrix($(problem.B), $(problem.dB_lanes))
        threshold = $(problem.threshold)
    end evals = 1

    return g
end

function ordqz_oop_static_group(problem)
    g = BenchmarkGroup()

    g["primal"] = @benchmarkable ordqz_oop_loss(A, B, threshold) setup = begin
        A = $(problem.A)
        B = $(problem.B)
        threshold = $(problem.threshold)
    end evals = 1

    g["enzyme_reverse"] = @benchmarkable Enzyme.autodiff(
        Reverse, ordqz_oop_loss, Active,
        Active(A), Active(B), Const(threshold),
    ) setup = begin
        A = $(problem.A)
        B = $(problem.B)
        threshold = $(problem.threshold)
    end evals = 1

    g["enzyme_batch_forward"] = @benchmarkable Enzyme.autodiff(
        Forward, ordqz_oop_loss, BatchDuplicated,
        BatchDuplicated(A, A_tans),
        BatchDuplicated(B, B_tans),
        Const(threshold),
    ) setup = begin
        A = $(problem.A)
        B = $(problem.B)
        A_tans = $(problem.dA_lanes)
        B_tans = $(problem.dB_lanes)
        threshold = $(problem.threshold)
    end evals = 1

    g["forwarddiff_jvp"] = @benchmarkable ordqz_oop_loss(A_dual, B_dual, threshold) setup = begin
        A_dual = dual_matrix($(problem.A), $(problem.dA_lanes))
        B_dual = dual_matrix($(problem.B), $(problem.dB_lanes))
        threshold = $(problem.threshold)
    end evals = 1

    return g
end

g = BenchmarkGroup()
g["heap_small"]  = ordqz_oop_heap_group(ordqz_small_problem())
g["heap_medium"] = ordqz_oop_heap_group(ordqz_medium_problem())
g["static_small"] = ordqz_oop_static_group(ordqz_static_problem())
g
