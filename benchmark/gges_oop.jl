using BenchmarkTools
using Enzyme: Active, BatchDuplicated, Const, Duplicated, Enzyme, Forward, Reverse
using MatrixEquationsAD
using StaticArrays: SMatrix

const GGES_OOP_ENZ_LANES = 4

function gges_oop_loss(A, B, criterium)
    r = gges(A, B; select = :ed, criterium)
    return sum(abs2, r.Q * r.S * r.Z')
end

function gges_oop_heap_group(problem)
    g = BenchmarkGroup()
    criterium = (1 - problem.threshold)^2

    g["primal"] = @benchmarkable gges_oop_loss(A, B, criterium) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        criterium = $criterium
    end evals = 1

    g["enzyme_reverse"] = @benchmarkable Enzyme.autodiff(
        Reverse, gges_oop_loss, Active,
        Duplicated(A, A_bar),
        Duplicated(B, B_bar),
        Const(criterium),
    ) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        A_bar = zeros(size(A))
        B_bar = zeros(size(B))
        criterium = $criterium
    end evals = 1

    g["enzyme_batch_forward"] = @benchmarkable Enzyme.autodiff(
        Forward, gges_oop_loss, BatchDuplicated,
        BatchDuplicated(A, A_tans),
        BatchDuplicated(B, B_tans),
        Const(criterium),
    ) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        A_tans = ntuple(i -> copy($(problem.dA_lanes)[i]), Val($GGES_OOP_ENZ_LANES))
        B_tans = ntuple(i -> copy($(problem.dB_lanes)[i]), Val($GGES_OOP_ENZ_LANES))
        criterium = $criterium
    end evals = 1

    g["forwarddiff_jvp"] = @benchmarkable gges_oop_loss(A_dual, B_dual, criterium) setup = begin
        A_dual = dual_matrix($(problem.A), $(problem.dA_lanes))
        B_dual = dual_matrix($(problem.B), $(problem.dB_lanes))
        criterium = $criterium
    end evals = 1

    return g
end

function gges_oop_static_group(problem)
    g = BenchmarkGroup()
    criterium = (1 - problem.threshold)^2

    g["primal"] = @benchmarkable gges_oop_loss(A, B, criterium) setup = begin
        A = $(problem.A)
        B = $(problem.B)
        criterium = $criterium
    end evals = 1

    # Enzyme on SMatrix uses Active (immutable); returns gradient as SMatrix.
    g["enzyme_reverse"] = @benchmarkable Enzyme.autodiff(
        Reverse, gges_oop_loss, Active,
        Active(A), Active(B), Const(criterium),
    ) setup = begin
        A = $(problem.A)
        B = $(problem.B)
        criterium = $criterium
    end evals = 1

    g["enzyme_batch_forward"] = @benchmarkable Enzyme.autodiff(
        Forward, gges_oop_loss, BatchDuplicated,
        BatchDuplicated(A, A_tans),
        BatchDuplicated(B, B_tans),
        Const(criterium),
    ) setup = begin
        A = $(problem.A)
        B = $(problem.B)
        A_tans = $(problem.dA_lanes)
        B_tans = $(problem.dB_lanes)
        criterium = $criterium
    end evals = 1

    g["forwarddiff_jvp"] = @benchmarkable gges_oop_loss(A_dual, B_dual, criterium) setup = begin
        A_dual = dual_matrix($(problem.A), $(problem.dA_lanes))
        B_dual = dual_matrix($(problem.B), $(problem.dB_lanes))
        criterium = $criterium
    end evals = 1

    return g
end

g = BenchmarkGroup()
g["heap_small"]  = gges_oop_heap_group(ordqz_small_problem())
g["heap_medium"] = gges_oop_heap_group(ordqz_medium_problem())
g["static_small"] = gges_oop_static_group(ordqz_static_problem())
g
