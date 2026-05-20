using BenchmarkTools
using Enzyme: Active, BatchDuplicated, Const, Duplicated, Enzyme, Forward, Reverse
using MatrixEquationsAD

const GGES_ENZ_LANES = 4

function gges_reverse_loss(A, B, criterium)
    S = zero(A)
    T = zero(B)
    Q = zero(A)
    Z = zero(A)
    gges!(S, T, Q, Z, A, B; select = :ed, criterium)
    return sum(abs2, Q * S * Z')
end

function gges_forward_loss!(S, T, Q, Z, A, B, criterium)
    gges!(S, T, Q, Z, A, B; select = :ed, criterium)
    return sum(abs2, Q * S * Z')
end

function gges_group(problem)
    g = BenchmarkGroup()
    criterium = (1 - problem.threshold)^2

    g["primal"] = @benchmarkable gges_reverse_loss(A, B, criterium) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        criterium = $criterium
    end evals = 1

    g["enzyme_reverse"] = @benchmarkable Enzyme.autodiff(
        Reverse, gges_reverse_loss, Active,
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
        Forward, gges_forward_loss!, BatchDuplicated,
        BatchDuplicated(S, S_tans),
        BatchDuplicated(T, T_tans),
        BatchDuplicated(Q, Q_tans),
        BatchDuplicated(Z, Z_tans),
        BatchDuplicated(A, A_tans),
        BatchDuplicated(B, B_tans),
        Const(criterium),
    ) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        S = zeros(size(A))
        T = zeros(size(B))
        Q = zeros(size(A))
        Z = zeros(size(A))
        A_tans = ntuple(i -> copy($(problem.dA_lanes)[i]), Val($GGES_ENZ_LANES))
        B_tans = ntuple(i -> copy($(problem.dB_lanes)[i]), Val($GGES_ENZ_LANES))
        S_tans = ntuple(_ -> zeros(size(A)), Val($GGES_ENZ_LANES))
        T_tans = ntuple(_ -> zeros(size(B)), Val($GGES_ENZ_LANES))
        Q_tans = ntuple(_ -> zeros(size(A)), Val($GGES_ENZ_LANES))
        Z_tans = ntuple(_ -> zeros(size(A)), Val($GGES_ENZ_LANES))
        criterium = $criterium
    end evals = 1

    g["forwarddiff_jvp"] = @benchmarkable gges_reverse_loss(A_dual, B_dual, criterium) setup = begin
        A_dual = dual_matrix($(problem.A), $(problem.dA_lanes))
        B_dual = dual_matrix($(problem.B), $(problem.dB_lanes))
        criterium = $criterium
    end evals = 1

    return g
end

g = BenchmarkGroup()
g["small"]  = gges_group(ordqz_small_problem())
g["medium"] = gges_group(ordqz_medium_problem())
g
