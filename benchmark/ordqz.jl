using BenchmarkTools
using Enzyme: Active, BatchDuplicated, Const, Duplicated, Enzyme, Forward, Reverse
using MatrixEquationsAD

const ORDQZ_ENZ_LANES = 4

# Reverse-mode loss reconstructs (A, B) from the ordered Schur factors so the
# rule can be exercised end-to-end on a scalar pullback.
function ordqz_reverse_loss(A, B, threshold)
    S = zero(A)
    T = zero(B)
    Q = zero(A)
    Z = zero(A)
    ordqz!(S, T, Q, Z, A, B, :bk; threshold)
    return sum(abs2, Q * S * Z')
end

# Forward-mode loss reuses caller-supplied buffers so Enzyme's Forward rule on
# the in-place ordqz! is the path actually timed.
function ordqz_forward_loss!(S, T, Q, Z, A, B, threshold)
    ordqz!(S, T, Q, Z, A, B, :bk; threshold)
    return sum(abs2, Q * S * Z')
end

function ordqz_group(problem)
    g = BenchmarkGroup()

    g["primal"] = @benchmarkable ordqz_reverse_loss(A, B, threshold) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        threshold = $(problem.threshold)
    end evals = 1

    g["enzyme_reverse"] = @benchmarkable Enzyme.autodiff(
        Reverse, ordqz_reverse_loss, Active,
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
        Forward, ordqz_forward_loss!, BatchDuplicated,
        BatchDuplicated(S, S_tans),
        BatchDuplicated(T, T_tans),
        BatchDuplicated(Q, Q_tans),
        BatchDuplicated(Z, Z_tans),
        BatchDuplicated(A, A_tans),
        BatchDuplicated(B, B_tans),
        Const(threshold),
    ) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        S = zeros(size(A))
        T = zeros(size(B))
        Q = zeros(size(A))
        Z = zeros(size(A))
        A_tans = ntuple(i -> copy($(problem.dA_lanes)[i]), Val($ORDQZ_ENZ_LANES))
        B_tans = ntuple(i -> copy($(problem.dB_lanes)[i]), Val($ORDQZ_ENZ_LANES))
        S_tans = ntuple(_ -> zeros(size(A)), Val($ORDQZ_ENZ_LANES))
        T_tans = ntuple(_ -> zeros(size(B)), Val($ORDQZ_ENZ_LANES))
        Q_tans = ntuple(_ -> zeros(size(A)), Val($ORDQZ_ENZ_LANES))
        Z_tans = ntuple(_ -> zeros(size(A)), Val($ORDQZ_ENZ_LANES))
        threshold = $(problem.threshold)
    end evals = 1

    g["forwarddiff_jvp"] = @benchmarkable ordqz_reverse_loss(A_dual, B_dual, threshold) setup = begin
        A_dual = dual_matrix($(problem.A), $(problem.dA_lanes))
        B_dual = dual_matrix($(problem.B), $(problem.dB_lanes))
        threshold = $(problem.threshold)
    end evals = 1

    return g
end

g = BenchmarkGroup()
g["small"]  = ordqz_group(ordqz_small_problem())
g["medium"] = ordqz_group(ordqz_medium_problem())
g
