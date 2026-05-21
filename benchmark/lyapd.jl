using BenchmarkTools
using Enzyme: Active, BatchDuplicated, Const, Duplicated, Enzyme, Forward, Reverse,
    make_zero
using ForwardDiff: Dual
using LinearAlgebra: I, dot
using MatrixEquations: lyapd
using MatrixEquationsAD
using Random: randn
using StaticArrays: SMatrix

include(joinpath(pkgdir(MatrixEquationsAD), "test", "example_matrices", "sgu.jl"))

function lyapd_dual_matrix(A, tangents::NTuple{N}) where {N}
    return map(A, tangents...) do a, ds...
        Dual{Nothing}(a, ds...)
    end
end

function lyapd_small_problem(n_tangents)
    A = [
        0.55 0.08 0.01 -0.02 0.0
        -0.04 0.42 0.05 0.01 -0.01
        0.02 -0.03 0.36 0.04 0.0
        0.0 0.02 -0.05 0.48 0.03
        -0.01 0.0 0.02 -0.04 0.51
    ]
    M = randn(5, 5)
    C = 0.5 .* (M + M') + 5 * I
    W = randn(size(A)...)
    A_tangents = ntuple(_ -> randn(size(A)...), n_tangents)
    C_tangents = ntuple(n_tangents) do _
        M = 0.1 .* randn(size(C)...)
        0.5 .* (M + M')
    end
    return (; A, C, W, A_tangents, C_tangents)
end

function lyapd_medium_problem(n_tangents)
    A = SGUExampleMatrices.sgu_h_x()
    B = SGUExampleMatrices.sgu_B_shock()
    C = B * B' + 1.0e-6 * I
    W = randn(size(A)...)
    A_tangents = ntuple(_ -> randn(size(A)...), n_tangents)
    C_tangents = ntuple(n_tangents) do _
        M = 0.1 .* randn(size(C)...)
        0.5 .* (M + M')
    end
    return (; A, C, W, A_tangents, C_tangents)
end

function lyapdkr_static_small_problem(n_tangents)
    problem = lyapd_small_problem(n_tangents)
    n = size(problem.A, 1)
    A = SMatrix{n, n, Float64}(problem.A)
    C = SMatrix{n, n, Float64}(problem.C)
    W = SMatrix{n, n, Float64}(problem.W)
    A_tangents = ntuple(n_tangents) do i
        SMatrix{n, n, Float64}(problem.A_tangents[i])
    end
    C_tangents = ntuple(n_tangents) do i
        SMatrix{n, n, Float64}(problem.C_tangents[i])
    end
    return (; A, C, W, A_tangents, C_tangents)
end

lyapd_loss(A, C, W) = dot(W, lyapd(A, C))
lyapdkr_loss(A, C, W) = dot(W, lyapdkr(A, C))

function lyapd_group(problem)
    g = BenchmarkGroup()

    g["primal"] = @benchmarkable lyapd_loss(A, C, W) setup = begin
        A = copy($(problem.A))
        C = copy($(problem.C))
        W = $(problem.W)
    end evals = 4

    g["enzyme_reverse"] = @benchmarkable Enzyme.autodiff(
        Reverse, lyapd_loss, Active,
        Duplicated(A, A_bar),
        Duplicated(C, C_bar),
        Const(W),
    ) setup = begin
        A = copy($(problem.A))
        C = copy($(problem.C))
        A_bar = make_zero(A)
        C_bar = make_zero(C)
        W = $(problem.W)
    end evals = 4

    g["enzyme_batch_forward"] = @benchmarkable Enzyme.autodiff(
        Forward, lyapd_loss, BatchDuplicated,
        BatchDuplicated(A, A_tangents),
        BatchDuplicated(C, C_tangents),
        Const(W),
    ) setup = begin
        A = copy($(problem.A))
        C = copy($(problem.C))
        A_tangents = ntuple(Val($(length(problem.A_tangents)))) do i
            tangent = make_zero(A)
            copyto!(tangent, $(problem.A_tangents)[i])
            tangent
        end
        C_tangents = ntuple(Val($(length(problem.C_tangents)))) do i
            tangent = make_zero(C)
            copyto!(tangent, $(problem.C_tangents)[i])
            tangent
        end
        W = $(problem.W)
    end evals = 4

    g["forwarddiff_chunked"] = @benchmarkable lyapd_loss(A_dual, C_dual, W) setup = begin
        A_dual = lyapd_dual_matrix($(problem.A), $(problem.A_tangents))
        C_dual = lyapd_dual_matrix($(problem.C), $(problem.C_tangents))
        W = $(problem.W)
    end evals = 4

    return g
end

function lyapdkr_group(problem)
    g = BenchmarkGroup()

    g["primal"] = @benchmarkable lyapdkr_loss(A, C, W) setup = begin
        A = copy($(problem.A))
        C = copy($(problem.C))
        W = $(problem.W)
    end evals = 4

    g["enzyme_reverse"] = @benchmarkable Enzyme.autodiff(
        Reverse, lyapdkr_loss, Active,
        Duplicated(A, A_bar),
        Duplicated(C, C_bar),
        Const(W),
    ) setup = begin
        A = copy($(problem.A))
        C = copy($(problem.C))
        A_bar = make_zero(A)
        C_bar = make_zero(C)
        W = $(problem.W)
    end evals = 4

    g["enzyme_batch_forward"] = @benchmarkable Enzyme.autodiff(
        Forward, lyapdkr_loss, BatchDuplicated,
        BatchDuplicated(A, A_tangents),
        BatchDuplicated(C, C_tangents),
        Const(W),
    ) setup = begin
        A = copy($(problem.A))
        C = copy($(problem.C))
        A_tangents = ntuple(Val($(length(problem.A_tangents)))) do i
            tangent = make_zero(A)
            copyto!(tangent, $(problem.A_tangents)[i])
            tangent
        end
        C_tangents = ntuple(Val($(length(problem.C_tangents)))) do i
            tangent = make_zero(C)
            copyto!(tangent, $(problem.C_tangents)[i])
            tangent
        end
        W = $(problem.W)
    end evals = 4

    # Static reverse needs a dedicated rule; don't benchmark Enzyme's fallback here.
    g["forwarddiff_chunked"] = @benchmarkable lyapdkr_loss(A_dual, C_dual, W) setup = begin
        A_dual = lyapd_dual_matrix($(problem.A), $(problem.A_tangents))
        C_dual = lyapd_dual_matrix($(problem.C), $(problem.C_tangents))
        W = $(problem.W)
    end evals = 4

    return g
end

function lyapdkr_static_group(problem)
    g = BenchmarkGroup()

    g["primal"] = @benchmarkable lyapdkr_loss(A, C, W) setup = begin
        A = $(problem.A)
        C = $(problem.C)
        W = $(problem.W)
    end evals = 4

    # Static reverse needs a dedicated rule; don't benchmark Enzyme's fallback here.
    g["enzyme_batch_forward"] = @benchmarkable Enzyme.autodiff(
        Forward, lyapdkr_loss, BatchDuplicated,
        BatchDuplicated(A, A_tangents),
        BatchDuplicated(C, C_tangents),
        Const(W),
    ) setup = begin
        A = $(problem.A)
        C = $(problem.C)
        A_tangents = $(problem.A_tangents)
        C_tangents = $(problem.C_tangents)
        W = $(problem.W)
    end evals = 4

    g["forwarddiff_chunked"] = @benchmarkable lyapdkr_loss(A_dual, C_dual, W) setup = begin
        A_dual = lyapd_dual_matrix($(problem.A), $(problem.A_tangents))
        C_dual = lyapd_dual_matrix($(problem.C), $(problem.C_tangents))
        W = $(problem.W)
    end evals = 4

    return g
end

lyapd_small = lyapd_small_problem(Val(4))
lyapd_medium = lyapd_medium_problem(Val(4))
lyapdkr_static_small = lyapdkr_static_small_problem(Val(4))

LYAPD_SUITE = BenchmarkGroup()
LYAPD_SUITE["lyapd"] = BenchmarkGroup()
LYAPD_SUITE["lyapd"]["small"] = lyapd_group(lyapd_small)
LYAPD_SUITE["lyapd"]["medium"] = lyapd_group(lyapd_medium)
LYAPD_SUITE["lyapdkr"] = BenchmarkGroup()
LYAPD_SUITE["lyapdkr"]["small"] = lyapdkr_group(lyapd_small)
LYAPD_SUITE["lyapdkr"]["medium"] = lyapdkr_group(lyapd_medium)
LYAPD_SUITE["lyapdkr"]["static_small"] = lyapdkr_static_group(lyapdkr_static_small)
LYAPD_SUITE
