using BenchmarkTools
using Enzyme: Active, BatchDuplicated, Const, Duplicated, Enzyme, Forward, Reverse,
    make_zero
using ForwardDiff: Dual
using LinearAlgebra: I, dot
using MatrixEquationsAD
using Random: randn
using StaticArrays: SMatrix

include(joinpath(pkgdir(MatrixEquationsAD), "test", "example_matrices", "rbc.jl"))
include(joinpath(pkgdir(MatrixEquationsAD), "test", "example_matrices", "sgu.jl"))
include(joinpath(pkgdir(MatrixEquationsAD), "test", "example_matrices", "fvgq.jl"))

function lyapdkr_dual_matrix(A, tangents::NTuple{N}) where {N}
    return map(A, tangents...) do a, ds...
        Dual{Nothing}(a, ds...)
    end
end

function lyapdkr_small_problem(n_tangents)
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

function lyapdkr_medium_problem(n_tangents)
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

function lyapdkr_large_problem(n_tangents)
    fo = FVGQExampleMatrices.fvgq_first_order_inputs()
    A = fo.h_x
    B = fo.B_shock
    n = size(A, 1)
    C = B * B' + 1.0e-6 * I(n)
    W = randn(n, n)
    A_tangents = ntuple(_ -> randn(n, n), n_tangents)
    C_tangents = ntuple(n_tangents) do _
        M = 0.1 .* randn(n, n)
        0.5 .* (M + M')
    end
    return (; A, C, W, A_tangents, C_tangents)
end

function lyapdkr_static_small_problem(n_tangents)
    problem = lyapdkr_small_problem(n_tangents)
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

lyapdkr_loss(A, C, W) = dot(W, lyapdkr(A, C))

# Heap lanes: small (n = 5), medium (n_SGU), large (n_FVGQ = 14). The
# Kronecker operator is n²×n² so past n ≈ 15 lyapdkr stops being
# competitive; the `large` lane is included to expose AD overheads at the
# upper edge of the useful range.
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

    g["forwarddiff_chunked"] = @benchmarkable lyapdkr_loss(A_dual, C_dual, W) setup = begin
        A_dual = lyapdkr_dual_matrix($(problem.A), $(problem.A_tangents))
        C_dual = lyapdkr_dual_matrix($(problem.C), $(problem.C_tangents))
        W = $(problem.W)
    end evals = 4

    return g
end

# Static (SMatrix) lane. Enzyme reverse on the SMatrix path needs a
# dedicated rule, so this lane only covers primal + forward.
function lyapdkr_static_group(problem)
    g = BenchmarkGroup()

    g["primal"] = @benchmarkable lyapdkr_loss(A, C, W) setup = begin
        A = $(problem.A)
        C = $(problem.C)
        W = $(problem.W)
    end evals = 4

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
        A_dual = lyapdkr_dual_matrix($(problem.A), $(problem.A_tangents))
        C_dual = lyapdkr_dual_matrix($(problem.C), $(problem.C_tangents))
        W = $(problem.W)
    end evals = 4

    return g
end

lyapdkr_loss_ws(A, C, W, M_ws) = dot(W, lyapdkr(A, C; M_ws))

# `lyapdkr(A, C; M_ws)`: same shape as `lyapdkr_group` but threads a
# caller-owned `n²×n²` scratch matrix through every AD path. Only run at
# the `large` size — at small / medium the M alloc is dust against the LU.
function lyapdkr_ws_group(problem)
    g = BenchmarkGroup()
    n = size(problem.A, 1)

    g["primal"] = @benchmarkable lyapdkr_loss_ws(A, C, W, M_ws) setup = begin
        A = copy($(problem.A))
        C = copy($(problem.C))
        W = $(problem.W)
        M_ws = Matrix{Float64}(undef, $(n * n), $(n * n))
    end evals = 4

    g["enzyme_reverse"] = @benchmarkable Enzyme.autodiff(
        Reverse, lyapdkr_loss_ws, Active,
        Duplicated(A, A_bar),
        Duplicated(C, C_bar),
        Const(W),
        Const(M_ws),
    ) setup = begin
        A = copy($(problem.A))
        C = copy($(problem.C))
        A_bar = make_zero(A)
        C_bar = make_zero(C)
        W = $(problem.W)
        M_ws = Matrix{Float64}(undef, $(n * n), $(n * n))
    end evals = 4

    g["enzyme_batch_forward"] = @benchmarkable Enzyme.autodiff(
        Forward, lyapdkr_loss_ws, BatchDuplicated,
        BatchDuplicated(A, A_tangents),
        BatchDuplicated(C, C_tangents),
        Const(W),
        Const(M_ws),
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
        M_ws = Matrix{Float64}(undef, $(n * n), $(n * n))
    end evals = 4

    g["forwarddiff_chunked"] = @benchmarkable lyapdkr_loss_ws(A_dual, C_dual, W, M_ws) setup = begin
        A_dual = lyapdkr_dual_matrix($(problem.A), $(problem.A_tangents))
        C_dual = lyapdkr_dual_matrix($(problem.C), $(problem.C_tangents))
        W = $(problem.W)
        M_ws = Matrix{Float64}(undef, $(n * n), $(n * n))
    end evals = 4

    return g
end

# `lyapdkr!(X, A, C)`: in-place output. Same four lanes as `lyapdkr_group`.
# Setup allocates `X` once outside the body.
function lyapdkr_inplace_loss(X, A, C, W)
    lyapdkr!(X, A, C)
    return dot(W, X)
end
function lyapdkr_inplace_loss_ws(X, A, C, W, M_ws)
    lyapdkr!(X, A, C; M_ws)
    return dot(W, X)
end
# Positional wrapper so M_ws can flow through Enzyme.autodiff for the
# forward (Const return) path that doesn't need a scalar reduction.
function lyapdkr_inplace_apply_ws(X, A, C, M_ws)
    lyapdkr!(X, A, C; M_ws)
    return nothing
end

function lyapdkr_inplace_group(problem)
    g = BenchmarkGroup()

    g["primal"] = @benchmarkable lyapdkr_inplace_loss(X, A, C, W) setup = begin
        A = copy($(problem.A))
        C = copy($(problem.C))
        W = $(problem.W)
        X = similar(A)
    end evals = 4

    g["enzyme_reverse"] = @benchmarkable Enzyme.autodiff(
        Reverse, lyapdkr_inplace_loss, Active,
        Duplicated(X, X_bar),
        Duplicated(A, A_bar),
        Duplicated(C, C_bar),
        Const(W),
    ) setup = begin
        A = copy($(problem.A))
        C = copy($(problem.C))
        A_bar = make_zero(A)
        C_bar = make_zero(C)
        X = similar(A)
        X_bar = make_zero(X)
        W = $(problem.W)
    end evals = 4

    g["enzyme_batch_forward"] = @benchmarkable Enzyme.autodiff(
        Forward, lyapdkr!, Const,
        BatchDuplicated(X, X_tangents),
        BatchDuplicated(A, A_tangents),
        BatchDuplicated(C, C_tangents),
    ) setup = begin
        A = copy($(problem.A))
        C = copy($(problem.C))
        X = similar(A)
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
        X_tangents = ntuple(_ -> make_zero(X), Val($(length(problem.A_tangents))))
    end evals = 4

    g["forwarddiff_chunked"] = @benchmarkable lyapdkr!(X_dual, A_dual, C_dual) setup = begin
        A_dual = lyapdkr_dual_matrix($(problem.A), $(problem.A_tangents))
        C_dual = lyapdkr_dual_matrix($(problem.C), $(problem.C_tangents))
        X_dual = similar(A_dual)
    end evals = 4

    return g
end

function lyapdkr_inplace_ws_group(problem)
    g = BenchmarkGroup()
    n = size(problem.A, 1)

    g["primal"] = @benchmarkable lyapdkr_inplace_loss_ws(X, A, C, W, M_ws) setup = begin
        A = copy($(problem.A))
        C = copy($(problem.C))
        W = $(problem.W)
        X = similar(A)
        M_ws = Matrix{Float64}(undef, $(n * n), $(n * n))
    end evals = 4

    g["enzyme_reverse"] = @benchmarkable Enzyme.autodiff(
        Reverse, lyapdkr_inplace_loss_ws, Active,
        Duplicated(X, X_bar),
        Duplicated(A, A_bar),
        Duplicated(C, C_bar),
        Const(W),
        Const(M_ws),
    ) setup = begin
        A = copy($(problem.A))
        C = copy($(problem.C))
        A_bar = make_zero(A)
        C_bar = make_zero(C)
        X = similar(A)
        X_bar = make_zero(X)
        W = $(problem.W)
        M_ws = Matrix{Float64}(undef, $(n * n), $(n * n))
    end evals = 4

    g["enzyme_batch_forward"] = @benchmarkable Enzyme.autodiff(
        Forward, lyapdkr_inplace_apply_ws, Const,
        BatchDuplicated(X, X_tangents),
        BatchDuplicated(A, A_tangents),
        BatchDuplicated(C, C_tangents),
        Const(M_ws),
    ) setup = begin
        A = copy($(problem.A))
        C = copy($(problem.C))
        X = similar(A)
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
        X_tangents = ntuple(_ -> make_zero(X), Val($(length(problem.A_tangents))))
        M_ws = Matrix{Float64}(undef, $(n * n), $(n * n))
    end evals = 4

    g["forwarddiff_chunked"] = @benchmarkable lyapdkr!(X_dual, A_dual, C_dual; M_ws) setup = begin
        A_dual = lyapdkr_dual_matrix($(problem.A), $(problem.A_tangents))
        C_dual = lyapdkr_dual_matrix($(problem.C), $(problem.C_tangents))
        X_dual = similar(A_dual)
        M_ws = Matrix{Float64}(undef, $(n * n), $(n * n))
    end evals = 4

    return g
end

LYAPDKR_SUITE = BenchmarkGroup()
LYAPDKR_SUITE["small"] = lyapdkr_group(lyapdkr_small_problem(Val(4)))
LYAPDKR_SUITE["medium"] = lyapdkr_group(lyapdkr_medium_problem(Val(4)))
LYAPDKR_SUITE["large"] = lyapdkr_group(lyapdkr_large_problem(Val(4)))
LYAPDKR_SUITE["large_ws"] = lyapdkr_ws_group(lyapdkr_large_problem(Val(4)))
LYAPDKR_SUITE["static_small"] = lyapdkr_static_group(lyapdkr_static_small_problem(Val(4)))

LYAPDKR_INPLACE = BenchmarkGroup()
LYAPDKR_INPLACE["small"] = lyapdkr_inplace_group(lyapdkr_small_problem(Val(4)))
LYAPDKR_INPLACE["medium"] = lyapdkr_inplace_group(lyapdkr_medium_problem(Val(4)))
LYAPDKR_INPLACE["large"] = lyapdkr_inplace_group(lyapdkr_large_problem(Val(4)))
LYAPDKR_INPLACE["large_ws"] = lyapdkr_inplace_ws_group(lyapdkr_large_problem(Val(4)))
LYAPDKR_SUITE["inplace"] = LYAPDKR_INPLACE

LYAPDKR_SUITE
