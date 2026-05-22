using BenchmarkTools
using Enzyme: Active, BatchDuplicated, Const, Duplicated, Enzyme, Forward, Reverse,
    make_zero
using ForwardDiff: Dual
using LinearAlgebra: I, dot
using MatrixEquations: lyapd
using MatrixEquationsAD
using Random: randn

include(joinpath(pkgdir(MatrixEquationsAD), "test", "example_matrices", "rbc.jl"))
include(joinpath(pkgdir(MatrixEquationsAD), "test", "example_matrices", "sgu.jl"))
include(joinpath(pkgdir(MatrixEquationsAD), "test", "example_matrices", "fvgq.jl"))
include(joinpath(pkgdir(MatrixEquationsAD), "test", "example_matrices", "sw07.jl"))

function lyapd_dual_matrix(A, tangents::NTuple{N}) where {N}
    return map(A, tangents...) do a, ds...
        Dual{Nothing}(a, ds...)
    end
end

# Build a `(h_x, C, W, A_tangents, C_tangents)` problem from a DSGE
# fixture bundle. `C = B_shock B_shock' + 1e-6 I` is the same nudging
# used by the existing SGU benchmark; the identity keeps `C` strictly PD
# even when the shock loading is rank-deficient (e.g. RBC has rank 1).
function lyapd_problem_from(fo, n_tangents)
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

lyapd_small_problem(n_tangents) =
    lyapd_problem_from(RBCExampleMatrices.dp_rbc_first_order_inputs(), n_tangents)
lyapd_medium_problem(n_tangents) =
    lyapd_problem_from(SGUExampleMatrices.dp_sgu_first_order_inputs(), n_tangents)
lyapd_fvgq_problem(n_tangents) =
    lyapd_problem_from(FVGQExampleMatrices.dp_fvgq_first_order_inputs(), n_tangents)
lyapd_sw07pfeifer_problem(n_tangents) =
    lyapd_problem_from(SW07ExampleMatrices.dp_sw07pfeifer_first_order_inputs(), n_tangents)

lyapd_loss(A, C, W) = dot(W, lyapd(A, C))

function lyapd_inplace_loss(A, C, W)
    X = Matrix{eltype(A)}(undef, size(A))
    lyapd!(X, A, C)
    return dot(W, X)
end

# OOP `lyapd` — primal + chunked-`Dual` ForwardDiff, Enzyme batched
# forward, and Enzyme reverse. Small / medium only (RBC and SGU).
function lyapd_oop_group(problem)
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

# In-place `lyapd!` — primal + chunked-`Dual` ForwardDiff, Enzyme batched
# forward, and Enzyme reverse. Runs at all four sizes (RBC, SGU, FVGQ,
# SW07PFEIFER); the in-place rule is the path users want exercised on the
# big pencils.
function lyapd_inplace_group(problem)
    g = BenchmarkGroup()

    g["primal"] = @benchmarkable lyapd!(X, A, C) setup = begin
        A = copy($(problem.A))
        C = copy($(problem.C))
        X = zeros(size(A))
    end evals = 4

    g["forwarddiff_chunked"] = @benchmarkable lyapd_inplace_loss(A_dual, C_dual, W) setup = begin
        A_dual = lyapd_dual_matrix($(problem.A), $(problem.A_tangents))
        C_dual = lyapd_dual_matrix($(problem.C), $(problem.C_tangents))
        W = $(problem.W)
    end evals = 4

    g["enzyme_batch_forward"] = @benchmarkable Enzyme.autodiff(
        Forward, lyapd_inplace_loss, BatchDuplicated,
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

    g["enzyme_reverse"] = @benchmarkable Enzyme.autodiff(
        Reverse, lyapd_inplace_loss, Active,
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

    return g
end

LYAPD_SUITE = BenchmarkGroup()
LYAPD_SUITE["lyapd"] = BenchmarkGroup()
LYAPD_SUITE["lyapd"]["small"] = lyapd_oop_group(lyapd_small_problem(Val(4)))
LYAPD_SUITE["lyapd"]["medium"] = lyapd_oop_group(lyapd_medium_problem(Val(4)))
LYAPD_SUITE["lyapd!"] = BenchmarkGroup()
LYAPD_SUITE["lyapd!"]["small"] = lyapd_inplace_group(lyapd_small_problem(Val(4)))
LYAPD_SUITE["lyapd!"]["medium"] = lyapd_inplace_group(lyapd_medium_problem(Val(4)))
LYAPD_SUITE["lyapd!"]["fvgq"] = lyapd_inplace_group(lyapd_fvgq_problem(Val(4)))
LYAPD_SUITE["lyapd!"]["sw07pfeifer"] =
    lyapd_inplace_group(lyapd_sw07pfeifer_problem(Val(4)))
LYAPD_SUITE
