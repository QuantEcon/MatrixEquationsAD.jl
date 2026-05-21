using BenchmarkTools
using Enzyme: Active, BatchDuplicated, Const, Duplicated, Enzyme, Forward, Reverse,
    make_zero
using ForwardDiff: Dual
using LinearAlgebra: I, dot
using MatrixEquations: gsylv, gsylvkr
using MatrixEquationsAD
using Random: randn

function gsylv_dual_matrix(A, tangents::NTuple{N}) where {N}
    return map(A, tangents...) do a, ds...
        Dual{Nothing}(a, ds...)
    end
end

function gsylv_small_problem(n_tangents)
    A = Matrix([4.0 0.1 0.0; -0.2 3.6 0.3; 0.1 0.0 3.8])
    C = Matrix(0.2I, 3, 3)
    B = Matrix([3.0 0.2; -0.1 2.7])
    D = Matrix(0.3I, 2, 2)
    E = [1.0 -0.4; 0.3 0.8; -0.2 0.5]
    W = randn(size(E)...)
    A_tangents = ntuple(_ -> randn(size(A)...), n_tangents)
    B_tangents = ntuple(_ -> randn(size(B)...), n_tangents)
    C_tangents = ntuple(_ -> randn(size(C)...), n_tangents)
    D_tangents = ntuple(_ -> randn(size(D)...), n_tangents)
    E_tangents = ntuple(_ -> randn(size(E)...), n_tangents)
    return (; A, B, C, D, E, W, A_tangents, B_tangents, C_tangents, D_tangents,
        E_tangents)
end

function gsylv_medium_problem(n_tangents)
    nA, nB = 7, 5
    A = 4.0 .* Matrix{Float64}(I, nA, nA) .+ 0.1 .* randn(nA, nA)
    C = 0.2 .* Matrix{Float64}(I, nA, nA)
    B = 3.0 .* Matrix{Float64}(I, nB, nB) .+ 0.1 .* randn(nB, nB)
    D = 0.3 .* Matrix{Float64}(I, nB, nB)
    E = randn(nA, nB)
    W = randn(nA, nB)
    A_tangents = ntuple(_ -> randn(size(A)...), n_tangents)
    B_tangents = ntuple(_ -> randn(size(B)...), n_tangents)
    C_tangents = ntuple(_ -> randn(size(C)...), n_tangents)
    D_tangents = ntuple(_ -> randn(size(D)...), n_tangents)
    E_tangents = ntuple(_ -> randn(size(E)...), n_tangents)
    return (; A, B, C, D, E, W, A_tangents, B_tangents, C_tangents, D_tangents,
        E_tangents)
end

gsylv_loss(A, B, C, D, E, W) = dot(W, gsylv(A, B, C, D, E))
gsylvkr_loss(A, B, C, D, E, W) = dot(W, gsylvkr(A, B, C, D, E))

function gsylv_group(problem)
    g = BenchmarkGroup()

    g["primal"] = @benchmarkable gsylv_loss(A, B, C, D, E, W) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        C = copy($(problem.C))
        D = copy($(problem.D))
        E = copy($(problem.E))
        W = $(problem.W)
    end evals = 4

    g["enzyme_reverse"] = @benchmarkable Enzyme.autodiff(
        Reverse, gsylv_loss, Active,
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
        A_bar = make_zero(A)
        B_bar = make_zero(B)
        C_bar = make_zero(C)
        D_bar = make_zero(D)
        E_bar = make_zero(E)
        W = $(problem.W)
    end evals = 4

    g["enzyme_batch_forward"] = @benchmarkable Enzyme.autodiff(
        Forward, gsylv_loss, BatchDuplicated,
        BatchDuplicated(A, A_tangents),
        BatchDuplicated(B, B_tangents),
        BatchDuplicated(C, C_tangents),
        BatchDuplicated(D, D_tangents),
        BatchDuplicated(E, E_tangents),
        Const(W),
    ) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        C = copy($(problem.C))
        D = copy($(problem.D))
        E = copy($(problem.E))
        A_tangents = ntuple(Val($(length(problem.A_tangents)))) do i
            tangent = make_zero(A)
            copyto!(tangent, $(problem.A_tangents)[i])
            tangent
        end
        B_tangents = ntuple(Val($(length(problem.B_tangents)))) do i
            tangent = make_zero(B)
            copyto!(tangent, $(problem.B_tangents)[i])
            tangent
        end
        C_tangents = ntuple(Val($(length(problem.C_tangents)))) do i
            tangent = make_zero(C)
            copyto!(tangent, $(problem.C_tangents)[i])
            tangent
        end
        D_tangents = ntuple(Val($(length(problem.D_tangents)))) do i
            tangent = make_zero(D)
            copyto!(tangent, $(problem.D_tangents)[i])
            tangent
        end
        E_tangents = ntuple(Val($(length(problem.E_tangents)))) do i
            tangent = make_zero(E)
            copyto!(tangent, $(problem.E_tangents)[i])
            tangent
        end
        W = $(problem.W)
    end evals = 4

    g["forwarddiff_chunked"] = @benchmarkable gsylv_loss(
        A_dual, B_dual, C_dual, D_dual, E_dual, W,
    ) setup = begin
        A_dual = gsylv_dual_matrix($(problem.A), $(problem.A_tangents))
        B_dual = gsylv_dual_matrix($(problem.B), $(problem.B_tangents))
        C_dual = gsylv_dual_matrix($(problem.C), $(problem.C_tangents))
        D_dual = gsylv_dual_matrix($(problem.D), $(problem.D_tangents))
        E_dual = gsylv_dual_matrix($(problem.E), $(problem.E_tangents))
        W = $(problem.W)
    end evals = 4

    return g
end

function gsylvkr_group(problem)
    g = BenchmarkGroup()

    g["primal"] = @benchmarkable gsylvkr_loss(A, B, C, D, E, W) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        C = copy($(problem.C))
        D = copy($(problem.D))
        E = copy($(problem.E))
        W = $(problem.W)
    end evals = 4

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
        A_bar = make_zero(A)
        B_bar = make_zero(B)
        C_bar = make_zero(C)
        D_bar = make_zero(D)
        E_bar = make_zero(E)
        W = $(problem.W)
    end evals = 4

    g["enzyme_batch_forward"] = @benchmarkable Enzyme.autodiff(
        Forward, gsylvkr_loss, BatchDuplicated,
        BatchDuplicated(A, A_tangents),
        BatchDuplicated(B, B_tangents),
        BatchDuplicated(C, C_tangents),
        BatchDuplicated(D, D_tangents),
        BatchDuplicated(E, E_tangents),
        Const(W),
    ) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        C = copy($(problem.C))
        D = copy($(problem.D))
        E = copy($(problem.E))
        A_tangents = ntuple(Val($(length(problem.A_tangents)))) do i
            tangent = make_zero(A)
            copyto!(tangent, $(problem.A_tangents)[i])
            tangent
        end
        B_tangents = ntuple(Val($(length(problem.B_tangents)))) do i
            tangent = make_zero(B)
            copyto!(tangent, $(problem.B_tangents)[i])
            tangent
        end
        C_tangents = ntuple(Val($(length(problem.C_tangents)))) do i
            tangent = make_zero(C)
            copyto!(tangent, $(problem.C_tangents)[i])
            tangent
        end
        D_tangents = ntuple(Val($(length(problem.D_tangents)))) do i
            tangent = make_zero(D)
            copyto!(tangent, $(problem.D_tangents)[i])
            tangent
        end
        E_tangents = ntuple(Val($(length(problem.E_tangents)))) do i
            tangent = make_zero(E)
            copyto!(tangent, $(problem.E_tangents)[i])
            tangent
        end
        W = $(problem.W)
    end evals = 4

    g["forwarddiff_chunked"] = @benchmarkable gsylvkr_loss(
        A_dual, B_dual, C_dual, D_dual, E_dual, W,
    ) setup = begin
        A_dual = gsylv_dual_matrix($(problem.A), $(problem.A_tangents))
        B_dual = gsylv_dual_matrix($(problem.B), $(problem.B_tangents))
        C_dual = gsylv_dual_matrix($(problem.C), $(problem.C_tangents))
        D_dual = gsylv_dual_matrix($(problem.D), $(problem.D_tangents))
        E_dual = gsylv_dual_matrix($(problem.E), $(problem.E_tangents))
        W = $(problem.W)
    end evals = 4

    return g
end

gsylv_small = gsylv_small_problem(Val(4))
gsylv_medium = gsylv_medium_problem(Val(4))

GSYLV_SUITE = BenchmarkGroup()
GSYLV_SUITE["gsylv"] = BenchmarkGroup()
GSYLV_SUITE["gsylv"]["small"] = gsylv_group(gsylv_small)
GSYLV_SUITE["gsylv"]["medium"] = gsylv_group(gsylv_medium)
GSYLV_SUITE["gsylvkr"] = BenchmarkGroup()
GSYLV_SUITE["gsylvkr"]["small"] = gsylvkr_group(gsylv_small)
GSYLV_SUITE["gsylvkr"]["medium"] = gsylvkr_group(gsylv_medium)
GSYLV_SUITE
