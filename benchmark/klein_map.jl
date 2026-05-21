using BenchmarkTools
using Enzyme: Active, BatchDuplicated, Const, Duplicated, Enzyme, Forward, Reverse,
    make_zero
using ForwardDiff: Dual
using LinearAlgebra: dot
using MatrixEquationsAD
using Random: randn
using StaticArrays: SMatrix

include(joinpath(pkgdir(MatrixEquationsAD), "test", "example_matrices", "rbc.jl"))
include(joinpath(pkgdir(MatrixEquationsAD), "test", "example_matrices", "sgu.jl"))
include(joinpath(pkgdir(MatrixEquationsAD), "test", "example_matrices", "fvgq.jl"))
include(joinpath(pkgdir(MatrixEquationsAD), "test", "example_matrices", "sw07.jl"))

function klein_small_problem(threshold, n_tangents)
    A, B, n_x = RBCExampleMatrices.dp_rbc_first_order_gschur()
    n_y = size(A, 1) - n_x
    A_tangents = ntuple(_ -> randn(size(A)...), n_tangents)
    B_tangents = ntuple(_ -> randn(size(B)...), n_tangents)
    Wg = randn(n_y, n_x)
    Wh = randn(n_x, n_x)
    return (; A, B, n_x, threshold, A_tangents, B_tangents, Wg, Wh)
end

function klein_medium_problem(threshold, n_tangents)
    A, B, n_x = SGUExampleMatrices.dp_sgu_first_order_gschur()
    n_y = size(A, 1) - n_x
    A_tangents = ntuple(_ -> randn(size(A)...), n_tangents)
    B_tangents = ntuple(_ -> randn(size(B)...), n_tangents)
    Wg = randn(n_y, n_x)
    Wh = randn(n_x, n_x)
    return (; A, B, n_x, threshold, A_tangents, B_tangents, Wg, Wh)
end

function klein_fvgq_problem(threshold, n_tangents)
    A = FVGQExampleMatrices.fvgq_klein_gschur_A()
    B = FVGQExampleMatrices.fvgq_klein_gschur_B()
    n_x = 14
    n_y = size(A, 1) - n_x
    A_tangents = ntuple(_ -> randn(size(A)...), n_tangents)
    B_tangents = ntuple(_ -> randn(size(B)...), n_tangents)
    Wg = randn(n_y, n_x)
    Wh = randn(n_x, n_x)
    return (; A, B, n_x, threshold, A_tangents, B_tangents, Wg, Wh)
end

function klein_sw07pfeifer_problem(threshold, n_tangents)
    A, B, n_x = SW07ExampleMatrices.dp_sw07pfeifer_first_order_gschur()
    n_y = size(A, 1) - n_x
    A_tangents = ntuple(_ -> randn(size(A)...), n_tangents)
    B_tangents = ntuple(_ -> randn(size(B)...), n_tangents)
    Wg = randn(n_y, n_x)
    Wh = randn(n_x, n_x)
    return (; A, B, n_x, threshold, A_tangents, B_tangents, Wg, Wh)
end

function klein_static_small_problem(threshold, n_tangents)
    A, B, n_x = RBCExampleMatrices.dp_rbc_first_order_gschur()
    n = size(A, 1)
    n_y = n - n_x
    As = SMatrix{n, n, Float64}(A)
    Bs = SMatrix{n, n, Float64}(B)
    A_tangents = ntuple(n_tangents) do _
        SMatrix{n, n, Float64}(randn(n, n))
    end
    B_tangents = ntuple(n_tangents) do _
        SMatrix{n, n, Float64}(randn(n, n))
    end
    Wg = SMatrix{n_y, n_x, Float64}(randn(n_y, n_x))
    Wh = SMatrix{n_x, n_x, Float64}(randn(n_x, n_x))
    return (;
        A = As, B = Bs, n_x, threshold,
        A_tangents, B_tangents, Wg, Wh,
    )
end

function klein_map_oop_loss(A, B, Wg, Wh, threshold)
    r = klein_map(A, B; threshold)
    return dot(Wg, r.g_x) + dot(Wh, r.h_x)
end

function klein_map_oop_loss(A, B, Wg, Wh, threshold, n_x)
    r = klein_map(A, B, n_x; threshold)
    return dot(Wg, r.g_x) + dot(Wh, r.h_x)
end

function klein_map_inplace_loss(A, B, Wg, Wh, threshold)
    g_x = Matrix{eltype(A)}(undef, size(Wg))
    h_x = Matrix{eltype(A)}(undef, size(Wh))
    klein_map!(g_x, h_x, A, B; threshold)
    return dot(Wg, g_x) + dot(Wh, h_x)
end

function klein_map_oop_heap_group(problem)
    g = BenchmarkGroup()

    g["primal"] = @benchmarkable klein_map_oop_loss(A, B, Wg, Wh, threshold) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        Wg = $(problem.Wg)
        Wh = $(problem.Wh)
        threshold = $(problem.threshold)
    end evals = 4

    g["forwarddiff_chunked"] = @benchmarkable begin
        klein_map_oop_loss(A_dual, B_dual, Wg, Wh, threshold)
    end setup = begin
        A_dual = map($(problem.A), $(problem.A_tangents)...) do a, ds...
            Dual{Nothing}(a, ds...)
        end
        B_dual = map($(problem.B), $(problem.B_tangents)...) do b, ds...
            Dual{Nothing}(b, ds...)
        end
        Wg = $(problem.Wg)
        Wh = $(problem.Wh)
        threshold = $(problem.threshold)
    end evals = 4

    g["enzyme_batch_forward"] = @benchmarkable Enzyme.autodiff(
        Forward, klein_map_oop_loss, BatchDuplicated,
        BatchDuplicated(A, A_tangents),
        BatchDuplicated(B, B_tangents),
        Const(Wg), Const(Wh), Const(threshold),
    ) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
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
        Wg = $(problem.Wg)
        Wh = $(problem.Wh)
        threshold = $(problem.threshold)
    end evals = 4

    g["enzyme_reverse"] = @benchmarkable Enzyme.autodiff(
        Reverse, klein_map_oop_loss, Active,
        Duplicated(A, A_bar),
        Duplicated(B, B_bar),
        Const(Wg), Const(Wh), Const(threshold),
    ) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        A_bar = make_zero(A)
        B_bar = make_zero(B)
        Wg = $(problem.Wg)
        Wh = $(problem.Wh)
        threshold = $(problem.threshold)
    end evals = 4

    return g
end

function klein_map_inplace_heap_group(problem)
    g = BenchmarkGroup()

    g["primal"] = @benchmarkable begin
        klein_map!(g_x, h_x, A, B; threshold)
    end setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        g_x = zeros(size($(problem.Wg)))
        h_x = zeros(size($(problem.Wh)))
        threshold = $(problem.threshold)
    end evals = 4

    g["forwarddiff_chunked"] = @benchmarkable begin
        klein_map_inplace_loss(A_dual, B_dual, Wg, Wh, threshold)
    end setup = begin
        A_dual = map($(problem.A), $(problem.A_tangents)...) do a, ds...
            Dual{Nothing}(a, ds...)
        end
        B_dual = map($(problem.B), $(problem.B_tangents)...) do b, ds...
            Dual{Nothing}(b, ds...)
        end
        Wg = $(problem.Wg)
        Wh = $(problem.Wh)
        threshold = $(problem.threshold)
    end evals = 4

    g["enzyme_batch_forward"] = @benchmarkable Enzyme.autodiff(
        Forward, klein_map_inplace_loss, BatchDuplicated,
        BatchDuplicated(A, A_tangents),
        BatchDuplicated(B, B_tangents),
        Const(Wg), Const(Wh), Const(threshold),
    ) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
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
        Wg = $(problem.Wg)
        Wh = $(problem.Wh)
        threshold = $(problem.threshold)
    end evals = 4

    g["enzyme_reverse"] = @benchmarkable Enzyme.autodiff(
        Reverse, klein_map_inplace_loss, Active,
        Duplicated(A, A_bar),
        Duplicated(B, B_bar),
        Const(Wg), Const(Wh), Const(threshold),
    ) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        A_bar = make_zero(A)
        B_bar = make_zero(B)
        Wg = $(problem.Wg)
        Wh = $(problem.Wh)
        threshold = $(problem.threshold)
    end evals = 4

    return g
end

function klein_map_oop_static_group(problem)
    g = BenchmarkGroup()

    g["primal"] = @benchmarkable begin
        klein_map_oop_loss(A, B, Wg, Wh, threshold, n_x)
    end setup = begin
        A = $(problem.A)
        B = $(problem.B)
        Wg = $(problem.Wg)
        Wh = $(problem.Wh)
        threshold = $(problem.threshold)
        n_x = Val($(problem.n_x))
    end evals = 4

    g["forwarddiff_chunked"] = @benchmarkable begin
        klein_map_oop_loss(A_dual, B_dual, Wg, Wh, threshold, n_x)
    end setup = begin
        A_dual = map($(problem.A), $(problem.A_tangents)...) do a, ds...
            Dual{Nothing}(a, ds...)
        end
        B_dual = map($(problem.B), $(problem.B_tangents)...) do b, ds...
            Dual{Nothing}(b, ds...)
        end
        Wg = $(problem.Wg)
        Wh = $(problem.Wh)
        threshold = $(problem.threshold)
        n_x = Val($(problem.n_x))
    end evals = 4

    g["enzyme_batch_forward"] = @benchmarkable Enzyme.autodiff(
        Forward, klein_map_oop_loss, BatchDuplicated,
        BatchDuplicated(A, A_tangents),
        BatchDuplicated(B, B_tangents),
        Const(Wg), Const(Wh), Const(threshold), Const(n_x),
    ) setup = begin
        A = $(problem.A)
        B = $(problem.B)
        A_tangents = $(problem.A_tangents)
        B_tangents = $(problem.B_tangents)
        Wg = $(problem.Wg)
        Wh = $(problem.Wh)
        threshold = $(problem.threshold)
        n_x = Val($(problem.n_x))
    end evals = 4

    g["enzyme_reverse"] = @benchmarkable Enzyme.autodiff(
        Reverse, klein_map_oop_loss, Active,
        Active(A), Active(B), Const(Wg), Const(Wh), Const(threshold),
        Const(n_x),
    ) setup = begin
        A = $(problem.A)
        B = $(problem.B)
        Wg = $(problem.Wg)
        Wh = $(problem.Wh)
        threshold = $(problem.threshold)
        n_x = Val($(problem.n_x))
    end evals = 4

    return g
end

KLEIN_SUITE = BenchmarkGroup()
KLEIN_SUITE["oop_heap_small"] = klein_map_oop_heap_group(klein_small_problem(1.0e-6, Val(4)))
KLEIN_SUITE["oop_heap_medium"] = klein_map_oop_heap_group(klein_medium_problem(1.0e-6, Val(4)))
KLEIN_SUITE["oop_heap_fvgq"] = klein_map_oop_heap_group(klein_fvgq_problem(1.0e-6, Val(4)))
KLEIN_SUITE["oop_heap_sw07pfeifer"] =
    klein_map_oop_heap_group(klein_sw07pfeifer_problem(1.0e-6, Val(4)))
KLEIN_SUITE["inplace_heap_small"] = klein_map_inplace_heap_group(
    klein_small_problem(1.0e-6, Val(4)),
)
KLEIN_SUITE["inplace_heap_medium"] = klein_map_inplace_heap_group(
    klein_medium_problem(1.0e-6, Val(4)),
)
KLEIN_SUITE["inplace_heap_fvgq"] = klein_map_inplace_heap_group(
    klein_fvgq_problem(1.0e-6, Val(4)),
)
KLEIN_SUITE["inplace_heap_sw07pfeifer"] = klein_map_inplace_heap_group(
    klein_sw07pfeifer_problem(1.0e-6, Val(4)),
)
KLEIN_SUITE["oop_static_small"] =
    klein_map_oop_static_group(klein_static_small_problem(1.0e-6, Val(4)))
KLEIN_SUITE
