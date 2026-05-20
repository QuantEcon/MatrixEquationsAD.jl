using BenchmarkTools
using MatrixEquationsAD
using StaticArrays: SMatrix

function klein_map_oop_heap_group(problem)
    g = BenchmarkGroup()

    g["primal"] = @benchmarkable klein_map(A, B; threshold) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        threshold = $(problem.threshold)
    end evals = 1

    g["in_place"] = @benchmarkable klein_map!(g_x, h_x, A, B; threshold) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        g_x = zeros(size(A, 1) - $(problem.n_x), $(problem.n_x))
        h_x = zeros($(problem.n_x), $(problem.n_x))
        threshold = $(problem.threshold)
    end evals = 1

    return g
end

function klein_map_oop_static_group(problem)
    g = BenchmarkGroup()

    g["primal"] = @benchmarkable klein_map(A, B; threshold) setup = begin
        A = $(problem.A)
        B = $(problem.B)
        threshold = $(problem.threshold)
    end evals = 1

    return g
end

g = BenchmarkGroup()
g["heap_small"] = klein_map_oop_heap_group(klein_small_problem())
g["heap_medium"] = klein_map_oop_heap_group(klein_medium_problem())
g["heap_fvgq"] = klein_map_oop_heap_group(klein_fvgq_problem())
g["heap_sw07pfeifer"] = klein_map_oop_heap_group(klein_sw07pfeifer_problem())
g["static_small"] = klein_map_oop_static_group(klein_static_small_problem())
g
