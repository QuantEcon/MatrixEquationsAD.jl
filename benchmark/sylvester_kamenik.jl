using BenchmarkTools
using Enzyme: Active, Const, Duplicated, Enzyme, Reverse, make_zero
using LinearAlgebra: dot
using MatrixEquationsAD: gsylv_kamenik

# Reuse the DSGE-scale fixtures from the test tree — single source of truth.
include(joinpath(pkgdir(MatrixEquationsAD), "test", "example_matrices", "sylvester_kamenik.jl"))
using .SylvesterKamenikFixtures:
    rbc_second_order_sylvester_inputs,
    rbc_sv_second_order_sylvester_inputs,
    sgu_second_order_sylvester_inputs

function gsylv_kamenik_problem(builder)
    fix = builder()
    A, B, C, D = fix.A, fix.B, fix.C, fix.D
    W = randn(size(D)...)
    return (; A, B, C, D, W)
end

gsylv_kamenik_loss(A, B, C, D, W) = dot(W, gsylv_kamenik(A, B, C, D))

function gsylv_kamenik_group(problem)
    g = BenchmarkGroup()

    g["primal"] = @benchmarkable gsylv_kamenik_loss(A, B, C, D, W) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        C = copy($(problem.C))
        D = copy($(problem.D))
        W = $(problem.W)
    end evals = 4

    g["enzyme_reverse"] = @benchmarkable Enzyme.autodiff(
        Reverse, gsylv_kamenik_loss, Active,
        Duplicated(A, A_bar),
        Duplicated(B, B_bar),
        Duplicated(C, C_bar),
        Duplicated(D, D_bar),
        Const(W),
    ) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        C = copy($(problem.C))
        D = copy($(problem.D))
        A_bar = make_zero(A)
        B_bar = make_zero(B)
        C_bar = make_zero(C)
        D_bar = make_zero(D)
        W = $(problem.W)
    end evals = 4

    return g
end

KAMENIK_SUITE = BenchmarkGroup()
KAMENIK_SUITE["gsylv_kamenik"] = BenchmarkGroup()
KAMENIK_SUITE["gsylv_kamenik"]["rbc"]    = gsylv_kamenik_group(
    gsylv_kamenik_problem(rbc_second_order_sylvester_inputs))
KAMENIK_SUITE["gsylv_kamenik"]["rbc_sv"] = gsylv_kamenik_group(
    gsylv_kamenik_problem(rbc_sv_second_order_sylvester_inputs))
KAMENIK_SUITE["gsylv_kamenik"]["sgu"]    = gsylv_kamenik_group(
    gsylv_kamenik_problem(sgu_second_order_sylvester_inputs))
KAMENIK_SUITE
