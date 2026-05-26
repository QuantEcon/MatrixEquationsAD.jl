using BenchmarkTools
using Enzyme:
    Active, BatchDuplicated, Const, Duplicated, Enzyme, Forward, Reverse,
    make_zero
using LinearAlgebra: dot
using MatrixEquationsAD: gsylv_kamenik, gsylv_kamenik!

# Reuse the DSGE-scale fixtures from the test tree — single source of truth.
include(joinpath(pkgdir(MatrixEquationsAD), "test", "example_matrices", "sylvester_kamenik.jl"))
using .SylvesterKamenikFixtures:
    rbc_second_order_sylvester_inputs,
    rbc_sv_second_order_sylvester_inputs,
    sgu_second_order_sylvester_inputs,
    fvgq_second_order_sylvester_inputs

const BATCH_WIDTH = 4

function gsylv_kamenik_problem(builder)
    fix = builder()
    A, B, C, D = fix.A, fix.B, fix.C, fix.D
    W = randn(size(D)...)
    return (; A, B, C, D, W)
end

# ─── Allocating form: gsylv_kamenik(A, B, C, D) ──────────────────────────────
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

    # Forward (JVP) with `BatchDuplicated(BATCH_WIDTH)`. The Kamenik
    # factorisation is amortised across the primal + all `BATCH_WIDTH`
    # tangent solves, so the per-direction cost should run well below
    # one full primal.
    g["enzyme_forward"] = @benchmarkable Enzyme.autodiff(
        Forward, gsylv_kamenik, BatchDuplicated,
        BatchDuplicated(A, dAs),
        BatchDuplicated(B, dBs),
        BatchDuplicated(C, dCs),
        BatchDuplicated(D, dDs),
    ) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        C = copy($(problem.C))
        D = copy($(problem.D))
        dAs = ntuple(_ -> make_zero(A), Val($BATCH_WIDTH))
        dBs = ntuple(_ -> make_zero(B), Val($BATCH_WIDTH))
        dCs = ntuple(_ -> make_zero(C), Val($BATCH_WIDTH))
        dDs = ntuple(_ -> make_zero(D), Val($BATCH_WIDTH))
    end evals = 4

    return g
end

# ─── In-place form: gsylv_kamenik!(D, A, B, C) ───────────────────────────────
function gsylv_kamenik_loss!(A, B, C, D, W)
    gsylv_kamenik!(D, A, B, C)
    return dot(W, D)
end

function gsylv_kamenik_inplace_group(problem)
    g = BenchmarkGroup()

    g["primal!"] = @benchmarkable gsylv_kamenik_loss!(A, B, C, D, W) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        C = copy($(problem.C))
        D = copy($(problem.D))
        W = $(problem.W)
    end evals = 4

    g["enzyme_reverse!"] = @benchmarkable Enzyme.autodiff(
        Reverse, gsylv_kamenik_loss!, Active,
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

    g["enzyme_forward!"] = @benchmarkable Enzyme.autodiff(
        Forward, gsylv_kamenik!, Const,
        BatchDuplicated(D, dDs),
        BatchDuplicated(A, dAs),
        BatchDuplicated(B, dBs),
        BatchDuplicated(C, dCs),
    ) setup = begin
        A = copy($(problem.A))
        B = copy($(problem.B))
        C = copy($(problem.C))
        D = copy($(problem.D))
        dAs = ntuple(_ -> make_zero(A), Val($BATCH_WIDTH))
        dBs = ntuple(_ -> make_zero(B), Val($BATCH_WIDTH))
        dCs = ntuple(_ -> make_zero(C), Val($BATCH_WIDTH))
        dDs = ntuple(_ -> make_zero(D), Val($BATCH_WIDTH))
    end evals = 4

    return g
end

KAMENIK_SUITE = BenchmarkGroup()
KAMENIK_SUITE["gsylv_kamenik"] = BenchmarkGroup()
for (model, builder) in (
        ("rbc", rbc_second_order_sylvester_inputs),
        ("rbc_sv", rbc_sv_second_order_sylvester_inputs),
        ("sgu", sgu_second_order_sylvester_inputs),
        ("fvgq", fvgq_second_order_sylvester_inputs),
    )
    prob = gsylv_kamenik_problem(builder)
    g = gsylv_kamenik_group(prob)
    # Allocating + in-place benches share the same fixture problem.
    inplace = gsylv_kamenik_inplace_group(prob)
    for (k, v) in inplace
        g[k] = v
    end
    KAMENIK_SUITE["gsylv_kamenik"][model] = g
end
KAMENIK_SUITE
