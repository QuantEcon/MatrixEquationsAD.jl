using Enzyme:
    Active, BatchDuplicated, BatchDuplicatedNoNeed, Const, Duplicated,
    DuplicatedNoNeed, Reverse, autodiff
using EnzymeTestUtils: test_forward, test_reverse
using FiniteDifferences: central_fdm, grad
using LinearAlgebra: I, dot, issymmetric, norm
using MatrixEquations
using MatrixEquationsAD
using Random: Random
using StaticArrays: SMatrix
using Test

include(joinpath(@__DIR__, "example_matrices", "fvgq.jl"))

# Test tier flag — see test/runtests.jl.
const _RUN_SLOW_TESTS = get(ENV, "RUN_SLOW_TESTS", "false") == "true"

function lyapdkr_weighted_sum(A, C, W)::Float64
    X = lyapdkr(A, C)
    return dot(W, X)
end

function lyapdkr_weighted_sum_ws(A, C, W, M_ws)::Float64
    X = lyapdkr(A, C; M_ws)
    return dot(W, X)
end

# Manual Enzyme reverse vs FD anchor. The full EnzymeTestUtils sweep
# is gated under `RUN_SLOW_TESTS` below.
@testset "lyapdkr Enzyme reverse vs FD — n=2 anchor" begin
    A = [0.55 0.08; -0.04 0.42]
    C = [1.0 0.2; 0.2 0.7]
    W = [0.3 -0.1; 0.2 0.5]
    X = lyapdkr(A, C)
    @test X ≈ lyapd(A, C)
    @test issymmetric(X)

    A_bar = zero(A); C_bar = zero(C)
    autodiff(
        Reverse, lyapdkr_weighted_sum, Active,
        Duplicated(copy(A), A_bar), Duplicated(copy(C), C_bar), Const(W),
    )

    fdm = central_fdm(5, 1)
    A_fd = reshape(grad(fdm, v -> lyapdkr_weighted_sum(reshape(v, 2, 2), C, W), vec(A))[1], 2, 2)
    C_fd = reshape(grad(fdm, v -> lyapdkr_weighted_sum(A, reshape(v, 2, 2), W), vec(C))[1], 2, 2)
    @test A_bar ≈ A_fd rtol = 1.0e-8
    @test C_bar ≈ C_fd rtol = 1.0e-8
end

if _RUN_SLOW_TESTS
@testset "lyapdkr Enzyme rules" begin
    A = [0.55 0.08; -0.04 0.42]
    C = [1.0 0.2; 0.2 0.7]
    W = [0.3 -0.1; 0.2 0.5]
    As = SMatrix{2, 2, Float64}(A)
    Cs = SMatrix{2, 2, Float64}(C)
    Ws = SMatrix{2, 2, Float64}(W)

    test_forward(
        lyapdkr, DuplicatedNoNeed, (A, Duplicated), (C, Duplicated)
    )
    test_forward(
        lyapdkr, BatchDuplicatedNoNeed, (A, BatchDuplicated), (C, BatchDuplicated)
    )
    test_reverse(
        lyapdkr, Duplicated, (A, Duplicated), (C, Duplicated)
    )
    test_reverse(
        lyapdkr_weighted_sum, Active,
        (A, Duplicated), (C, Duplicated), (W, Const)
    )

    test_forward(
        lyapdkr_weighted_sum, BatchDuplicated,
        (As, BatchDuplicated), (Cs, BatchDuplicated), (Ws, Const)
    )
end

@testset "lyapdkr Enzyme forward — SMatrix native (n=3)" begin
    A3 = SMatrix{3, 3, Float64}(
        [0.55 0.08 0.01; -0.04 0.42 0.05; 0.02 -0.03 0.36],
    )
    C3 = SMatrix{3, 3, Float64}(
        [1.0 0.2 0.1; 0.2 0.7 0.05; 0.1 0.05 0.5],
    )
    W3 = SMatrix{3, 3, Float64}(randn(3, 3))

    # n=3 random-tangent shadows are O(10); FD precision caps around
    # 1e-5 relative, so default 1e-9 is too tight.
    fd_kwargs = (atol = 1.0e-5, rtol = 1.0e-4)

    test_forward(
        lyapdkr, DuplicatedNoNeed,
        (A3, Duplicated), (C3, Duplicated); fd_kwargs...
    )
    test_forward(
        lyapdkr, BatchDuplicatedNoNeed,
        (A3, BatchDuplicated), (C3, BatchDuplicated); fd_kwargs...
    )
    test_forward(
        lyapdkr_weighted_sum, BatchDuplicated,
        (A3, BatchDuplicated), (C3, BatchDuplicated), (W3, Const);
        fd_kwargs...
    )
end
end  # if _RUN_SLOW_TESTS

if _RUN_SLOW_TESTS
    # FVGQ-scale (n=38) lyapdkr with the M_ws workspace path. The M⁻¹
    # solve is O((n²)³); each test_forward/test_reverse call compiles a
    # full Enzyme rule sweep against FD probes.
    @testset "lyapdkr Enzyme rules — M_ws workspace (FVGQ large)" begin
        Random.seed!(0x1d3f)
        fo = FVGQExampleMatrices.fvgq_first_order_inputs()
        A = fo.h_x
        B = fo.B_shock
        n = size(A, 1)
        C = B * B' + 1.0e-6 * I(n)
        W = randn(n, n)
        M_ws = Matrix{Float64}(undef, n * n, n * n)

        # FVGQ gradients are O(10^3-10^4); FD reference precision caps around
        # 1e-3 relative, so relax accordingly.
        fd_kwargs = (atol = 1.0e-2, rtol = 1.0e-3)

        test_forward(
            lyapdkr_weighted_sum_ws, Duplicated,
            (A, Duplicated), (C, Duplicated), (W, Const), (M_ws, Const);
            fd_kwargs...
        )
        test_forward(
            lyapdkr_weighted_sum_ws, BatchDuplicated,
            (A, BatchDuplicated), (C, BatchDuplicated), (W, Const), (M_ws, Const);
            fd_kwargs...
        )
        test_reverse(
            lyapdkr_weighted_sum_ws, Active,
            (A, Duplicated), (C, Duplicated), (W, Const), (M_ws, Const);
            fd_kwargs...
        )
    end
end

function lyapdkr_inplace_weighted_sum(X, A, C, W)::Float64
    lyapdkr!(X, A, C)
    return dot(W, X)
end

if _RUN_SLOW_TESTS
    # EnzymeTestUtils sweep on `lyapdkr!`. The n=2 reverse anchor at the
    # top of this file already covers the OOP form.
    @testset "lyapdkr! Enzyme rules" begin
        A = [0.55 0.08; -0.04 0.42]
        C = [1.0 0.2; 0.2 0.7]
        W = [0.3 -0.1; 0.2 0.5]

        X = similar(A)
        test_forward(
            lyapdkr!, Const,
            (X, Duplicated), (A, Duplicated), (C, Duplicated)
        )
        X = similar(A)
        test_forward(
            lyapdkr!, Const,
            (X, BatchDuplicated), (A, BatchDuplicated), (C, BatchDuplicated)
        )
        X = similar(A)
        test_reverse(
            lyapdkr_inplace_weighted_sum, Active,
            (X, Duplicated), (A, Duplicated), (C, Duplicated), (W, Const)
        )
    end
end
