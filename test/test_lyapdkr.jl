using LinearAlgebra: I, issymmetric, norm
using MatrixEquations
using MatrixEquationsAD
using StaticArrays: SMatrix
using Test

include(joinpath(@__DIR__, "example_matrices", "fvgq.jl"))

@testset "lyapdkr primal" begin
    A = [0.55 0.08; -0.04 0.42]
    C = [1.0 0.2; 0.2 0.7]
    X = lyapdkr(A, C)

    @test X ≈ lyapd(A, C)
    @test issymmetric(X)
    @test norm(A * X * A' - X + C) < 1.0e-12

    C_general = [1.0 -0.3; 0.5 0.7]
    C_symmetric_part = 0.5 .* (C_general + C_general')
    X_general = lyapdkr(A, C_general)
    @test X_general ≈ lyapd(A, C_symmetric_part)
    @test issymmetric(X_general)
    @test norm(A * X_general * A' - X_general + C_symmetric_part) < 1.0e-12

    As = SMatrix{2, 2, Float64}(A)
    Cs = SMatrix{2, 2, Float64}(C)
    Xs = @inferred lyapdkr(As, Cs)
    @test Xs isa SMatrix{2, 2, Float64}
    @test Matrix(Xs) ≈ lyapd(A, C)
    @test issymmetric(Matrix(Xs))
    @test norm(As * Xs * As' - Xs + Cs) < 1.0e-12
end

@testset "lyapdkr primal — M_ws workspace (FVGQ large)" begin
    fo = FVGQExampleMatrices.fvgq_first_order_inputs()
    A = fo.h_x
    B = fo.B_shock
    n = size(A, 1)
    C = B * B' + 1.0e-6 * I(n)

    X_alloc = lyapdkr(A, C)
    M_ws = Matrix{Float64}(undef, n * n, n * n)
    X_ws = lyapdkr(A, C; M_ws)

    @test X_ws == X_alloc                       # bit-equivalent: build_M!! overwrites M_ws
    @test issymmetric(X_ws)
    @test norm(A * X_ws * A' - X_ws + C) < 1.0e-8
end

@testset "lyapdkr! primal" begin
    A = [0.55 0.08; -0.04 0.42]
    C = [1.0 0.2; 0.2 0.7]
    X_oop = lyapdkr(A, C)

    X = similar(A)
    result = lyapdkr!(X, A, C)
    @test result === X                          # returns the buffer
    @test X == X_oop                            # bit-equivalent to OOP version

    # With M_ws
    M_ws = Matrix{Float64}(undef, 4, 4)
    X2 = similar(A)
    lyapdkr!(X2, A, C; M_ws)
    @test X2 == X_oop

    # Large fixture
    fo = FVGQExampleMatrices.fvgq_first_order_inputs()
    Al = fo.h_x
    Bl = fo.B_shock
    nl = size(Al, 1)
    Cl = Bl * Bl' + 1.0e-6 * I(nl)
    Xl = similar(Al)
    lyapdkr!(Xl, Al, Cl)
    @test Xl == lyapdkr(Al, Cl)
end
