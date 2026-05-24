using FiniteDifferences: central_fdm, jvp
using ForwardDiff
using ForwardDiff: Dual
using LinearAlgebra: I, issymmetric
using MatrixEquations
using MatrixEquationsAD
using StaticArrays: SMatrix
using Test

include(joinpath(@__DIR__, "example_matrices", "fvgq.jl"))

@testset "lyapdkr ForwardDiff rules" begin
    A = [0.55 0.08; -0.04 0.42]
    C = [1.0 0.2; 0.2 0.7]
    x = [vec(A); vec(C)]
    fdm = central_fdm(5, 1)

    function lyapdkr_vec(x)
        A_x = reshape(x[1:4], 2, 2)
        C_x = reshape(x[5:8], 2, 2)
        return vec(lyapdkr(A_x, C_x))
    end

    J = ForwardDiff.jacobian(lyapdkr_vec, x)
    @test lyapdkr_vec(x) ≈ vec(lyapdkr(A, C))

    for dx in (
            0.01 .* sin.(1:length(x)),
            0.01 .* cos.(2.0 .* collect(1:length(x))),
        )
        ad = reshape(J * dx, 2, 2)
        fd = reshape(jvp(fdm, lyapdkr_vec, (x, dx)), 2, 2)
        @test issymmetric(ad)
        @test ad ≈ fd
    end

    function lyapdkr_static_vec(x)
        A_x = SMatrix{2, 2, eltype(x)}(reshape(x[1:4], 2, 2))
        C_x = SMatrix{2, 2, eltype(x)}(reshape(x[5:8], 2, 2))
        return vec(lyapdkr(A_x, C_x))
    end

    J_static = ForwardDiff.jacobian(lyapdkr_static_vec, x)
    @test lyapdkr_static_vec(x) ≈ vec(lyapdkr(A, C))

    x_dual = map(v -> Dual{Nothing}(v, one(v)), x)
    A_dual = SMatrix{2, 2, eltype(x_dual)}(reshape(x_dual[1:4], 2, 2))
    C_dual = SMatrix{2, 2, eltype(x_dual)}(reshape(x_dual[5:8], 2, 2))
    X_dual = @inferred lyapdkr(A_dual, C_dual)
    @test X_dual isa SMatrix{2, 2, eltype(x_dual)}

    for dx in (
            0.01 .* sin.(3.0 .* collect(1:length(x))),
            0.01 .* cos.(4.0 .* collect(1:length(x))),
        )
        ad = reshape(J_static * dx, 2, 2)
        fd = reshape(jvp(fdm, lyapdkr_static_vec, (x, dx)), 2, 2)
        @test issymmetric(ad)
        @test ad ≈ fd
    end
end

@testset "lyapdkr ForwardDiff rules — SMatrix native (n=3)" begin
    A3 = [0.55 0.08 0.01; -0.04 0.42 0.05; 0.02 -0.03 0.36]
    C3 = [1.0 0.2 0.1; 0.2 0.7 0.05; 0.1 0.05 0.5]
    x = [vec(A3); vec(C3)]

    function lyapdkr_static_vec3(x::AbstractVector)
        A = SMatrix{3, 3, eltype(x)}(reshape(x[1:9], 3, 3))
        C = SMatrix{3, 3, eltype(x)}(reshape(x[10:18], 3, 3))
        return vec(lyapdkr(A, C))
    end
    function lyapdkr_heap_vec3(x::AbstractVector)
        A = reshape(x[1:9], 3, 3)
        C = reshape(x[10:18], 3, 3)
        return vec(lyapdkr(A, C))
    end

    @test lyapdkr_static_vec3(x) ≈ lyapdkr_heap_vec3(x)
    J_static = ForwardDiff.jacobian(lyapdkr_static_vec3, x)
    J_heap = ForwardDiff.jacobian(lyapdkr_heap_vec3, x)
    @test J_static ≈ J_heap

    # Inferred type — Dual SMatrix → Dual SMatrix
    A3s = SMatrix{3, 3, Float64}(A3)
    C3s = SMatrix{3, 3, Float64}(C3)
    x_dual = map(v -> Dual{Nothing}(v, one(v)), x)
    A_dual = SMatrix{3, 3, eltype(x_dual)}(reshape(x_dual[1:9], 3, 3))
    C_dual = SMatrix{3, 3, eltype(x_dual)}(reshape(x_dual[10:18], 3, 3))
    X_dual = @inferred lyapdkr(A_dual, C_dual)
    @test X_dual isa SMatrix{3, 3, <:Dual}
end

@testset "lyapdkr ForwardDiff rules — M_ws workspace (FVGQ large)" begin
    fo = FVGQExampleMatrices.fvgq_first_order_inputs()
    A = fo.h_x
    B = fo.B_shock
    n = size(A, 1)
    C = B * B' + 1.0e-6 * I(n)
    M_ws = Matrix{Float64}(undef, n * n, n * n)

    # Probe vector covering both A and C entries
    x = [vec(A); vec(C)]
    lyapdkr_vec(x) = vec(lyapdkr(reshape(x[1:(n * n)], n, n),
                                 reshape(x[(n * n + 1):end], n, n)))
    function lyapdkr_vec_ws(x)
        return vec(
            lyapdkr(
                reshape(x[1:(n * n)], n, n),
                reshape(x[(n * n + 1):end], n, n);
                M_ws,
            ),
        )
    end

    @test lyapdkr_vec_ws(x) ≈ lyapdkr_vec(x)

    J = ForwardDiff.jacobian(lyapdkr_vec, x)
    J_ws = ForwardDiff.jacobian(lyapdkr_vec_ws, x)
    @test J ≈ J_ws
end

@testset "lyapdkr! ForwardDiff rules" begin
    A = [0.55 0.08; -0.04 0.42]
    C = [1.0 0.2; 0.2 0.7]
    x = [vec(A); vec(C)]

    function lyapdkr_inplace_vec(x::AbstractVector{V}) where {V}
        A_x = reshape(x[1:4], 2, 2)
        C_x = reshape(x[5:8], 2, 2)
        Xout = Matrix{V}(undef, 2, 2)
        lyapdkr!(Xout, A_x, C_x)
        return vec(Xout)
    end

    J = ForwardDiff.jacobian(lyapdkr_inplace_vec, x)
    J_oop = ForwardDiff.jacobian(x -> vec(lyapdkr(
                reshape(x[1:4], 2, 2),
                reshape(x[5:8], 2, 2),
            )), x)
    @test J ≈ J_oop
end
