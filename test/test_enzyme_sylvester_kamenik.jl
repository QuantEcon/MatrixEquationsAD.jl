# Primal correctness + Enzyme reverse rule for gsylv_kamenik.
#
# Primal is checked against three DSGE-scale fixtures (RBC 5×2, RBC_SV 8×4,
# SGU 15×7) captured in `test/example_matrices/sylvester_kamenik.jl`, with
# `X_ref` obtained from a dense Kronecker direct solve at capture time.
# Enzyme reverse is cross-checked against `FiniteDifferences.central_fdm(5,1)`
# on the RBC fixture (smallest — keeps FD probes cheap and inside the linear
# regime) and exercised on the SGU fixture (largest — exercises the realistic
# m=7 path).

using Enzyme: Active, Const, Duplicated, Reverse, autodiff
using FiniteDifferences: central_fdm, grad
using LinearAlgebra: I, kron, norm
using MatrixEquationsAD: gsylv_kamenik
using Test

include(joinpath(@__DIR__, "example_matrices", "sylvester_kamenik.jl"))
using .SylvesterKamenikFixtures:
    rbc_second_order_sylvester_inputs,
    rbc_sv_second_order_sylvester_inputs,
    sgu_second_order_sylvester_inputs,
    kamenik_threshold

@testset "gsylv_kamenik primal — DSGE-scale fixtures" begin
    for (name, builder) in (
            ("rbc",    rbc_second_order_sylvester_inputs),
            ("rbc_sv", rbc_sv_second_order_sylvester_inputs),
            ("sgu",    sgu_second_order_sylvester_inputs),
        )
        (; A, B, C, D, X_ref) = builder()
        H2 = kron(C, C)

        # Sanity-check the embedded reference solution.
        @test norm(A * X_ref + B * X_ref * H2 - D) / norm(D) ≤ kamenik_threshold

        X = gsylv_kamenik(A, B, C, D)
        res = norm(A * X + B * X * H2 - D) / norm(D)
        relerr = norm(X - X_ref) / norm(X_ref)
        @info "gsylv_kamenik" model = name residual = res relerr = relerr
        @test res ≤ kamenik_threshold
        @test relerr ≤ kamenik_threshold
    end
end

# Scalar loss for AD checks. ½‖X‖² has cotangent X̄ = X, which makes the
# resulting gradients well-conditioned for FD comparison.
kamenik_loss(A, B, C, D) = sum(abs2, gsylv_kamenik(A, B, C, D)) / 2

@testset "gsylv_kamenik Enzyme reverse vs FD — RBC fixture" begin
    (; A, B, C, D) = rbc_second_order_sylvester_inputs()
    n = size(A, 1); m = size(C, 1)

    fdm = central_fdm(5, 1)
    A_fd = reshape(grad(fdm, v -> kamenik_loss(reshape(v, n, n), B, C, D), vec(A))[1], n, n)
    B_fd = reshape(grad(fdm, v -> kamenik_loss(A, reshape(v, n, n), C, D), vec(B))[1], n, n)
    C_fd = reshape(grad(fdm, v -> kamenik_loss(A, B, reshape(v, m, m), D), vec(C))[1], m, m)
    D_fd = reshape(grad(fdm, v -> kamenik_loss(A, B, C, reshape(v, n, m^2)), vec(D))[1], n, m^2)

    A_sh = zero(A); B_sh = zero(B); C_sh = zero(C); D_sh = zero(D)
    autodiff(Reverse, kamenik_loss, Active,
        Duplicated(A, A_sh),
        Duplicated(B, B_sh),
        Duplicated(C, C_sh),
        Duplicated(D, D_sh))

    rel(a, b) = norm(a - b) / max(norm(b), eps())
    @test rel(A_sh, A_fd) ≤ 1.0e-9
    @test rel(B_sh, B_fd) ≤ 1.0e-9
    @test rel(C_sh, C_fd) ≤ 1.0e-9
    @test rel(D_sh, D_fd) ≤ 1.0e-9
end

@testset "gsylv_kamenik Enzyme reverse smoke — SGU fixture (m=7)" begin
    (; A, B, C, D) = sgu_second_order_sylvester_inputs()
    A_sh = zero(A); B_sh = zero(B); C_sh = zero(C); D_sh = zero(D)
    autodiff(Reverse, kamenik_loss, Active,
        Duplicated(A, A_sh),
        Duplicated(B, B_sh),
        Duplicated(C, C_sh),
        Duplicated(D, D_sh))
    # The realistic SGU case mainly verifies the rule survives at scale —
    # FD is skipped (high curvature + 15² + 7²·15 directional probes).
    @test all(isfinite, A_sh)
    @test all(isfinite, B_sh)
    @test all(isfinite, C_sh)
    @test all(isfinite, D_sh)
    @test norm(D_sh) > 0      # cotangent flowed through to inputs
end
