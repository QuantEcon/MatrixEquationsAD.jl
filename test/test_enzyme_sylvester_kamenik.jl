# Primal correctness + Enzyme reverse rule for gsylv_kamenik /
# gsylv_kamenik!.
#
# Primal is checked against four DSGE-scale fixtures (RBC 5×2, RBC_SV 8×4,
# SGU 15×7, FVGQ 38×14) captured in
# `test/example_matrices/sylvester_kamenik.jl`, with `X_ref` obtained from
# a dense Kronecker direct solve at capture time.
#
# Enzyme reverse for both the allocating `gsylv_kamenik` and the in-place
# `gsylv_kamenik!` is cross-checked against `FiniteDifferences.central_fdm(5,1)`
# on the RBC fixture (smallest — FD probes cheap, deep inside linear regime)
# and smoke-exercised on SGU (m=7) and FVGQ (m=14, has complex Schur
# sub-blocks → also exercises the w==2 path inside the primal).

using Enzyme: Active, Const, Duplicated, Reverse, autodiff
using FiniteDifferences: central_fdm, grad
using LinearAlgebra: I, kron, norm
using MatrixEquationsAD: gsylv_kamenik, gsylv_kamenik!
using Test

include(joinpath(@__DIR__, "example_matrices", "sylvester_kamenik.jl"))
using .SylvesterKamenikFixtures:
    rbc_second_order_sylvester_inputs,
    rbc_sv_second_order_sylvester_inputs,
    sgu_second_order_sylvester_inputs,
    fvgq_second_order_sylvester_inputs,
    kamenik_threshold

const FIXTURES = (
    ("rbc",    rbc_second_order_sylvester_inputs),
    ("rbc_sv", rbc_sv_second_order_sylvester_inputs),
    ("sgu",    sgu_second_order_sylvester_inputs),
    ("fvgq",   fvgq_second_order_sylvester_inputs),
)

@testset "gsylv_kamenik primal — DSGE-scale fixtures" begin
    for (name, builder) in FIXTURES
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

        # Same problem via the in-place form: D becomes X. Use a fresh
        # copy so the fixture stays intact.
        D_copy = copy(D)
        gsylv_kamenik!(D_copy, A, B, C)
        @test norm(A * D_copy + B * D_copy * H2 - D) / norm(D) ≤ kamenik_threshold
        @test norm(D_copy - X_ref) / norm(X_ref) ≤ kamenik_threshold
    end
end

# Scalar loss for AD checks. ½‖X‖² has cotangent X̄ = X, well-conditioned
# for FD comparison.
kamenik_loss(A, B, C, D) = sum(abs2, gsylv_kamenik(A, B, C, D)) / 2

function kamenik_loss!(A, B, C, D)
    gsylv_kamenik!(D, A, B, C)
    return sum(abs2, D) / 2
end

@testset "gsylv_kamenik Enzyme reverse vs FD — allocating form, RBC" begin
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

@testset "gsylv_kamenik! Enzyme reverse vs FD — in-place form, RBC" begin
    (; A, B, C, D) = rbc_second_order_sylvester_inputs()
    n = size(A, 1); m = size(C, 1)

    fdm = central_fdm(5, 1)
    A_fd = reshape(grad(fdm, v -> kamenik_loss!(reshape(v, n, n), B, C, copy(D)), vec(A))[1], n, n)
    B_fd = reshape(grad(fdm, v -> kamenik_loss!(A, reshape(v, n, n), C, copy(D)), vec(B))[1], n, n)
    C_fd = reshape(grad(fdm, v -> kamenik_loss!(A, B, reshape(v, m, m), copy(D)), vec(C))[1], m, m)
    D_fd = reshape(grad(fdm, v -> kamenik_loss!(A, B, C, reshape(copy(v), n, m^2)), vec(D))[1], n, m^2)

    A_sh = zero(A); B_sh = zero(B); C_sh = zero(C)
    D_work = copy(D); D_sh = zero(D_work)
    autodiff(Reverse, kamenik_loss!, Active,
        Duplicated(A, A_sh),
        Duplicated(B, B_sh),
        Duplicated(C, C_sh),
        Duplicated(D_work, D_sh))

    rel(a, b) = norm(a - b) / max(norm(b), eps())
    @test rel(A_sh, A_fd) ≤ 1.0e-9
    @test rel(B_sh, B_fd) ≤ 1.0e-9
    @test rel(C_sh, C_fd) ≤ 1.0e-9
    @test rel(D_sh, D_fd) ≤ 1.0e-9
end

@testset "gsylv_kamenik Enzyme reverse smoke — SGU / FVGQ" begin
    for (name, builder) in (
            ("sgu",  sgu_second_order_sylvester_inputs),
            ("fvgq", fvgq_second_order_sylvester_inputs),
        )
        (; A, B, C, D) = builder()
        A_sh = zero(A); B_sh = zero(B); C_sh = zero(C); D_sh = zero(D)
        autodiff(Reverse, kamenik_loss, Active,
            Duplicated(A, A_sh),
            Duplicated(B, B_sh),
            Duplicated(C, C_sh),
            Duplicated(D, D_sh))
        # FD is skipped at this scale — high curvature + many directional
        # probes. The smoke checks survival + nonzero cotangent flow.
        @test all(isfinite, A_sh)
        @test all(isfinite, B_sh)
        @test all(isfinite, C_sh)
        @test all(isfinite, D_sh)
        @test norm(D_sh) > 0
        @info "gsylv_kamenik smoke" model = name Dbar_norm = norm(D_sh)
    end
end

@testset "gsylv_kamenik! Enzyme reverse smoke — SGU / FVGQ" begin
    for (name, builder) in (
            ("sgu",  sgu_second_order_sylvester_inputs),
            ("fvgq", fvgq_second_order_sylvester_inputs),
        )
        (; A, B, C, D) = builder()
        A_sh = zero(A); B_sh = zero(B); C_sh = zero(C)
        D_work = copy(D); D_sh = zero(D_work)
        autodiff(Reverse, kamenik_loss!, Active,
            Duplicated(A, A_sh),
            Duplicated(B, B_sh),
            Duplicated(C, C_sh),
            Duplicated(D_work, D_sh))
        @test all(isfinite, A_sh)
        @test all(isfinite, B_sh)
        @test all(isfinite, C_sh)
        @test all(isfinite, D_sh)
        @test norm(D_sh) > 0
        @info "gsylv_kamenik! smoke" model = name Dbar_norm = norm(D_sh)
    end
end
