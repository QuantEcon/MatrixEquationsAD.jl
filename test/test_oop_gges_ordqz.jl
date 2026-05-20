using Enzyme: Active, Const, Duplicated, Enzyme, Forward, Reverse, autodiff
using FiniteDifferences: central_fdm, jvp
using ForwardDiff
using LinearAlgebra: I, norm
using MatrixEquationsAD
using StaticArrays: SMatrix, @SMatrix
using Test

const _oop_fdm = central_fdm(5, 1)

function _oop_gges_loss(A, B)
    r = gges(A, B; select = :ed, criterium = (1 - 1.0e-6)^2)
    return sum(abs2, r.Q * r.S * r.Z') + 0.7 * sum(abs2, r.Q * r.T * r.Z')
end

function _oop_ordqz_loss(A, B)
    r = ordqz(A, B, :bk; threshold = 1.0e-6)
    return sum(abs2, r.Q * r.S * r.Z') + 0.7 * sum(abs2, r.Q * r.T * r.Z')
end

@testset "out-of-place gges/ordqz wrappers" begin
    A = [1.6 0.2 0.1; 0.0 0.35 -0.1; 0.0 0.0 1.9]
    B = [1.0 0.1 0.0; 0.0 1.2 0.2; 0.0 0.0 0.8]

    @testset "heap primal interface" begin
        rg = gges(A, B; select = :ed, criterium = (1 - 1.0e-6)^2)
        @test rg.S isa Matrix{Float64}
        @test rg.sdim == 2
        @test A ≈ rg.Q * rg.S * rg.Z'
        @test B ≈ rg.Q * rg.T * rg.Z'

        ro = ordqz(A, B, :bk; threshold = 1.0e-6)
        @test ro.S isa Matrix{Float64}
        @test ro.sdim == 2
        @test A ≈ ro.Q * ro.S * ro.Z'
        @test B ≈ ro.Q * ro.T * ro.Z'
    end

    @testset "ForwardDiff dispatch (heap)" begin
        rng_seed_dA = 0.1 .* [0.5 -0.3 0.1; 0.2 0.4 -0.1; 0.0 0.3 0.2]
        rng_seed_dB = 0.1 .* [0.1 -0.2 0.0; 0.3 0.5 0.1; -0.1 0.0 0.2]
        A_dual = map(A, rng_seed_dA) do a, da
            ForwardDiff.Dual{Nothing}(a, (da,))
        end
        B_dual = map(B, rng_seed_dB) do b, db
            ForwardDiff.Dual{Nothing}(b, (db,))
        end
        rg = gges(A_dual, B_dual; select = :ed, criterium = (1 - 1.0e-6)^2)
        rg_v = gges(A, B; select = :ed, criterium = (1 - 1.0e-6)^2)
        @test map(ForwardDiff.value, rg.S) ≈ rg_v.S
        @test rg.sdim == 2

        ro = ordqz(A_dual, B_dual, :bk; threshold = 1.0e-6)
        ro_v = ordqz(A, B, :bk; threshold = 1.0e-6)
        @test map(ForwardDiff.value, ro.S) ≈ ro_v.S
        @test ro.sdim == 2
    end

    @testset "Enzyme heap reverse" begin
        dA = zero(A); dB = zero(B)
        autodiff(Reverse, _oop_gges_loss, Active, Duplicated(A, dA), Duplicated(B, dB))
        fd = jvp(
            _oop_fdm, (Av, Bv) -> _oop_gges_loss(Av, Bv),
            (A, dA), (B, dB),
        )
        # First-order check: ∇f · g = f'(x)g where g = ∇f from reverse mode
        @test fd ≈ sum(abs2, dA) + sum(abs2, dB)  rtol=1.0e-4

        dA2 = zero(A); dB2 = zero(B)
        autodiff(Reverse, _oop_ordqz_loss, Active, Duplicated(A, dA2), Duplicated(B, dB2))
        @test dA ≈ dA2
        @test dB ≈ dB2
    end

    @testset "Enzyme heap forward matches FD" begin
        dA = 0.1 .* [0.5 -0.3 0.1; 0.2 0.4 -0.1; 0.0 0.3 0.2]
        dB = 0.1 .* [0.1 -0.2 0.0; 0.3 0.5 0.1; -0.1 0.0 0.2]
        fwd_g = only(autodiff(Forward, _oop_gges_loss, Duplicated(A, dA), Duplicated(B, dB)))
        fwd_o = only(autodiff(Forward, _oop_ordqz_loss, Duplicated(A, dA), Duplicated(B, dB)))
        fd = jvp(
            _oop_fdm, (Av, Bv) -> _oop_gges_loss(Av, Bv),
            (A, dA), (B, dB),
        )
        @test fwd_g ≈ fd  rtol=1.0e-4
        @test fwd_o ≈ fd  rtol=1.0e-4
    end

    @testset "SMatrix primal interface" begin
        As = SMatrix{3, 3, Float64}(A)
        Bs = SMatrix{3, 3, Float64}(B)
        rg = gges(As, Bs; select = :ed, criterium = (1 - 1.0e-6)^2)
        @test rg.S isa SMatrix{3, 3, Float64}
        @test rg.T isa SMatrix{3, 3, Float64}
        @test rg.Q isa SMatrix{3, 3, Float64}
        @test rg.Z isa SMatrix{3, 3, Float64}
        @test rg.sdim == 2
        @test Matrix(As) ≈ rg.Q * rg.S * rg.Z'

        ro = ordqz(As, Bs, :bk; threshold = 1.0e-6)
        @test ro.S isa SMatrix{3, 3, Float64}
        @test ro.sdim == 2
    end

    @testset "Enzyme SMatrix paths match heap" begin
        As = SMatrix{3, 3, Float64}(A)
        Bs = SMatrix{3, 3, Float64}(B)

        # Heap reference gradient
        dA = zero(A); dB = zero(B)
        autodiff(Reverse, _oop_gges_loss, Active, Duplicated(A, dA), Duplicated(B, dB))

        # SMatrix reverse: Active SMatrix returns SMatrix gradients
        res = autodiff(Reverse, _oop_gges_loss, Active, Active(As), Active(Bs))
        @test Matrix(res[1][1]) ≈ dA
        @test Matrix(res[1][2]) ≈ dB

        res_o = autodiff(Reverse, _oop_ordqz_loss, Active, Active(As), Active(Bs))
        @test Matrix(res_o[1][1]) ≈ dA
        @test Matrix(res_o[1][2]) ≈ dB

        # SMatrix forward
        dA_dir = 0.1 .* [0.5 -0.3 0.1; 0.2 0.4 -0.1; 0.0 0.3 0.2]
        dB_dir = 0.1 .* [0.1 -0.2 0.0; 0.3 0.5 0.1; -0.1 0.0 0.2]
        dAs = SMatrix{3, 3, Float64}(dA_dir)
        dBs = SMatrix{3, 3, Float64}(dB_dir)
        fwd_h = only(autodiff(
            Forward, _oop_gges_loss, Duplicated(A, dA_dir), Duplicated(B, dB_dir),
        ))
        fwd_s = only(autodiff(
            Forward, _oop_gges_loss, Duplicated(As, dAs), Duplicated(Bs, dBs),
        ))
        @test fwd_h ≈ fwd_s
    end

    @testset "ForwardDiff SMatrix path" begin
        As = SMatrix{3, 3, Float64}(A)
        Bs = SMatrix{3, 3, Float64}(B)
        dA = 0.1 .* [0.5 -0.3 0.1; 0.2 0.4 -0.1; 0.0 0.3 0.2]
        dB = 0.1 .* [0.1 -0.2 0.0; 0.3 0.5 0.1; -0.1 0.0 0.2]
        As_dual = SMatrix{3, 3}(map(As, dA) do a, da
            ForwardDiff.Dual{Nothing}(a, (da,))
        end)
        Bs_dual = SMatrix{3, 3}(map(Bs, dB) do b, db
            ForwardDiff.Dual{Nothing}(b, (db,))
        end)
        rg = gges(As_dual, Bs_dual; select = :ed, criterium = (1 - 1.0e-6)^2)
        @test rg.S isa SMatrix{3, 3}
        rg_v = gges(As, Bs; select = :ed, criterium = (1 - 1.0e-6)^2)
        @test all(ForwardDiff.value.(rg.S) .≈ rg_v.S)
    end
end
