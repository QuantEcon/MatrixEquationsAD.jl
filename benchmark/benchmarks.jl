using BenchmarkTools
using Enzyme: Active, BatchDuplicated, Const, Duplicated, Enzyme, Forward, Reverse
using ForwardDiff
using LinearAlgebra: dot
using MatrixEquations: ared, gsylv, gsylvkr, lyapd
using MatrixEquationsAD

const _bdir = @__DIR__

include(joinpath(_bdir, "fixtures.jl"))

const SUITE = BenchmarkGroup()

SUITE["lyapd"]   = include(joinpath(_bdir, "lyapd.jl"))
SUITE["lyapdkr"] = include(joinpath(_bdir, "lyapdkr.jl"))
SUITE["ared"]    = include(joinpath(_bdir, "ared.jl"))
SUITE["gsylv"]   = include(joinpath(_bdir, "gsylv.jl"))
SUITE["gsylvkr"] = include(joinpath(_bdir, "gsylvkr.jl"))
SUITE["ordqz"]     = include(joinpath(_bdir, "ordqz.jl"))
SUITE["gges"]      = include(joinpath(_bdir, "gges.jl"))
SUITE["ordqz_oop"] = include(joinpath(_bdir, "ordqz_oop.jl"))
SUITE["gges_oop"]  = include(joinpath(_bdir, "gges_oop.jl"))

# Warmup forces Enzyme rule precompilation and ForwardDiff Dual dispatch so the
# timed runs don't measure compile time. Mirrors DP's warmup_problem.
function _warmup_lyapd(p)
    A = copy(p.A); C = copy(p.C); W = p.W
    A_bar = zero(A); C_bar = zero(C)
    lyapd_loss(A, C, W)
    Enzyme.autodiff(
        Reverse, lyapd_loss, Active,
        Duplicated(A, A_bar), Duplicated(C, C_bar), Const(W),
    )
    A_tans = ntuple(i -> copy(p.dA_lanes[i]), Val(LYAPD_ENZ_LANES))
    C_tans = ntuple(i -> copy(p.dC_lanes[i]), Val(LYAPD_ENZ_LANES))
    Enzyme.autodiff(
        Forward, lyapd_loss, BatchDuplicated,
        BatchDuplicated(A, A_tans), BatchDuplicated(C, C_tans), Const(W),
    )
    lyapd_loss(dual_matrix(p.A, p.dA_lanes), dual_matrix(p.C, p.dC_lanes), W)
    return nothing
end

function _warmup_lyapdkr(p)
    A = copy(p.A); C = copy(p.C); W = p.W
    A_bar = zero(A); C_bar = zero(C)
    lyapdkr_loss(A, C, W)
    Enzyme.autodiff(
        Reverse, lyapdkr_loss, Active,
        Duplicated(A, A_bar), Duplicated(C, C_bar), Const(W),
    )
    A_tans = ntuple(i -> copy(p.dA_lanes[i]), Val(LYAPDKR_ENZ_LANES))
    C_tans = ntuple(i -> copy(p.dC_lanes[i]), Val(LYAPDKR_ENZ_LANES))
    Enzyme.autodiff(
        Forward, lyapdkr_loss, BatchDuplicated,
        BatchDuplicated(A, A_tans), BatchDuplicated(C, C_tans), Const(W),
    )
    lyapdkr_loss(dual_matrix(p.A, p.dA_lanes), dual_matrix(p.C, p.dC_lanes), W)
    return nothing
end

function _warmup_ared(p)
    A = copy(p.A); B = copy(p.B); R = copy(p.R); Q = copy(p.Q)
    WX = p.WX; WF = p.WF
    A_bar = zero(A); B_bar = zero(B); R_bar = zero(R); Q_bar = zero(Q)
    ared_loss(A, B, R, Q, WX, WF)
    Enzyme.autodiff(
        Reverse, ared_loss, Active,
        Duplicated(A, A_bar), Duplicated(B, B_bar),
        Duplicated(R, R_bar), Duplicated(Q, Q_bar),
        Const(WX), Const(WF),
    )
    A_tans = ntuple(i -> copy(p.dA_lanes[i]), Val(ARED_ENZ_LANES))
    B_tans = ntuple(i -> copy(p.dB_lanes[i]), Val(ARED_ENZ_LANES))
    R_tans = ntuple(i -> copy(p.dR_lanes[i]), Val(ARED_ENZ_LANES))
    Q_tans = ntuple(i -> copy(p.dQ_lanes[i]), Val(ARED_ENZ_LANES))
    Enzyme.autodiff(
        Forward, ared_loss, BatchDuplicated,
        BatchDuplicated(A, A_tans), BatchDuplicated(B, B_tans),
        BatchDuplicated(R, R_tans), BatchDuplicated(Q, Q_tans),
        Const(WX), Const(WF),
    )
    ared_loss(
        dual_matrix(p.A, p.dA_lanes), dual_matrix(p.B, p.dB_lanes),
        dual_matrix(p.R, p.dR_lanes), dual_matrix(p.Q, p.dQ_lanes), WX, WF,
    )
    return nothing
end

function _warmup_gsylv(p)
    A = copy(p.A); B = copy(p.B); C = copy(p.C); D = copy(p.D); E = copy(p.E); W = p.W
    A_bar = zero(A); B_bar = zero(B); C_bar = zero(C); D_bar = zero(D); E_bar = zero(E)
    gsylv_loss(A, B, C, D, E, W)
    Enzyme.autodiff(
        Reverse, gsylv_loss, Active,
        Duplicated(A, A_bar), Duplicated(B, B_bar), Duplicated(C, C_bar),
        Duplicated(D, D_bar), Duplicated(E, E_bar), Const(W),
    )
    A_tans = ntuple(i -> copy(p.dA_lanes[i]), Val(GSYLV_ENZ_LANES))
    B_tans = ntuple(i -> copy(p.dB_lanes[i]), Val(GSYLV_ENZ_LANES))
    C_tans = ntuple(i -> copy(p.dC_lanes[i]), Val(GSYLV_ENZ_LANES))
    D_tans = ntuple(i -> copy(p.dD_lanes[i]), Val(GSYLV_ENZ_LANES))
    E_tans = ntuple(i -> copy(p.dE_lanes[i]), Val(GSYLV_ENZ_LANES))
    Enzyme.autodiff(
        Forward, gsylv_loss, BatchDuplicated,
        BatchDuplicated(A, A_tans), BatchDuplicated(B, B_tans),
        BatchDuplicated(C, C_tans), BatchDuplicated(D, D_tans),
        BatchDuplicated(E, E_tans), Const(W),
    )
    gsylv_loss(
        dual_matrix(p.A, p.dA_lanes), dual_matrix(p.B, p.dB_lanes),
        dual_matrix(p.C, p.dC_lanes), dual_matrix(p.D, p.dD_lanes),
        dual_matrix(p.E, p.dE_lanes), W,
    )
    return nothing
end

function _warmup_gsylvkr(p)
    A = copy(p.A); B = copy(p.B); C = copy(p.C); D = copy(p.D); E = copy(p.E); W = p.W
    A_bar = zero(A); B_bar = zero(B); C_bar = zero(C); D_bar = zero(D); E_bar = zero(E)
    gsylvkr_loss(A, B, C, D, E, W)
    Enzyme.autodiff(
        Reverse, gsylvkr_loss, Active,
        Duplicated(A, A_bar), Duplicated(B, B_bar), Duplicated(C, C_bar),
        Duplicated(D, D_bar), Duplicated(E, E_bar), Const(W),
    )
    A_tans = ntuple(i -> copy(p.dA_lanes[i]), Val(GSYLVKR_ENZ_LANES))
    B_tans = ntuple(i -> copy(p.dB_lanes[i]), Val(GSYLVKR_ENZ_LANES))
    C_tans = ntuple(i -> copy(p.dC_lanes[i]), Val(GSYLVKR_ENZ_LANES))
    D_tans = ntuple(i -> copy(p.dD_lanes[i]), Val(GSYLVKR_ENZ_LANES))
    E_tans = ntuple(i -> copy(p.dE_lanes[i]), Val(GSYLVKR_ENZ_LANES))
    Enzyme.autodiff(
        Forward, gsylvkr_loss, BatchDuplicated,
        BatchDuplicated(A, A_tans), BatchDuplicated(B, B_tans),
        BatchDuplicated(C, C_tans), BatchDuplicated(D, D_tans),
        BatchDuplicated(E, E_tans), Const(W),
    )
    gsylvkr_loss(
        dual_matrix(p.A, p.dA_lanes), dual_matrix(p.B, p.dB_lanes),
        dual_matrix(p.C, p.dC_lanes), dual_matrix(p.D, p.dD_lanes),
        dual_matrix(p.E, p.dE_lanes), W,
    )
    return nothing
end

function _warmup_ordqz(p)
    A = copy(p.A); B = copy(p.B)
    threshold = p.threshold
    A_bar = zero(A); B_bar = zero(B)
    Enzyme.autodiff(
        Reverse, ordqz_reverse_loss, Active,
        Duplicated(A, A_bar), Duplicated(B, B_bar),
        Const(threshold),
    )
    A2 = copy(p.A); B2 = copy(p.B)
    S = zeros(size(A)); T = zeros(size(B)); Q = zeros(size(A)); Z = zeros(size(A))
    A_tans = ntuple(i -> copy(p.dA_lanes[i]), Val(ORDQZ_ENZ_LANES))
    B_tans = ntuple(i -> copy(p.dB_lanes[i]), Val(ORDQZ_ENZ_LANES))
    S_tans = ntuple(_ -> zeros(size(A)), Val(ORDQZ_ENZ_LANES))
    T_tans = ntuple(_ -> zeros(size(B)), Val(ORDQZ_ENZ_LANES))
    Q_tans = ntuple(_ -> zeros(size(A)), Val(ORDQZ_ENZ_LANES))
    Z_tans = ntuple(_ -> zeros(size(A)), Val(ORDQZ_ENZ_LANES))
    Enzyme.autodiff(
        Forward, ordqz_forward_loss!, BatchDuplicated,
        BatchDuplicated(S, S_tans), BatchDuplicated(T, T_tans),
        BatchDuplicated(Q, Q_tans), BatchDuplicated(Z, Z_tans),
        BatchDuplicated(A2, A_tans), BatchDuplicated(B2, B_tans),
        Const(threshold),
    )
    ordqz_reverse_loss(dual_matrix(p.A, p.dA_lanes), dual_matrix(p.B, p.dB_lanes), threshold)
    return nothing
end

function _warmup_gges(p)
    A = copy(p.A); B = copy(p.B)
    criterium = (1 - p.threshold)^2
    A_bar = zero(A); B_bar = zero(B)
    Enzyme.autodiff(
        Reverse, gges_reverse_loss, Active,
        Duplicated(A, A_bar), Duplicated(B, B_bar),
        Const(criterium),
    )
    A2 = copy(p.A); B2 = copy(p.B)
    S = zeros(size(A)); T = zeros(size(B)); Q = zeros(size(A)); Z = zeros(size(A))
    A_tans = ntuple(i -> copy(p.dA_lanes[i]), Val(GGES_ENZ_LANES))
    B_tans = ntuple(i -> copy(p.dB_lanes[i]), Val(GGES_ENZ_LANES))
    S_tans = ntuple(_ -> zeros(size(A)), Val(GGES_ENZ_LANES))
    T_tans = ntuple(_ -> zeros(size(B)), Val(GGES_ENZ_LANES))
    Q_tans = ntuple(_ -> zeros(size(A)), Val(GGES_ENZ_LANES))
    Z_tans = ntuple(_ -> zeros(size(A)), Val(GGES_ENZ_LANES))
    Enzyme.autodiff(
        Forward, gges_forward_loss!, BatchDuplicated,
        BatchDuplicated(S, S_tans), BatchDuplicated(T, T_tans),
        BatchDuplicated(Q, Q_tans), BatchDuplicated(Z, Z_tans),
        BatchDuplicated(A2, A_tans), BatchDuplicated(B2, B_tans),
        Const(criterium),
    )
    gges_reverse_loss(dual_matrix(p.A, p.dA_lanes), dual_matrix(p.B, p.dB_lanes), criterium)
    return nothing
end

for p in (lyapd_small_problem(), lyapd_medium_problem())
    _warmup_lyapd(p)
    _warmup_lyapdkr(p)
end
for p in (ared_small_problem(), ared_medium_problem())
    _warmup_ared(p)
end
for p in (gsylv_small_problem(), gsylv_medium_problem())
    _warmup_gsylv(p)
    _warmup_gsylvkr(p)
end
function _warmup_qz_oop_heap(p)
    A = copy(p.A); B = copy(p.B); thr = p.threshold; crit = (1 - thr)^2
    gges_oop_loss(A, B, crit)
    ordqz_oop_loss(A, B, thr)
    dA = zero(A); dB = zero(B)
    Enzyme.autodiff(
        Reverse, gges_oop_loss, Active,
        Duplicated(A, dA), Duplicated(B, dB), Const(crit),
    )
    dA .= 0; dB .= 0
    Enzyme.autodiff(
        Reverse, ordqz_oop_loss, Active,
        Duplicated(A, dA), Duplicated(B, dB), Const(thr),
    )
    A_tans = ntuple(i -> copy(p.dA_lanes[i]), Val(GGES_OOP_ENZ_LANES))
    B_tans = ntuple(i -> copy(p.dB_lanes[i]), Val(GGES_OOP_ENZ_LANES))
    Enzyme.autodiff(
        Forward, gges_oop_loss, BatchDuplicated,
        BatchDuplicated(A, A_tans), BatchDuplicated(B, B_tans), Const(crit),
    )
    Enzyme.autodiff(
        Forward, ordqz_oop_loss, BatchDuplicated,
        BatchDuplicated(A, A_tans), BatchDuplicated(B, B_tans), Const(thr),
    )
    gges_oop_loss(dual_matrix(p.A, p.dA_lanes), dual_matrix(p.B, p.dB_lanes), crit)
    ordqz_oop_loss(dual_matrix(p.A, p.dA_lanes), dual_matrix(p.B, p.dB_lanes), thr)
    return nothing
end

function _warmup_qz_oop_static(p)
    A = p.A; B = p.B; thr = p.threshold; crit = (1 - thr)^2
    gges_oop_loss(A, B, crit)
    ordqz_oop_loss(A, B, thr)
    Enzyme.autodiff(Reverse, gges_oop_loss, Active, Active(A), Active(B), Const(crit))
    Enzyme.autodiff(Reverse, ordqz_oop_loss, Active, Active(A), Active(B), Const(thr))
    A_tans = p.dA_lanes; B_tans = p.dB_lanes
    Enzyme.autodiff(
        Forward, gges_oop_loss, BatchDuplicated,
        BatchDuplicated(A, A_tans), BatchDuplicated(B, B_tans), Const(crit),
    )
    Enzyme.autodiff(
        Forward, ordqz_oop_loss, BatchDuplicated,
        BatchDuplicated(A, A_tans), BatchDuplicated(B, B_tans), Const(thr),
    )
    return nothing
end

for p in (ordqz_small_problem(), ordqz_medium_problem())
    _warmup_ordqz(p)
    _warmup_gges(p)
    _warmup_qz_oop_heap(p)
end
_warmup_qz_oop_static(ordqz_static_problem())

SUITE
