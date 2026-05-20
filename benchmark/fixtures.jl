# Benchmark fixtures: small + medium per-solver problem instances plus
# pre-randomized tangent lanes for ForwardDiff/Enzyme. Tangents are seeded
# off a fixed MersenneTwister so timings are reproducible across runs.
#
# Small ordqz/gges fixtures reuse DP-derived literals from test/ordqz_fixtures.jl.
# Medium fixtures use SGU first-order outputs (h_x, B_shock) and the SGU
# ordered-QZ (A, B) pair pasted into _sgu_raw.jl; the gsylv and ared problems
# at medium size are constructed at SGU dimensions but with synthetic
# diagonally-dominant operators so the solvers stay well-conditioned.

using ForwardDiff: Dual
using LinearAlgebra: I
using Random: MersenneTwister, randn
using StaticArrays: SMatrix

include(joinpath(@__DIR__, "..", "test", "ordqz_fixtures.jl"))
include(joinpath(@__DIR__, "_sgu_raw.jl"))

const BENCH_TANGENT_LANES = 4
const BENCH_SEED = 0xBE57AB1E

function _random_symmetric(rng, n, scale = 0.1)
    M = scale .* randn(rng, n, n)
    return 0.5 .* (M + M')
end

function _lanes(rng, dims, lanes = BENCH_TANGENT_LANES)
    return ntuple(_ -> randn(rng, dims...), Val(lanes))
end

# Build a Dual matrix whose value is A and whose N partials come from `lanes`.
# Equivalent to the test-file `map(A, lanes...) do a, ds...; Dual(a, ds...) end`
# pattern, but lifted into a named function so callers don't need to use a
# do-block lambda.
function dual_matrix(A, lanes::NTuple{N, <:AbstractMatrix}) where {N}
    return map(A, lanes...) do a, ds...
        Dual{Nothing}(a, ds...)
    end
end

# ------------------------------------------------------------------
# lyapd / lyapdkr
# ------------------------------------------------------------------

function lyapd_small_problem()
    rng = MersenneTwister(BENCH_SEED)
    A = [
        0.55 0.08 0.01 -0.02 0.00
        -0.04 0.42 0.05 0.01 -0.01
        0.02 -0.03 0.36 0.04 0.00
        0.00 0.02 -0.05 0.48 0.03
        -0.01 0.00 0.02 -0.04 0.51
    ]
    C0 = randn(rng, 5, 5)
    C = 0.5 .* (C0 + C0') + 5 * I  # symmetric, well-conditioned
    W = randn(rng, 5, 5)
    dA_lanes = _lanes(rng, (5, 5))
    dC_lanes = ntuple(_ -> _random_symmetric(rng, 5), Val(BENCH_TANGENT_LANES))
    return (; A, C, W, dA_lanes, dC_lanes)
end

function lyapd_medium_problem()
    rng = MersenneTwister(BENCH_SEED + 1)
    A = sgu_h_x()
    B = sgu_B_shock()
    C = B * B' + 1.0e-6 * I  # tiny regularization to keep C strictly PD
    W = randn(rng, size(A))
    dA_lanes = _lanes(rng, size(A))
    dC_lanes = ntuple(_ -> _random_symmetric(rng, size(C, 1)), Val(BENCH_TANGENT_LANES))
    return (; A, C, W, dA_lanes, dC_lanes)
end

# ------------------------------------------------------------------
# ared (discrete-time algebraic Riccati)
# ------------------------------------------------------------------

function ared_small_problem()
    rng = MersenneTwister(BENCH_SEED + 10)
    A = Matrix([0.95 0.0; 0.0 0.95]')
    B = Matrix([0.5 0.0; 0.0 0.5]')
    R = Matrix(0.2I, 2, 2)
    Q = Matrix(0.5I, 2, 2)
    WX = _random_symmetric(rng, size(Q, 1), 1.0)
    WF = randn(rng, size(B, 2), size(A, 1))  # F is (n_u, n_x)
    dA_lanes = _lanes(rng, size(A))
    dB_lanes = _lanes(rng, size(B))
    dR_lanes = ntuple(_ -> _random_symmetric(rng, size(R, 1)), Val(BENCH_TANGENT_LANES))
    dQ_lanes = ntuple(_ -> _random_symmetric(rng, size(Q, 1)), Val(BENCH_TANGENT_LANES))
    return (; A, B, R, Q, WX, WF, dA_lanes, dB_lanes, dR_lanes, dQ_lanes)
end

function ared_medium_problem()
    rng = MersenneTwister(BENCH_SEED + 11)
    n_x = 7
    n_u = 3
    # Synthetic stable A at SGU size; B from SGU's B_shock shape.
    A = 0.9 .* Matrix{Float64}(I, n_x, n_x) .+ 0.02 .* randn(rng, n_x, n_x)
    B = sgu_B_shock()
    R = Matrix(0.2I, n_u, n_u)
    Q = Matrix(0.5I, n_x, n_x)
    WX = _random_symmetric(rng, n_x, 1.0)
    WF = randn(rng, n_u, n_x)
    dA_lanes = _lanes(rng, size(A))
    dB_lanes = _lanes(rng, size(B))
    dR_lanes = ntuple(_ -> _random_symmetric(rng, size(R, 1)), Val(BENCH_TANGENT_LANES))
    dQ_lanes = ntuple(_ -> _random_symmetric(rng, size(Q, 1)), Val(BENCH_TANGENT_LANES))
    return (; A, B, R, Q, WX, WF, dA_lanes, dB_lanes, dR_lanes, dQ_lanes)
end

# ------------------------------------------------------------------
# gsylv / gsylvkr  (solves A*X*B + C*X*D = E)
# ------------------------------------------------------------------

function gsylv_small_problem()
    rng = MersenneTwister(BENCH_SEED + 20)
    A = Matrix([4.0 0.1 0.0; -0.2 3.6 0.3; 0.1 0.0 3.8])
    C = Matrix(0.2I, 3, 3)
    B = Matrix([3.0 0.2; -0.1 2.7])
    D = Matrix(0.3I, 2, 2)
    E = [1.0 -0.4; 0.3 0.8; -0.2 0.5]
    W = randn(rng, size(E))
    dA_lanes = _lanes(rng, size(A))
    dB_lanes = _lanes(rng, size(B))
    dC_lanes = _lanes(rng, size(C))
    dD_lanes = _lanes(rng, size(D))
    dE_lanes = _lanes(rng, size(E))
    return (; A, B, C, D, E, W, dA_lanes, dB_lanes, dC_lanes, dD_lanes, dE_lanes)
end

function gsylv_medium_problem()
    rng = MersenneTwister(BENCH_SEED + 21)
    nA, nB = 7, 5
    # Diagonally dominant ⇒ joint spectrum well-separated, gsylv well-posed.
    A = 4.0 .* Matrix{Float64}(I, nA, nA) .+ 0.1 .* randn(rng, nA, nA)
    C = 0.2 .* Matrix{Float64}(I, nA, nA)
    B = 3.0 .* Matrix{Float64}(I, nB, nB) .+ 0.1 .* randn(rng, nB, nB)
    D = 0.3 .* Matrix{Float64}(I, nB, nB)
    E = randn(rng, nA, nB)
    W = randn(rng, nA, nB)
    dA_lanes = _lanes(rng, size(A))
    dB_lanes = _lanes(rng, size(B))
    dC_lanes = _lanes(rng, size(C))
    dD_lanes = _lanes(rng, size(D))
    dE_lanes = _lanes(rng, size(E))
    return (; A, B, C, D, E, W, dA_lanes, dB_lanes, dC_lanes, dD_lanes, dE_lanes)
end

# ------------------------------------------------------------------
# ordqz / gges
# ------------------------------------------------------------------

function ordqz_small_problem()
    rng = MersenneTwister(BENCH_SEED + 30)
    A, B, sdim = dp_rbc_ordqz_problem()
    threshold = dp_ordqz_threshold
    dA_lanes = _lanes(rng, size(A))
    dB_lanes = _lanes(rng, size(B))
    return (; A, B, sdim, threshold, dA_lanes, dB_lanes)
end

function ordqz_medium_problem()
    rng = MersenneTwister(BENCH_SEED + 31)
    A = sgu_ordqz_A()
    B = sgu_ordqz_B()
    sdim = 7
    threshold = dp_ordqz_threshold
    dA_lanes = _lanes(rng, size(A))
    dB_lanes = _lanes(rng, size(B))
    return (; A, B, sdim, threshold, dA_lanes, dB_lanes)
end

# Static (SMatrix) fixtures — only the very-small RBC 5×5 ordered-QZ pair.
function ordqz_static_problem()
    rng = MersenneTwister(BENCH_SEED + 32)
    A, B, sdim = dp_rbc_ordqz_problem()
    n = size(A, 1)
    As = SMatrix{n, n, Float64}(A)
    Bs = SMatrix{n, n, Float64}(B)
    threshold = dp_ordqz_threshold
    dA_lanes = ntuple(_ -> SMatrix{n, n, Float64}(randn(rng, n, n)), Val(BENCH_TANGENT_LANES))
    dB_lanes = ntuple(_ -> SMatrix{n, n, Float64}(randn(rng, n, n)), Val(BENCH_TANGENT_LANES))
    return (; A = As, B = Bs, sdim, threshold, dA_lanes, dB_lanes)
end
