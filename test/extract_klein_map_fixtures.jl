# Regenerate test/klein_map_fixtures.jl (and, idempotently, append
# dp_sw07pfeifer_first_order_pencil() to test/dsge_qz_fixtures.jl):
#
#   julia --project=<DP project> test/extract_klein_map_fixtures.jl
#
# This is a generator script — it is NOT included in runtests.jl. It depends
# on DifferentiablePerturbation.jl (DP) for two reasons:
#   1. ground-truth (g_x, h_x) computation via DP's first_order_perturbation!
#   2. materializing the SW07Pfeifer pencil (we don't carry DP's symbolic
#      assembly machinery in this repo)
#
# Cross-validation gate: after computing (g_x, h_x) on each pencil, the
# generator checks DP's published anchor values for RBC_SV (DP
# test/first_order_perturbation.jl:138-152, 10 spot-checks at atol=1e-10)
# and SW07 (DP test/sw07/sw07_model.jl:64-75, 10 spot-checks at atol=1e-10).
# A failure stops the generator and the fixture file is not written.

using DifferentiablePerturbation: DifferentiablePerturbation, FirstOrderConstWorkspace,
    first_order_perturbation!, RBC_SV, SW07, SW07Pfeifer, sw07_parameter_vector
using LinearAlgebra: I, norm
using Printf

const REPO_DIR = normpath(joinpath(@__DIR__, ".."))
const PENCIL_FILE = joinpath(REPO_DIR, "test", "dsge_qz_fixtures.jl")
const FIXTURE_FILE = joinpath(REPO_DIR, "test", "klein_map_fixtures.jl")
const FVGQ_FILE = joinpath(REPO_DIR, "test", "fvgq_ordqz_fixture.jl")

include(PENCIL_FILE)
include(FVGQ_FILE)

# -------------------------------------------------------------------------
# DP-based (g_x, h_x) computation
# -------------------------------------------------------------------------

function dp_klein_solution(A::AbstractMatrix, B::AbstractMatrix, n_x::Int)
    n = size(A, 1)
    n_y = n - n_x
    g_x = zeros(n_y, n_x)
    h_x = zeros(n_x, n_x)
    C = zeros(n_x, n_x)            # unused but required by signature
    Q_mat = zeros(n_x, n)          # unused but required by signature
    const_ws = FirstOrderConstWorkspace(n_x, n_y; n_z = n_x, n_epsilon = n_x)
    first_order_perturbation!(g_x, h_x, C, Q_mat, copy(A), copy(B), const_ws)
    return (; g_x, h_x)
end

# -------------------------------------------------------------------------
# Cross-validation anchors (DP test files)
# -------------------------------------------------------------------------

function check_rbc_sv_anchor(sol)
    g_x = sol.g_x; h_x = sol.h_x
    atol = 1.0e-10
    checks = (
        (h_x[1, 1], 0.9568351489231561, "h_x[1,1]"),
        (h_x[1, 2], 1.2418742011511346, "h_x[1,2]"),
        (h_x[1, 4], 0.11372859706759705, "h_x[1,4]"),
        (h_x[2, 2], 0.2, "h_x[2,2]"),
        (h_x[2, 4], 0.018315638888734168, "h_x[2,4]"),
        (h_x[3, 3], 0.9, "h_x[3,3]"),
        (g_x[1, 1], 0.09579643002421252, "g_x[1,1]"),
        (g_x[1, 2], 0.1349373930517625, "g_x[1,2]"),
        (g_x[2, 2], 1.3768115942028973, "g_x[2,2]"),
        (g_x[4, 1], -0.023164851076844073, "g_x[4,1]"),
    )
    for (got, expected, label) in checks
        if !isapprox(got, expected; atol)
            error(
                "RBC_SV anchor mismatch at $label: got $got, expected $expected " *
                    "(|diff|=$(abs(got - expected)))",
            )
        end
    end
    @info "RBC_SV anchors OK (10 checks)"
    return nothing
end

function check_sw07_anchor(sol)
    h_x = sol.h_x; g_x = sol.g_x
    coarse_atol = 1.0e-6
    fine_atol = 1.0e-10
    coarse = (
        (h_x[12, 12], 0.9977, "h_x[12,12]"),
        (h_x[13, 13], 0.5799, "h_x[13,13]"),
        (h_x[14, 14], 0.9957, "h_x[14,14]"),
        (h_x[15, 15], 0.7165, "h_x[15,15]"),
    )
    fine = (
        (h_x[1, 12], 0.8289571727240924, "h_x[1,12]"),
        (h_x[2, 1], -0.28813568284395347, "h_x[2,1]"),
        (h_x[4, 12], 0.009663148175603426, "h_x[4,12]"),
        (h_x[7, 15], 3.521449996172597, "h_x[7,15]"),
        (h_x[8, 1], -0.3665018568647873, "h_x[8,1]"),
        (g_x[1, 12], 0.4036468793051167, "g_x[1,12]"),
        (g_x[5, 5], 0.05427590265538092, "g_x[5,5]"),
        (g_x[10, 10], 0.8077853813490226, "g_x[10,10]"),
        (g_x[7, 15], 0.25316933941423964, "g_x[7,15]"),
    )
    for (got, expected, label) in coarse
        if !isapprox(got, expected; atol = coarse_atol)
            error("SW07 anchor mismatch at $label: got $got, expected $expected")
        end
    end
    for (got, expected, label) in fine
        if !isapprox(got, expected; atol = fine_atol)
            error("SW07 anchor mismatch at $label: got $got, expected $expected")
        end
    end
    @info "SW07 anchors OK (13 checks)"
    return nothing
end

# -------------------------------------------------------------------------
# Pencil capture for SW07Pfeifer (Path A)
# -------------------------------------------------------------------------

function sw07pfeifer_pencil()
    n_x = SW07Pfeifer.n_x
    n_y = SW07Pfeifer.n_y
    n_eps = SW07Pfeifer.n_epsilon
    n_z = SW07Pfeifer.n_z
    n = n_x + n_y
    A = zeros(n, n)
    B = zeros(n, n)
    Gamma = zeros(n_eps, n_eps)
    Omega = zeros(n_z)
    B_shock = zeros(n_x, n_eps)
    p = sw07_parameter_vector()
    SW07Pfeifer.first_order_assembly!(A, B, Gamma, Omega, B_shock, p)
    return (; A, B, n_x)
end

function sw07_pencil()
    n_x = SW07.n_x
    n_y = SW07.n_y
    n_eps = SW07.n_epsilon
    n_z = SW07.n_z
    n = n_x + n_y
    A = zeros(n, n)
    B = zeros(n, n)
    Gamma = zeros(n_eps, n_eps)
    Omega = zeros(n_z)
    B_shock = zeros(n_x, n_eps)
    p = sw07_parameter_vector()
    SW07.first_order_assembly!(A, B, Gamma, Omega, B_shock, p)
    return (; A, B, n_x)
end

# -------------------------------------------------------------------------
# Pretty-print helpers (cloned from test/extract_dsge_qz_ad_fixtures.jl)
# -------------------------------------------------------------------------

function _format_float(io::IO, x::Float64)
    if x == 0.0
        print(io, signbit(x) ? "-0.0" : "0.0")
    else
        Base.print(io, repr(x))
    end
    return nothing
end

function _format_matrix(io::IO, M::AbstractMatrix{Float64}; indent::Int = 8)
    n, m = size(M)
    print(io, "[\n")
    pad = " "^indent
    for i in 1:n
        print(io, pad)
        for j in 1:m
            _format_float(io, M[i, j])
            if j < m
                print(io, " ")
            end
        end
        if i < n
            print(io, ";")
        end
        print(io, "\n")
    end
    print(io, " "^(indent - 4), "]")
    return nothing
end

function _emit_named_matrix(io::IO, name::Symbol, M::AbstractMatrix{Float64}; indent::Int = 4)
    print(io, " "^indent, name, " = ")
    _format_matrix(io, M; indent = indent + 4)
    print(io, ",\n")
    return nothing
end

function _emit_fixture(io::IO, const_name::Symbol, sol)
    print(io, "const ", const_name, " = (;\n")
    _emit_named_matrix(io, :g_x, sol.g_x)
    _emit_named_matrix(io, :h_x, sol.h_x)
    print(io, ")\n")
    return nothing
end

# -------------------------------------------------------------------------
# Idempotent append of dp_sw07pfeifer_first_order_pencil() to dsge_qz_fixtures.jl
# -------------------------------------------------------------------------

function append_sw07pfeifer_pencil_if_missing(p)
    existing = read(PENCIL_FILE, String)
    if occursin("dp_sw07pfeifer_first_order_pencil", existing)
        @info "sw07pfeifer pencil already present in dsge_qz_fixtures.jl — skipping append"
        return nothing
    end
    open(PENCIL_FILE, "a") do io
        print(io, "\n")
        print(io, "function dp_sw07pfeifer_first_order_pencil()\n")
        print(io, "    A = ")
        _format_matrix(io, p.A; indent = 8)
        print(io, "\n")
        print(io, "    B = ")
        _format_matrix(io, p.B; indent = 8)
        print(io, "\n")
        print(io, "    return A, B, ", p.n_x, "\n")
        print(io, "end\n")
    end
    @info "Appended dp_sw07pfeifer_first_order_pencil() to" PENCIL_FILE
    return nothing
end

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

function main()
    @info "Computing klein solutions"

    rbc_A, rbc_B, rbc_n_x = dp_rbc_first_order_pencil()
    sol_rbc = dp_klein_solution(rbc_A, rbc_B, rbc_n_x)

    rbc_sv_A, rbc_sv_B, rbc_sv_n_x = dp_rbc_sv_first_order_pencil()
    sol_rbc_sv = dp_klein_solution(rbc_sv_A, rbc_sv_B, rbc_sv_n_x)
    check_rbc_sv_anchor(sol_rbc_sv)

    sgu_A, sgu_B, sgu_n_x = dp_sgu_first_order_pencil()
    sol_sgu = dp_klein_solution(sgu_A, sgu_B, sgu_n_x)

    fvgq_A = fvgq_ordqz_problem_A()
    fvgq_B = fvgq_ordqz_problem_B()
    # FVGQ pencil: sdim = 14 from existing test/test_fvgq_ordqz.jl:47
    sol_fvgq = dp_klein_solution(fvgq_A, fvgq_B, 14)

    sw07p = sw07pfeifer_pencil()
    sol_sw07p = dp_klein_solution(sw07p.A, sw07p.B, sw07p.n_x)

    # Cross-validation anchor: SW07 (not Pfeifer) per plan
    sw07 = sw07_pencil()
    sol_sw07 = dp_klein_solution(sw07.A, sw07.B, sw07.n_x)
    check_sw07_anchor(sol_sw07)

    append_sw07pfeifer_pencil_if_missing(sw07p)

    open(FIXTURE_FILE, "w") do io
        print(
            io,
            """
            # AUTO-GENERATED by test/extract_klein_map_fixtures.jl.
            # Do NOT edit by hand — regenerate via:
            #   julia --project=<DP project> test/extract_klein_map_fixtures.jl
            #
            # Ground-truth (g_x, h_x) policy-function matrices for each DSGE
            # pencil, computed by DifferentiablePerturbation's
            # first_order_perturbation!. Used by test/test_klein_map.jl as
            # pure value targets — no DP dependency at test time.

            """,
        )
        _emit_fixture(io, :KLEIN_RBC, sol_rbc); print(io, "\n")
        _emit_fixture(io, :KLEIN_RBC_SV, sol_rbc_sv); print(io, "\n")
        _emit_fixture(io, :KLEIN_SGU, sol_sgu); print(io, "\n")
        _emit_fixture(io, :KLEIN_FVGQ, sol_fvgq); print(io, "\n")
        _emit_fixture(io, :KLEIN_SW07PFEIFER, sol_sw07p)
    end
    @info "Wrote fixture file" FIXTURE_FILE
    return nothing
end

main()
