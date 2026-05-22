# Regenerate `benchmark/baseline.json` from the current `SUITE`.
#
# Usage:
#
#     julia --project=benchmark benchmark/run_baseline.jl
#
# Reads the existing baseline (if any) for comparison and writes a fresh
# one alongside this script. To diff a new result against the committed
# baseline in a session, do:
#
#     using BenchmarkTools
#     include("benchmark/benchmarks.jl")
#     tune!(SUITE)
#     results = run(SUITE; verbose = false)
#     baseline = BenchmarkTools.load(joinpath(@__DIR__, "baseline.json"))[1]
#     judge(median(results), median(baseline))
#
# The committed baseline is a low-effort regression tracker — re-run
# this script whenever the benchmark surface changes and review the diff.

using BenchmarkTools

const BASELINE_PATH = joinpath(@__DIR__, "baseline.json")

include(joinpath(@__DIR__, "benchmarks.jl"))

tune!(SUITE)
results = run(SUITE; verbose = false)

BenchmarkTools.save(BASELINE_PATH, results)
@info "baseline written" path = BASELINE_PATH
