if Sys.isapple() && Base.find_package("AppleAccelerate") !== nothing
    using AppleAccelerate
elseif Base.find_package("MKL") !== nothing
    using MKL
end

using BenchmarkTools
using MatrixEquationsAD

const SUITE = BenchmarkGroup()
const bench_dir = joinpath(pkgdir(MatrixEquationsAD), "benchmark")

SUITE["klein_map"] = include(joinpath(bench_dir, "klein_map.jl"))
SUITE["gsylv"] = include(joinpath(bench_dir, "gsylv.jl"))
SUITE["lyapd"] = include(joinpath(bench_dir, "lyapd.jl"))
SUITE["ared"] = include(joinpath(bench_dir, "ared.jl"))

SUITE
