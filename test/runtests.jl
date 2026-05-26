using MatrixEquations
using MatrixEquationsAD
using Test

# ─── Test tiers ──────────────────────────────────────────────────────────────
# Two tiers controlled by a single environment variable:
#
#   default (CI)           — primal correctness on all fixtures + one
#                            Enzyme-vs-FD cross-check per AD'd function.
#                            Target: ~4 min.
#   RUN_SLOW_TESTS=true    — adds EnzymeTestUtils sweeps, BatchDuplicated
#                            smoke, ForwardDiff Dual coverage on AD'd
#                            surfaces, FVGQ-scale (n=38) AD paths, the
#                            M_ws workspace tests, and AD tests for
#                            non-exported wrappers (gsylv, gsylvkr,
#                            sylvkr). Target: ~12 min.
#
# Inside individual test files, the flag is re-read so files can be
# `include`d standalone for quick iteration.

const RUN_SLOW_TESTS = get(ENV, "RUN_SLOW_TESTS", "false") == "true"

include("symmetric_matrix_utils.jl")
include("readme_examples.jl")
include("test_lyapd.jl")
include("test_lyapdkr.jl")
include("test_enzyme_dlyap.jl")
include("test_enzyme_lyapdkr.jl")
include("test_enzyme_riccati.jl")
if RUN_SLOW_TESTS
    include("test_enzyme_sylvester.jl")     # gsylv / gsylvkr — not exported.
end
include("test_enzyme_sylvester_kamenik.jl")
if RUN_SLOW_TESTS
    include("test_enzyme_sylvkr.jl")        # gsylvkr — not exported.
end
include("test_forwarddiff_dlyap.jl")
include("test_forwarddiff_lyapdkr.jl")
include("test_forwarddiff_riccati.jl")
if RUN_SLOW_TESTS
    include("test_forwarddiff_sylvester.jl")    # gsylv — not exported.
    include("test_forwarddiff_sylvkr.jl")       # gsylvkr — not exported.
end
include("test_klein_map.jl")
include("test_differentiationinterface.jl")
