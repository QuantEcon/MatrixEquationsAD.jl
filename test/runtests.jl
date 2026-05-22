using MatrixEquations
using MatrixEquationsAD
using Test

include("symmetric_matrix_utils.jl")
include("readme_examples.jl")
include("test_lyapdkr.jl")
include("test_enzyme_dlyap.jl")
include("test_enzyme_lyapdkr.jl")
include("test_enzyme_riccati.jl")
include("test_enzyme_sylvester.jl")
include("test_enzyme_sylvkr.jl")
include("test_forwarddiff_dlyap.jl")
include("test_forwarddiff_lyapdkr.jl")
include("test_forwarddiff_riccati.jl")
include("test_forwarddiff_sylvester.jl")
include("test_forwarddiff_sylvkr.jl")
include("test_klein_map.jl")
include("test_forwarddiff_klein_map.jl")
include("test_enzyme_klein_map.jl")
include("test_lyapd_inplace.jl")
include("test_differentiationinterface.jl")
