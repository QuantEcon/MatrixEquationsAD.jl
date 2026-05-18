using MatrixEquations
using MatrixEquationsAD
using Test

@testset "MatrixEquationsAD.jl" begin
    include("ordqz_fixtures.jl")
    include("test_ordqz.jl")
    include("test_enzyme_dlyap.jl")
    include("test_enzyme_sylvester.jl")
    include("test_enzyme_sylvkr.jl")
    include("test_enzyme_ordqz.jl")
    include("test_forwarddiff_dlyap.jl")
    include("test_forwarddiff_sylvester.jl")
    include("test_forwarddiff_sylvkr.jl")
    include("test_forwarddiff_ordqz.jl")
end
