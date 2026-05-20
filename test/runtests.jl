using MatrixEquations
using MatrixEquationsAD
using Test

@testset "MatrixEquationsAD.jl" begin
    include("readme_examples.jl")
    include("ordqz_fixtures.jl")
    include("riccati_fixtures.jl")
    include("test_lyapdkr.jl")
    include("test_ordqz.jl")
    include("test_enzyme_dlyap.jl")
    include("test_enzyme_lyapdkr.jl")
    include("test_enzyme_riccati.jl")
    include("test_enzyme_sylvester.jl")
    include("test_enzyme_sylvkr.jl")
    include("test_enzyme_ordqz.jl")
    include("test_forwarddiff_dlyap.jl")
    include("test_forwarddiff_lyapdkr.jl")
    include("test_forwarddiff_riccati.jl")
    include("test_forwarddiff_sylvester.jl")
    include("test_forwarddiff_sylvkr.jl")
    include("test_forwarddiff_ordqz.jl")
    include("test_oop_ordqz.jl")
    include("test_fvgq_ordqz.jl")
    include("test_dsge_qz_ad.jl")
    include("test_dsge_qz_ad_exact.jl")
    include("test_klein_map.jl")
end
