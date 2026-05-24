module MatrixEquationsAD

using ConcreteStructs: @concrete
using LinearAlgebra:
    Symmetric, UpperTriangular, checksquare, ldiv!, lu!, mul!,
    ordschur!, schur
using MatrixEquations: MatrixEquations, lyapds!, sylvds!, utqu, utqu!

export klein_map, klein_map!, lyapd!, lyapdkr

include("lyapdkr.jl")
include("klein_map.jl")
include("lyapd.jl")

end
