module MatrixEquationsAD

using ConcreteStructs: @concrete
using LinearAlgebra:
    LU, UpperTriangular, checksquare, issuccess, ldiv!, lu!, mul!, ordschur!, schur

export klein_map, klein_map!, lyapdkr

include("lyapdkr.jl")
include("klein_map.jl")

end
