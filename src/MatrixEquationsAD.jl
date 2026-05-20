module MatrixEquationsAD

using ConcreteStructs: @concrete
using LinearAlgebra:
    GeneralizedSchur, LU, UpperTriangular, checksquare, issuccess, ldiv!, lu!, mul!,
    ordschur!, schur

export klein_map, klein_map!, lyapdkr, ordqz, ordqz!

include("lyapdkr.jl")
include("ordqz.jl")
include("klein_map.jl")

end
