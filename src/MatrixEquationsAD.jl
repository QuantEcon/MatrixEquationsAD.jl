module MatrixEquationsAD

using ConcreteStructs: @concrete
using LinearAlgebra:
    I, Symmetric, UpperTriangular, checksquare, kron!, ldiv!, lu, lu!, mul!,
    ordschur!, schur
using MatrixEquations: MatrixEquations, lyapds!, sylvds!, utqu, utqu!

export klein_map, klein_map!, lyapd!, lyapdkr, lyapdkr!, gsylv_kamenik, gsylv_kamenik!

include("lyapdkr.jl")
include("klein_map.jl")
include("lyapd.jl")
include("sylvester_kamenik.jl")

end
