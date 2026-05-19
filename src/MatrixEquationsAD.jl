module MatrixEquationsAD

using ConcreteStructs: @concrete
using LinearAlgebra:
    GeneralizedSchur, LU, checksquare, issuccess, ldiv!, lu!, ordschur!, schur
using LinearAlgebra.LAPACK: BlasInt, @blasfunc, chklapackerror, chkstride1, liblapack

export gges, gges!, lyapdkr, ordqz, ordqz!

include("lyapdkr.jl")
include("ordqz.jl")
include("gges.jl")

end
