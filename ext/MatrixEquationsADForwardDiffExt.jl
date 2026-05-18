module MatrixEquationsADForwardDiffExt

using ConcreteStructs: @concrete
using ForwardDiff
using MatrixEquations
using MatrixEquationsAD
using LinearAlgebra: LinearAlgebra, StridedMatrix, Symmetric, mul!, schur

const FDDual{T, V, N} = ForwardDiff.Dual{T, V, N}
const FDDualMatrix{T, V, N} = StridedMatrix{<:FDDual{T, V, N}}

include("forwarddiff_lyapunov.jl")
include("forwarddiff_sylvester.jl")
include("ordqz_derivatives.jl")
include("forwarddiff_ordqz.jl")

end
