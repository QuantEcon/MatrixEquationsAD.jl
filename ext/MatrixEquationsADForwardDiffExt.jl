module MatrixEquationsADForwardDiffExt

using ConcreteStructs: @concrete
using ForwardDiff: Dual, Partials, partials, value
using MatrixEquations
using MatrixEquationsAD
using LinearAlgebra: LinearAlgebra, StridedMatrix, Symmetric, mul!, schur

import MatrixEquations: gsylv, gsylvkr, lyapd
import MatrixEquationsAD: _ordqz!

include("forwarddiff_lyapunov.jl")
include("forwarddiff_sylvester.jl")
include("ordqz_derivatives.jl")
include("forwarddiff_ordqz.jl")

end
