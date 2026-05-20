module MatrixEquationsADForwardDiffExt

using ConcreteStructs: @concrete
using ForwardDiff: Dual, Partials, partials, value
using MatrixEquations
using MatrixEquationsAD
using LinearAlgebra:
    LinearAlgebra, StridedMatrix, Symmetric,
    ldiv!, lu, lu!, mul!, schur, transpose!

import MatrixEquations: ared, gsylv, gsylvkr, lyapd
import MatrixEquationsAD:
    DEFAULT_BK_THRESHOLD, _gges!, _gges_ordschur!, _lyapdkr_check!, _ordqz!,
    _symmetrize_square!, gges, lyapdkr, lyapdkrfactor, lyapdkrsolve, ordqz

include("forwarddiff_lyapunov.jl")
include("forwarddiff_lyapdkr.jl")
include("riccati_derivatives.jl")
include("forwarddiff_riccati.jl")
include("forwarddiff_sylvester.jl")
include("ordqz_derivatives.jl")
include("forwarddiff_ordqz.jl")
include("forwarddiff_gges.jl")
include("forwarddiff_qz_oop.jl")

end
