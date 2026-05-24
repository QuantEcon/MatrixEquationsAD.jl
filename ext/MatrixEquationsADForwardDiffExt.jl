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
    LyapDSchurCache, _build_lyapdkr_matrix!, _lyapdkr_check!, _symmetrize_square!,
    klein_map, klein_map!, lyapd!, lyapdfactor, lyapdkr, lyapdsolve

include("forwarddiff_lyapunov.jl")
include("forwarddiff_lyapdkr.jl")
include("riccati_derivatives.jl")
include("forwarddiff_riccati.jl")
include("forwarddiff_sylvester.jl")
include("klein_map_derivatives.jl")
include("forwarddiff_klein_map.jl")

end
