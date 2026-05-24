module MatrixEquationsADEnzymeExt

using MatrixEquations
using MatrixEquationsAD

import Enzyme.EnzymeRules
import MatrixEquations: ared, gsylv, gsylvkr, lyapd
import MatrixEquationsAD:
    LyapDSchurCache, _build_lyapdkr_matrix!, _symmetrize_square!,
    klein_map, klein_map!, lyapd!, lyapdadjointsolve, lyapdfactor, lyapdkr,
    lyapdsolve
using ConcreteStructs: @concrete
using Enzyme:
    Annotation, BatchDuplicated, BatchDuplicatedNoNeed, Const,
    Duplicated, DuplicatedNoNeed
using LinearAlgebra:
    Diagonal, LinearAlgebra, StridedMatrix, Symmetric,
    ldiv!, lu, lu!, mul!, schur, transpose!

include("enzyme_sylvester.jl")
include("enzyme_lyapunov.jl")
include("enzyme_lyapdkr.jl")
include("riccati_derivatives.jl")
include("enzyme_riccati.jl")
include("klein_map_derivatives.jl")
include("enzyme_klein_map.jl")

end
