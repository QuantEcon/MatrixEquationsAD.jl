module MatrixEquationsADEnzymeExt

using MatrixEquations
using MatrixEquationsAD

import Enzyme.EnzymeRules
import MatrixEquations: ared, gsylv, gsylvkr, lyapd
import MatrixEquationsAD:
    DEFAULT_BK_THRESHOLD, _gges!, _gges_ordschur!, _lyapdkr_check!, _ordqz!,
    _symmetrize_square!, gges, lyapdkr, lyapdkradjointsolve, lyapdkrfactor,
    lyapdkrsolve, ordqz
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
include("ordqz_derivatives.jl")
include("enzyme_ordqz.jl")
include("enzyme_ordqz_oop.jl")

end
