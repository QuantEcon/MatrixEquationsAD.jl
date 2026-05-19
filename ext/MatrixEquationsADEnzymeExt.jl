module MatrixEquationsADEnzymeExt

using MatrixEquations
using MatrixEquationsAD

import Enzyme.EnzymeRules
import MatrixEquations: ared, gsylv, gsylvkr, lyapd
import MatrixEquationsAD:
    _gges!, _gges_ordschur!, _lyapdkr_check!, _ordqz!, _symmetrize_square!, lyapdkr,
    lyapdkradjointsolve, lyapdkrfactor, lyapdkrsolve
using ConcreteStructs: @concrete
using Enzyme:
    Annotation, BatchDuplicated, BatchDuplicatedNoNeed, Const,
    Duplicated, DuplicatedNoNeed
using LinearAlgebra: Diagonal, LinearAlgebra, StridedMatrix, Symmetric, mul!, schur

include("enzyme_sylvester.jl")
include("enzyme_lyapunov.jl")
include("enzyme_lyapdkr.jl")
include("riccati_derivatives.jl")
include("enzyme_riccati.jl")
include("ordqz_derivatives.jl")
include("enzyme_ordqz.jl")

end
