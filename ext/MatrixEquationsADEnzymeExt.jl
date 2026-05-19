module MatrixEquationsADEnzymeExt

using MatrixEquations
using MatrixEquationsAD

import Enzyme.EnzymeRules
import MatrixEquations: gsylv, gsylvkr, lyapd
import MatrixEquationsAD:
    _lyapdkr_check!, _ordqz!, _symmetrize_square!, lyapdkr,
    lyapdkradjointsolve, lyapdkrfactor, lyapdkrsolve
using ConcreteStructs: @concrete
using Enzyme:
    Annotation, BatchDuplicated, BatchDuplicatedNoNeed, Const,
    Duplicated, DuplicatedNoNeed
using LinearAlgebra: LinearAlgebra, StridedMatrix, Symmetric, mul!, schur

include("enzyme_sylvester.jl")
include("enzyme_lyapunov.jl")
include("enzyme_lyapdkr.jl")
include("ordqz_derivatives.jl")
include("enzyme_ordqz.jl")

end
