module MatrixEquationsADEnzymeExt

using MatrixEquations
using MatrixEquationsAD

import Enzyme.EnzymeRules
import MatrixEquations: gsylv, gsylvkr, lyapd
import MatrixEquationsAD: _ordqz!
using ConcreteStructs: @concrete
using Enzyme:
    Annotation, BatchDuplicated, BatchDuplicatedNoNeed, Const,
    Duplicated, DuplicatedNoNeed
using LinearAlgebra: LinearAlgebra, StridedMatrix, Symmetric, mul!, schur

include("enzyme_sylvester.jl")
include("enzyme_lyapunov.jl")
include("ordqz_derivatives.jl")
include("enzyme_ordqz.jl")

end
