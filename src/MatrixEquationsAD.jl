module MatrixEquationsAD

using LinearAlgebra: GeneralizedSchur, ordschur!, schur

export ordqz, ordqz!

include("ordqz.jl")

end
