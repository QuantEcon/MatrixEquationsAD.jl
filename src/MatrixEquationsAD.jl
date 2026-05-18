module MatrixEquationsAD

using LinearAlgebra: GeneralizedSchur, ordschur!, schur

export ordqz, ordqz!, qzselect_right_half_plane, qzselect_left_half_plane
export qzselect_outside_unit, qzselect_inside_unit

include("ordqz.jl")

end
