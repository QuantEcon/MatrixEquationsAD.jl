module MatrixEquationsADForwardDiffStaticArraysExt

# ForwardDiff method for the static-native `lyapdkr` path. When inputs are
# SMatrices of `Dual{T, V, P}`, extract value matrices, take one static
# `lu(I − A⊗A)`, and reuse `F` across the primal and each of the `P`
# partial solves. Pack values + partials back into an `SMatrix{Dual}`.

using ForwardDiff: Dual, Partials, partials, value
using LinearAlgebra: lu
using MatrixEquationsAD: build_M!!, symmetrize!!
using StaticArrays: SMatrix

import MatrixEquationsAD: lyapdkr

function lyapdkr(
        A::SMatrix{N, N, <:Dual{T, V, P}},
        C::SMatrix{N, N, <:Dual{T, V, P}};
        M_ws = nothing,  # accepted for API parity with the heap path; ignored
    ) where {N, T, V <: Union{Float32, Float64}, P}
    Aval = SMatrix{N, N, V}(map(value, A))
    Cval = SMatrix{N, N, V}(map(value, C))
    M = build_M!!(nothing, Aval)
    F = lu(M)
    X = symmetrize!!(SMatrix{N, N, V}(F \ vec(Cval)))

    dXs = ntuple(Val(P)) do i
        Base.@_inline_meta
        dA = SMatrix{N, N, V}(map(x -> partials(x, i), A))
        dC = SMatrix{N, N, V}(map(x -> partials(x, i), C))
        rhs = dC + dA * X * Aval' + Aval * X * dA'
        symmetrize!!(SMatrix{N, N, V}(F \ vec(rhs)))
    end

    return SMatrix{N, N, Dual{T, V, P}}(
        ntuple(Val(N * N)) do idx
            Base.@_inline_meta
            Dual{T}(X[idx], Partials(ntuple(k -> dXs[k][idx], Val(P))))
        end,
    )
end

end
