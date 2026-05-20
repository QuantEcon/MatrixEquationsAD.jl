module MatrixEquationsADStaticArraysExt

using MatrixEquationsAD: MatrixEquationsAD
using StaticArrays: SMatrix

import MatrixEquationsAD: ordqz

# Default constants reused from the main module.
const DEFAULT_BK_THRESHOLD = MatrixEquationsAD.DEFAULT_BK_THRESHOLD

function ordqz(
        A::SMatrix{n, n, T}, B::SMatrix{n, n, T}, ordering::Symbol = :bk;
        threshold = DEFAULT_BK_THRESHOLD, regularize_A = 0,
    ) where {n, T}
    Aheap = Matrix(A)
    Bheap = Matrix(B)
    result = MatrixEquationsAD.ordqz(
        Aheap, Bheap, ordering; threshold, regularize_A,
    )
    return (;
        S = SMatrix{n, n, T}(result.S),
        T = SMatrix{n, n, T}(result.T),
        Q = SMatrix{n, n, T}(result.Q),
        Z = SMatrix{n, n, T}(result.Z),
        sdim = result.sdim,
    )
end

end
