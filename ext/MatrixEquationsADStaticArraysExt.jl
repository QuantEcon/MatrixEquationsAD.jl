module MatrixEquationsADStaticArraysExt

using MatrixEquationsAD: MatrixEquationsAD
using StaticArrays: SMatrix

import MatrixEquationsAD: gges, ordqz

# Default constants reused from the main module.
const DEFAULT_BK_THRESHOLD = MatrixEquationsAD.DEFAULT_BK_THRESHOLD
const DEFAULT_GGES_CRITERIUM = (1 - DEFAULT_BK_THRESHOLD)^2

function gges(
        A::SMatrix{n, n, T}, B::SMatrix{n, n, T};
        select::Symbol = :ed, criterium = DEFAULT_GGES_CRITERIUM,
    ) where {n, T}
    Aheap = Matrix(A)
    Bheap = Matrix(B)
    result = MatrixEquationsAD.gges(Aheap, Bheap; select, criterium)
    return (;
        S = SMatrix{n, n, T}(result.S),
        T = SMatrix{n, n, T}(result.T),
        Q = SMatrix{n, n, T}(result.Q),
        Z = SMatrix{n, n, T}(result.Z),
        sdim = result.sdim,
    )
end

function ordqz(
        A::SMatrix{n, n, T}, B::SMatrix{n, n, T}, ordering::Symbol = :bk;
        threshold = DEFAULT_BK_THRESHOLD,
    ) where {n, T}
    Aheap = Matrix(A)
    Bheap = Matrix(B)
    result = MatrixEquationsAD.ordqz(Aheap, Bheap, ordering; threshold)
    return (;
        S = SMatrix{n, n, T}(result.S),
        T = SMatrix{n, n, T}(result.T),
        Q = SMatrix{n, n, T}(result.Q),
        Z = SMatrix{n, n, T}(result.Z),
        sdim = result.sdim,
    )
end

end
