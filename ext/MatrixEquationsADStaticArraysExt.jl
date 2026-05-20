module MatrixEquationsADStaticArraysExt

using MatrixEquationsAD: MatrixEquationsAD
using StaticArrays: SMatrix

import MatrixEquationsAD: klein_map, ordqz

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

function klein_map(
        A::SMatrix{n, n, T}, B::SMatrix{n, n, T};
        threshold = DEFAULT_BK_THRESHOLD,
    ) where {n, T}
    result = MatrixEquationsAD.klein_map(Matrix(A), Matrix(B); threshold)
    n_x = size(result.h_x, 1)
    n_y = n - n_x
    return (;
        g_x = SMatrix{n_y, n_x, T}(result.g_x),
        h_x = SMatrix{n_x, n_x, T}(result.h_x),
    )
end

end
