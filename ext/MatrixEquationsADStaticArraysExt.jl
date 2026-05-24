module MatrixEquationsADStaticArraysExt

using MatrixEquationsAD: MatrixEquationsAD
using StaticArrays: SMatrix

import MatrixEquationsAD: klein_map, lyapdkr

function klein_map(
        A::SMatrix{n, n, T}, B::SMatrix{n, n, T}, ::Val{n_x};
        threshold = 1.0e-6,
    ) where {n, T, n_x}
    if !(0 <= n_x <= n)
        throw(DimensionMismatch("Val(n_x) must be between 0 and matrix size $n"))
    end
    n_y = n - n_x
    result = MatrixEquationsAD.klein_map(Matrix(A), Matrix(B); threshold)
    if size(result.h_x, 1) != n_x
        throw(
            DimensionMismatch(
                "BK split produced n_x = $(size(result.h_x, 1)); expected $n_x",
            ),
        )
    end
    return (;
        g_x = SMatrix{n_y, n_x, T}(result.g_x),
        h_x = SMatrix{n_x, n_x, T}(result.h_x),
    )
end

function lyapdkr(A::SMatrix{n, n, T}, C::SMatrix{n, n, T}) where {n, T}
    X = MatrixEquationsAD.lyapdkr(Matrix(A), Matrix(C))
    return SMatrix{n, n, T}(X)
end

end
