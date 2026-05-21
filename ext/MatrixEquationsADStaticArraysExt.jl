module MatrixEquationsADStaticArraysExt

using MatrixEquationsAD: MatrixEquationsAD
using StaticArrays: SMatrix

import MatrixEquationsAD: klein_map

function klein_map(
        A::SMatrix{n, n, T}, B::SMatrix{n, n, T}, ::Val{n_x}, ::Val{n_y};
        threshold = 1.0e-6,
    ) where {n, T, n_x, n_y}
    if n_x + n_y != n
        throw(DimensionMismatch("Val(n_x) + Val(n_y) must equal matrix size $n"))
    end
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

end
