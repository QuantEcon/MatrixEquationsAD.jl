module MatrixEquationsADStaticArraysExt

using LinearAlgebra: I, lu
using MatrixEquationsAD: MatrixEquationsAD
using StaticArrays: SMatrix

import MatrixEquationsAD: build_M!!, klein_map, lyapdkr, symmetrize!!

@inline function build_M!!(_, A::SMatrix{N, N, T}) where {N, T}
    return SMatrix{N * N, N * N, T}(I) - kron(A, A)
end

@inline symmetrize!!(X::SMatrix) = (X + X') / 2

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

function lyapdkr(
        A::SMatrix{N, N, T}, C::SMatrix{N, N, T};
        M_ws = nothing,  # accepted for API parity with the heap path; ignored
    ) where {N, T}
    # StaticArrays caps native LU at total elements ≤ 14×14 = 196. For the
    # `n²×n²` pencil that means truly heap-free only at N ≤ 3. Past N ≥ 4,
    # SA's LU fallback wraps a heap LU back into static form which costs
    # more than just running the heap path once — so dispatch by size.
    # N is a type parameter so the compiler folds this branch.
    if N * N <= 14
        M = build_M!!(nothing, A)
        F = lu(M)
        return symmetrize!!(SMatrix{N, N, T}(F \ vec(C)))
    else
        return SMatrix{N, N, T}(MatrixEquationsAD.lyapdkr(Matrix(A), Matrix(C)))
    end
end

end
