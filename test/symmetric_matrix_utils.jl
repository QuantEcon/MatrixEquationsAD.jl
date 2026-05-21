function vech_symmetric(A::AbstractMatrix)
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("matrix must be square"))
    n = size(A, 1)
    v = Vector{eltype(A)}(undef, n * (n + 1) ÷ 2)
    k = 1
    for j in 1:n
        for i in j:n
            v[k] = A[i, j]
            k += 1
        end
    end
    return v
end

function unvech_symmetric(v::AbstractVector, n::Integer)
    length(v) == n * (n + 1) ÷ 2 ||
        throw(DimensionMismatch("vech length does not match symmetric matrix size"))
    A = Matrix{eltype(v)}(undef, n, n)
    k = 1
    for j in 1:n
        for i in j:n
            A[i, j] = v[k]
            A[j, i] = v[k]
            k += 1
        end
    end
    return A
end
