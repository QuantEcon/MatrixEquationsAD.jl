# ForwardDiff `Dual` dispatches and AD plumbing for `MatrixEquations.lyapd`
# and the in-place `MatrixEquationsAD.lyapd!`. Primal kernels and the cache
# type (`LyapDSchurCache`, `lyapdfactor`, `lyapdsolve`, the cache-aware
# `lyapd` shadow, and all `lyapd!` Float methods) live in `src/lyapd.jl`.

@inline _primal_argument(A::StridedMatrix) = map(value, A)
@inline _primal_argument(A::Symmetric)     = Symmetric(map(value, parent(A)), Symbol(A.uplo))

@inline _partial_argument(A::StridedMatrix, i) = map(x -> partials(x, i), A)
@inline _partial_argument(A::Symmetric, i)     = Symmetric(map(x -> partials(x, i), parent(A)), Symbol(A.uplo))

@inline _dense_copy(A::StridedMatrix) = copy(A)
@inline _dense_copy(A::Symmetric)     = Matrix(A)

@inline _symmetric_like(::StridedMatrix, A) = A
@inline _symmetric_like(C::Symmetric, A)    = Symmetric(A, Symbol(C.uplo))

# OOP `lyapd(::Dual…, ::Dual…)`: one `schur(A_val)`, `N` triangular tangent
# solves, then package values + partials back into `Dual` outputs.

function _lyapd_forwarddiff(A, C, ::Type{D}) where {T, V, N, D <: Dual{T, V, N}}
    Aval = _primal_argument(A)
    Cval = _primal_argument(C)
    cache = lyapdfactor(Aval)
    X = lyapdsolve(cache, Cval)

    dXs = ntuple(Val(N)) do i
        Base.@_inline_meta
        dC = _partial_argument(C, i)
        rhs = _dense_copy(dC)
        dA = _partial_argument(A, i)
        rhs .+= dA * X * Aval'
        rhs .+= Aval * X * dA'
        lyapdsolve(cache, _symmetric_like(Cval, rhs))
    end

    return map(CartesianIndices(X)) do idx
        Base.@_inline_meta
        Dual{T}(X[idx], Partials(ntuple(k -> dXs[k][idx], Val(N))))
    end
end

function MatrixEquations.lyapd(
        A::StridedMatrix{D}, C::StridedMatrix{D},
    ) where {T, V <: Union{Float32, Float64}, N, D <: Dual{T, V, N}}
    return _lyapd_forwarddiff(A, C, D)
end

function MatrixEquations.lyapd(
        A::StridedMatrix{D}, C::Symmetric{D, <:StridedMatrix{D}},
    ) where {T, V <: Union{Float32, Float64}, N, D <: Dual{T, V, N}}
    return _lyapd_forwarddiff(A, C, D)
end

# In-place `lyapd!(::Dual…, ::Dual…, ::Dual…)`: extract values, build the
# cache once, write the primal into a heap `Matrix{V}` workspace, then run
# `N` tangent solves and pack `Dual`s back into the caller's `X` buffer.

function _lyapd_inplace_forwarddiff!(X, A, C, ::Type{D}) where {T, V, N, D <: Dual{T, V, N}}
    Aval = _primal_argument(A)
    Cval = _primal_argument(C)
    cache = lyapdfactor(Aval)
    X_val = Matrix{V}(undef, size(X))
    lyapd!(X_val, cache, Cval)

    dXs = ntuple(Val(N)) do i
        Base.@_inline_meta
        dC = _partial_argument(C, i)
        rhs = _dense_copy(dC)
        dA = _partial_argument(A, i)
        rhs .+= dA * X_val * Aval'
        rhs .+= Aval * X_val * dA'
        lyapdsolve(cache, _symmetric_like(Cval, rhs))
    end

    @inbounds for idx in CartesianIndices(X)
        X[idx] = Dual{T}(X_val[idx], Partials(ntuple(k -> dXs[k][idx], Val(N))))
    end
    return nothing
end

function lyapd!(
        X::StridedMatrix{D}, A::StridedMatrix{D}, C::StridedMatrix{D},
    ) where {T, V <: Union{Float32, Float64}, N, D <: Dual{T, V, N}}
    return _lyapd_inplace_forwarddiff!(X, A, C, D)
end

function lyapd!(
        X::StridedMatrix{D}, A::StridedMatrix{D},
        C::Symmetric{D, <:StridedMatrix{D}},
    ) where {T, V <: Union{Float32, Float64}, N, D <: Dual{T, V, N}}
    return _lyapd_inplace_forwarddiff!(X, A, C, D)
end
