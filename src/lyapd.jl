# Cache-aware shadow of `MatrixEquations.lyapd` and the in-place `lyapd!`
# variant, sharing one `schur(A)` across the primal and any AD directions.
# The Enzyme and ForwardDiff extensions add rules on top; the primal kernels
# live here so both extensions see the same `LyapDSchurCache` concrete type.

"""
    LyapDSchurCache(T, Z)

Cached real Schur factors of `A` for use by [`lyapdsolve`](@ref) /
[`lyapdadjointsolve`](@ref) / [`lyapd!`](@ref) — `T` is the (quasi-)upper
triangular factor and `Z` is the orthogonal change-of-basis, satisfying
`A = Z * T * Z'`.
"""
@concrete struct LyapDSchurCache
    T
    Z
end

function lyapdfactor(A::StridedMatrix{T}) where {T <: Union{Float32, Float64}}
    F = schur(A)
    return LyapDSchurCache(F.T, F.Z)
end

# Triangular-form Lyapunov solve in the Schur basis, then untransform.
# Dense-`C` path routes onto `sylvds!`; Symmetric-`C` path onto `lyapds!`.

function lyapdsolve(cache::LyapDSchurCache, C::StridedMatrix{T}) where {T}
    rhs = cache.Z' * C * cache.Z
    sylvds!(-cache.T, cache.T, rhs; adjB = true)
    rhs = cache.Z * rhs * cache.Z'
    return rhs
end

function lyapdsolve(
        cache::LyapDSchurCache, C::Symmetric{T, <:StridedMatrix{T}},
    ) where {T}
    rhs = utqu(C, cache.Z)
    lyapds!(cache.T, rhs)
    utqu!(rhs, cache.Z')
    return rhs
end

function lyapdadjointsolve(cache::LyapDSchurCache, C::StridedMatrix{T}) where {T}
    rhs = cache.Z' * C * cache.Z
    sylvds!(-cache.T, cache.T, rhs; adjA = true)
    rhs = cache.Z * rhs * cache.Z'
    return rhs
end

function lyapdadjointsolve(
        cache::LyapDSchurCache, C::Symmetric{T, <:StridedMatrix{T}},
    ) where {T}
    rhs = utqu(C, cache.Z)
    lyapds!(cache.T, rhs; adj = true)
    utqu!(rhs, cache.Z')
    return rhs
end

# Cache-aware shadow of `MatrixEquations.lyapd(A, C)` for Float32/64 strided
# inputs. Identical result to upstream; we override to wire the same cache
# pattern the AD rules rely on.

function MatrixEquations.lyapd(
        A::StridedMatrix{T}, C::StridedMatrix{T},
    ) where {T <: Union{Float32, Float64}}
    return lyapdsolve(lyapdfactor(A), C)
end

function MatrixEquations.lyapd(
        A::StridedMatrix{T}, C::Symmetric{T, <:StridedMatrix{T}},
    ) where {T <: Union{Float32, Float64}}
    return lyapdsolve(lyapdfactor(A), C)
end

# `lyapd!` — write the solution into a caller-supplied `X`. Returns `nothing`.
#
#   • Cache-taking overload: caller owns the `schur(A)` factorisation. Used
#     by the AD rules so one `schur(A)` is reused across every tangent /
#     cotangent direction.
#   • `A`-taking overloads: build a one-shot cache and forward to the
#     cache-taking version. Standard user entry point.
#
# Both `StridedMatrix` and `Symmetric{T,<:StridedMatrix{T}}` C are supported;
# the Symmetric path routes onto upstream `lyapds!`, the dense path onto
# `sylvds!`. Element type is restricted to `Float32`/`Float64` because the
# underlying upstream kernels are LAPACK-backed.

function lyapd!(
        X::StridedMatrix{T}, cache::LyapDSchurCache, C::StridedMatrix{T},
    ) where {T <: Union{Float32, Float64}}
    rhs = cache.Z' * C * cache.Z
    sylvds!(-cache.T, cache.T, rhs; adjB = true)
    mul!(X, cache.Z * rhs, cache.Z')
    return nothing
end

function lyapd!(
        X::StridedMatrix{T}, cache::LyapDSchurCache,
        C::Symmetric{T, <:StridedMatrix{T}},
    ) where {T <: Union{Float32, Float64}}
    rhs = utqu(C, cache.Z)
    lyapds!(cache.T, rhs)
    utqu!(rhs, cache.Z')
    copyto!(X, rhs)
    return nothing
end

function lyapd!(
        X::StridedMatrix{T}, A::StridedMatrix{T}, C::StridedMatrix{T},
    ) where {T <: Union{Float32, Float64}}
    n = checksquare(X)
    (checksquare(A) == n && checksquare(C) == n) ||
        throw(DimensionMismatch("lyapd!: X, A, C must all be $(n)×$(n)"))
    lyapd!(X, lyapdfactor(A), C)
    return nothing
end

function lyapd!(
        X::StridedMatrix{T}, A::StridedMatrix{T},
        C::Symmetric{T, <:StridedMatrix{T}},
    ) where {T <: Union{Float32, Float64}}
    n = checksquare(X)
    (checksquare(A) == n && checksquare(C) == n) ||
        throw(DimensionMismatch("lyapd!: X, A, C must all be $(n)×$(n)"))
    lyapd!(X, lyapdfactor(A), C)
    return nothing
end
