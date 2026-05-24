@inline function _shadow_or_zero(x, i, N)
    if typeof(x) <: Const
        return zero(x.val)
    elseif N == 1
        return x.dval
    else
        return x.dval[i]
    end
end

@inline function _copy_shadow_or_zero(x, i, N)
    if typeof(x) <: Const
        return zero(x.val)
    elseif N == 1
        return copy(x.dval)
    else
        return copy(x.dval[i])
    end
end

@inline function _ared_output(X, evals, F, Z, scalinfo)
    return (X, evals, F, Z, scalinfo)
end

@inline function _ared_zero_output(X, evals, F, Z, scalinfo)
    return (
        zero(X),
        zero(evals),
        zero(F),
        zero(Z),
        map(zero, scalinfo),
    )
end

@inline function _ared_shadow_output(dX, dF, X, evals, F, Z, scalinfo)
    out = _ared_zero_output(X, evals, F, Z, scalinfo)
    return (dX, out[2], dF, out[4], out[5])
end

function _ared_primal(
        A::StridedMatrix{T}, B::StridedMatrix{T}, R::StridedMatrix{T},
        Q::StridedMatrix{T}, S::StridedMatrix{T}; scaling = 'B',
        pow2 = false, as = false,
        rtol::Real = size(A, 1) * eps(real(float(one(T)))), nrm = 1
    ) where {T <: Union{Float32, Float64}}
    X, evals, F, Z, scalinfo = ared(A, B, R, Q, S; scaling, pow2, as, rtol, nrm)
    Acl = A - B * F
    cache = lyapdfactor(Matrix(Acl'))
    return X, evals, F, Z, scalinfo, Acl, cache
end

function _ared_tangent(
        A::StridedMatrix{T}, B::StridedMatrix{T}, R::StridedMatrix{T},
        Q::StridedMatrix{T}, S::StridedMatrix{T}, X::StridedMatrix{T},
        F::StridedMatrix{T}, Acl::StridedMatrix{T}, cache::LyapDSchurCache,
        dA::StridedMatrix{T}, dB::StridedMatrix{T}, dR::StridedMatrix{T},
        dQ::StridedMatrix{T}, dS::StridedMatrix{T}
    ) where {T}
    G = R + B' * X * B
    rhs = copy(dQ)
    rhs .+= dA' * X * Acl
    rhs .+= Acl' * X * dA
    rhs .-= Acl' * X * dB * F
    rhs .-= F' * dB' * X * Acl
    rhs .+= F' * dR * F
    rhs .-= dS * F
    rhs .-= F' * dS'
    symmetrize!!(rhs)
    dX = lyapdsolve(cache, rhs)

    dM = dB' * X * A
    dM .+= B' * dX * A
    dM .+= B' * X * dA
    dM .+= dS'
    dG = copy(dR)
    dG .+= dB' * X * B
    dG .+= B' * dX * B
    dG .+= B' * X * dB
    dF = G \ (dM - dG * F)

    return dX, dF
end

function _ared_adjoint!(
        dA::StridedMatrix{T}, dB::StridedMatrix{T}, dR::StridedMatrix{T},
        dQ::StridedMatrix{T}, dS::StridedMatrix{T}, A::StridedMatrix{T},
        B::StridedMatrix{T}, R::StridedMatrix{T}, X::StridedMatrix{T},
        F::StridedMatrix{T}, Acl::StridedMatrix{T}, cache::LyapDSchurCache,
        Xbar::StridedMatrix{T}, Fbar::StridedMatrix{T}
    ) where {T}
    G = R + B' * X * B
    Λ = G' \ Fbar
    Θ = -Λ * F'
    symmetrize!!(Θ)

    dA .+= X' * B * Λ
    dB .+= X * A * Λ'
    dB .+= X * B * Θ'
    dB .+= X' * B * Θ
    dR .+= Θ
    dS .+= Λ'

    Xbar_total = copy(Xbar)
    Xbar_total .+= B * Λ * A'
    Xbar_total .+= B * Θ * B'
    symmetrize!!(Xbar_total)

    Y = lyapdadjointsolve(cache, Xbar_total)

    dQ .+= Y
    dA .+= X * Acl * Y'
    dA .+= X' * Acl * Y
    dB .-= X' * Acl * Y * F'
    dB .-= X * Acl * Y' * F'
    dR .+= F * Y * F'
    dS .-= Y * F'
    dS .-= Y' * F'

    return nothing
end
