function ordqz_adjoint!(dA, dB, S, T, Q, Z, dS, dT, dQ, dZ)
    n = size(S, 1)
    starts, sizes = qzblocks(S)
    nb = length(starts)

    OmegaQ = zeros(eltype(S), n, n)
    OmegaZ = zeros(eltype(S), n, n)
    tmp1 = zeros(eltype(S), n, n)
    tmp2 = zeros(eltype(S), n, n)
    bar_E = zeros(eltype(S), n, n)
    bar_F = zeros(eltype(S), n, n)

    @inbounds for j in 1:n, i in 1:n
        tmp1[i, j] = dS[i, j]
    end
    for b in 1:nb
        b_start = starts[b]
        b_end = b_start + sizes[b] - 1
        for j in b_start:b_end, i in (b_end + 1):n
            tmp1[i, j] = zero(eltype(tmp1))
        end
    end

    mul!(OmegaQ, transpose(Q), dQ)
    mul!(tmp2, tmp1, transpose(S))
    OmegaQ .-= tmp2
    copyto!(bar_E, tmp1)

    mul!(OmegaZ, transpose(Z), dZ)
    mul!(tmp2, transpose(S), tmp1)
    OmegaZ .+= tmp2

    @inbounds for j in 1:n, i in 1:n
        tmp1[i, j] = dT[i, j]
    end
    @inbounds for j in 1:n, i in (j + 1):n
        tmp1[i, j] = zero(eltype(tmp1))
    end

    mul!(tmp2, tmp1, transpose(T))
    OmegaQ .-= tmp2
    copyto!(bar_F, tmp1)

    mul!(tmp2, transpose(T), tmp1)
    OmegaZ .+= tmp2

    @inbounds for jb in (nb - 1):-1:1
        j_start = starts[jb]
        qj = sizes[jb]
        j_range = j_start:(j_start + qj - 1)

        for ib in (jb + 1):nb
            i_start = starts[ib]
            pi = sizes[ib]
            i_range = i_start:(i_start + pi - 1)
            n_unknowns = 2 * pi * qj
            M = zeros(eltype(S), n_unknowns, n_unknowns)

            for (jj_loc, jj) in enumerate(j_range)
                for (ii_loc, ii) in enumerate(i_range)
                    eq_S = (jj_loc - 1) * pi + ii_loc
                    eq_T = pi * qj + eq_S
                    for (kk_loc, kk) in enumerate(j_range)
                        col = (kk_loc - 1) * pi + ii_loc
                        M[eq_S, col] -= S[kk, jj]
                        M[eq_T, col] -= T[kk, jj]
                    end
                    for (kk_loc, kk) in enumerate(i_range)
                        col = pi * qj + (jj_loc - 1) * pi + kk_loc
                        M[eq_S, col] += S[ii, kk]
                        M[eq_T, col] += T[ii, kk]
                    end
                end
            end

            bar_x = zeros(eltype(S), n_unknowns)
            for (jj_loc, jj) in enumerate(j_range)
                for (ii_loc, ii) in enumerate(i_range)
                    idx = (jj_loc - 1) * pi + ii_loc
                    bar_x[idx] = OmegaQ[ii, jj] - OmegaQ[jj, ii]
                    bar_x[pi * qj + idx] = OmegaZ[ii, jj] - OmegaZ[jj, ii]
                end
            end

            bar_rhs = transpose(M) \ bar_x
            for (jj_loc, jj) in enumerate(j_range)
                for (ii_loc, ii) in enumerate(i_range)
                    eq_S = (jj_loc - 1) * pi + ii_loc
                    eq_T = pi * qj + eq_S
                    bar_E[ii, jj] -= bar_rhs[eq_S]
                    bar_F[ii, jj] -= bar_rhs[eq_T]
                    for k in (i_start + pi):n
                        OmegaZ[k, jj] -= bar_rhs[eq_S] * S[ii, k]
                        OmegaZ[k, jj] -= bar_rhs[eq_T] * T[ii, k]
                    end
                    for k in 1:(j_start - 1)
                        OmegaQ[ii, k] += bar_rhs[eq_S] * S[k, jj]
                        OmegaQ[ii, k] += bar_rhs[eq_T] * T[k, jj]
                    end
                end
            end

            for jj in j_range, ii in i_range
                OmegaQ[ii, jj] = zero(eltype(OmegaQ))
                OmegaQ[jj, ii] = zero(eltype(OmegaQ))
                OmegaZ[ii, jj] = zero(eltype(OmegaZ))
                OmegaZ[jj, ii] = zero(eltype(OmegaZ))
            end
        end
    end

    mul!(tmp1, Q, bar_E)
    mul!(tmp2, tmp1, transpose(Z))
    dA .+= tmp2

    mul!(tmp1, Q, bar_F)
    mul!(tmp2, tmp1, transpose(Z))
    dB .+= tmp2

    fill!(dS, zero(eltype(dS)))
    fill!(dT, zero(eltype(dT)))
    fill!(dQ, zero(eltype(dQ)))
    fill!(dZ, zero(eltype(dZ)))
    return nothing
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(ordqz!)},
        ::Type{<:Const},
        S::Annotation{<:StridedMatrix{T}},
        Targ::Annotation{<:StridedMatrix{T}},
        Q::Annotation{<:StridedMatrix{T}},
        Z::Annotation{<:StridedMatrix{T}},
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}},
        select::Const
    ) where {T <: Union{Float32, Float64}}
    sdim = func.val(S.val, Targ.val, Q.val, Z.val, A.val, B.val, select.val)

    if EnzymeRules.needs_shadow(config)
        N = EnzymeRules.width(config)
        for i in 1:N
            dA = if typeof(A) <: Const
                zero(A.val)
            elseif N == 1
                A.dval
            else
                A.dval[i]
            end
            dB = if typeof(B) <: Const
                zero(B.val)
            elseif N == 1
                B.dval
            else
                B.dval[i]
            end
            dS = if typeof(S) <: Const
                zero(S.val)
            elseif N == 1
                S.dval
            else
                S.dval[i]
            end
            dT = if typeof(Targ) <: Const
                zero(Targ.val)
            elseif N == 1
                Targ.dval
            else
                Targ.dval[i]
            end
            dQ = if typeof(Q) <: Const
                zero(Q.val)
            elseif N == 1
                Q.dval
            else
                Q.dval[i]
            end
            dZ = if typeof(Z) <: Const
                zero(Z.val)
            elseif N == 1
                Z.dval
            else
                Z.dval[i]
            end
            ordqz_tangent!(
                dS, dT, dQ, dZ,
                S.val, Targ.val, Q.val, Z.val,
                dA, dB,
            )
        end
    end

    if EnzymeRules.needs_primal(config)
        return sdim
    else
        return nothing
    end
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(ordqz!)},
        ::Type{<:Const},
        S::Annotation{<:StridedMatrix{T}},
        Targ::Annotation{<:StridedMatrix{T}},
        Q::Annotation{<:StridedMatrix{T}},
        Z::Annotation{<:StridedMatrix{T}},
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}},
        select::Const
    ) where {T <: Union{Float32, Float64}}
    sdim = func.val(S.val, Targ.val, Q.val, Z.val, A.val, B.val, select.val)
    tape = (copy(S.val), copy(Targ.val), copy(Q.val), copy(Z.val))
    primal = EnzymeRules.needs_primal(config) ? sdim : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, tape)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{typeof(ordqz!)},
        ::Type{<:Const},
        tape,
        S::Annotation{<:StridedMatrix{T}},
        Targ::Annotation{<:StridedMatrix{T}},
        Q::Annotation{<:StridedMatrix{T}},
        Z::Annotation{<:StridedMatrix{T}},
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}},
        select::Const
    ) where {T <: Union{Float32, Float64}}
    Sv, Tv, Qv, Zv = tape
    N = EnzymeRules.width(config)

    for i in 1:N
        dA = if typeof(A) <: Const
            zero(A.val)
        elseif N == 1
            A.dval
        else
            A.dval[i]
        end
        dB = if typeof(B) <: Const
            zero(B.val)
        elseif N == 1
            B.dval
        else
            B.dval[i]
        end
        dS = if typeof(S) <: Const
            zero(S.val)
        elseif N == 1
            S.dval
        else
            S.dval[i]
        end
        dT = if typeof(Targ) <: Const
            zero(Targ.val)
        elseif N == 1
            Targ.dval
        else
            Targ.dval[i]
        end
        dQ = if typeof(Q) <: Const
            zero(Q.val)
        elseif N == 1
            Q.dval
        else
            Q.dval[i]
        end
        dZ = if typeof(Z) <: Const
            zero(Z.val)
        elseif N == 1
            Z.dval
        else
            Z.dval[i]
        end
        ordqz_adjoint!(
            dA, dB,
            Sv, Tv, Qv, Zv,
            dS, dT, dQ, dZ,
        )
    end

    return (nothing, nothing, nothing, nothing, nothing, nothing, nothing)
end
