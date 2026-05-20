function ordqz_adjoint!(dA, dB, S, T, Q, Z, dS, dT, dQ, dZ)
    n = size(S, 1)
    starts, sizes = qzblocks(S)
    nb = length(starts)

    Tel = eltype(S)
    OmegaQ = zeros(Tel, n, n)
    OmegaZ = zeros(Tel, n, n)
    tmp1 = zeros(Tel, n, n)
    tmp2 = zeros(Tel, n, n)
    bar_E = zeros(Tel, n, n)
    bar_F = zeros(Tel, n, n)

    # Per-block-pair scratch hoisted out of the inner loop (blocks ≤ 2 each).
    M_buf = Matrix{Tel}(undef, 8, 8)
    Mt_buf = Matrix{Tel}(undef, 8, 8)
    bar_x_buf = Vector{Tel}(undef, 8)

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
            Mv = view(M_buf, 1:n_unknowns, 1:n_unknowns)
            Mtv = view(Mt_buf, 1:n_unknowns, 1:n_unknowns)
            bar_x = view(bar_x_buf, 1:n_unknowns)
            fill!(Mv, zero(Tel))

            for (jj_loc, jj) in enumerate(j_range)
                for (ii_loc, ii) in enumerate(i_range)
                    eq_S = (jj_loc - 1) * pi + ii_loc
                    eq_T = pi * qj + eq_S
                    for (kk_loc, kk) in enumerate(j_range)
                        col = (kk_loc - 1) * pi + ii_loc
                        Mv[eq_S, col] -= S[kk, jj]
                        Mv[eq_T, col] -= T[kk, jj]
                    end
                    for (kk_loc, kk) in enumerate(i_range)
                        col = pi * qj + (jj_loc - 1) * pi + kk_loc
                        Mv[eq_S, col] += S[ii, kk]
                        Mv[eq_T, col] += T[ii, kk]
                    end
                end
            end

            for (jj_loc, jj) in enumerate(j_range)
                for (ii_loc, ii) in enumerate(i_range)
                    idx = (jj_loc - 1) * pi + ii_loc
                    bar_x[idx] = OmegaQ[ii, jj] - OmegaQ[jj, ii]
                    bar_x[pi * qj + idx] = OmegaZ[ii, jj] - OmegaZ[jj, ii]
                end
            end

            # Solve transpose(M) * bar_rhs = bar_x.
            transpose!(Mtv, Mv)
            _ordqz_block_solve!(Mtv, bar_x)
            bar_rhs = bar_x
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

function ordered_qz_rule_primal!(
        func::Const{typeof(_ordqz!)},
        S, Targ, Q, Z, A, B, ordering, threshold
    )
    return func.val(S, Targ, Q, Z, A, B, ordering, threshold)
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{F},
        ::Type{<:Const},
        S::Annotation{<:StridedMatrix{T}},
        Targ::Annotation{<:StridedMatrix{T}},
        Q::Annotation{<:StridedMatrix{T}},
        Z::Annotation{<:StridedMatrix{T}},
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}},
        ordering::Const,
        threshold::Const
    ) where {F <: typeof(_ordqz!), T <: Union{Float32, Float64}}
    primal = ordered_qz_rule_primal!(
        func,
        S.val, Targ.val, Q.val, Z.val, A.val, B.val, ordering.val, threshold.val
    )

    # NOTE: the tangent loop must run whenever any of the input/output args has
    # a Duplicated/BatchDuplicated annotation -- _not_ gated on
    # EnzymeRules.needs_shadow(config), which queries the return-value shadow.
    # _ordqz! returns an Int (sdim), so the return is always Const and
    # needs_shadow is false; that previously caused the rule to silently skip
    # tangent propagation through the mutated S/T/Q/Z buffers.
    any_dup = !(
        typeof(S) <: Const && typeof(Targ) <: Const &&
            typeof(Q) <: Const && typeof(Z) <: Const &&
            typeof(A) <: Const && typeof(B) <: Const
    )
    if any_dup
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
        return primal
    else
        return nothing
    end
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfig,
        func::Const{F},
        ::Type{<:Const},
        S::Annotation{<:StridedMatrix{T}},
        Targ::Annotation{<:StridedMatrix{T}},
        Q::Annotation{<:StridedMatrix{T}},
        Z::Annotation{<:StridedMatrix{T}},
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}},
        ordering::Const,
        threshold::Const
    ) where {F <: typeof(_ordqz!), T <: Union{Float32, Float64}}
    primal = ordered_qz_rule_primal!(
        func,
        S.val, Targ.val, Q.val, Z.val, A.val, B.val, ordering.val, threshold.val
    )
    tape = (copy(S.val), copy(Targ.val), copy(Q.val), copy(Z.val))
    returned = EnzymeRules.needs_primal(config) ? primal : nothing
    return EnzymeRules.AugmentedReturn(returned, nothing, tape)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfig,
        func::Const{F},
        ::Type{<:Const},
        tape,
        S::Annotation{<:StridedMatrix{T}},
        Targ::Annotation{<:StridedMatrix{T}},
        Q::Annotation{<:StridedMatrix{T}},
        Z::Annotation{<:StridedMatrix{T}},
        A::Annotation{<:StridedMatrix{T}},
        B::Annotation{<:StridedMatrix{T}},
        ordering::Const,
        threshold::Const
    ) where {F <: typeof(_ordqz!), T <: Union{Float32, Float64}}
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

    return (nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
end
