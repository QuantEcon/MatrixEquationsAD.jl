# In-place per-block solve. Mv, rhsv are views into a hoisted 8x8 / length-8
# buffer; we factor in-place with `lu!` and solve in-place with `ldiv!` so the
# happy path stays alloc-free. When M is singular (e.g. coincident generalized
# eigenvalues, common in DSGE pencils with many static equations), we
# Tikhonov-regularize by adding `δI` to M and re-solving. The tangent at a
# coincident-eigenvalue point isn't unique anyway, so any small regularization
# that keeps the factorization stable is acceptable.
function _ordqz_block_solve!(Mv, rhsv, M_save)
    copyto!(M_save, Mv)
    Tel = real(eltype(Mv))
    m_scale = zero(Tel)
    @inbounds for j in axes(M_save, 2), i in axes(M_save, 1)
        m_scale = max(m_scale, abs(M_save[i, j]))
    end
    F = lu!(Mv; check = false)
    # Treat as singular if `lu!` failed, or if the smallest pivot is tiny
    # relative to the matrix's largest entry (ill-conditioned but not exactly
    # singular -- common in DSGE pencils with near-coincident eigenvalues).
    min_pivot = LinearAlgebra.issuccess(F) ? abs(Mv[1, 1]) : zero(Tel)
    if LinearAlgebra.issuccess(F)
        @inbounds for i in 2:size(Mv, 1)
            min_pivot = min(min_pivot, abs(Mv[i, i]))
        end
    end
    if min_pivot > sqrt(eps(Tel)) * m_scale
        ldiv!(F, rhsv)
        return rhsv
    end
    delta = sqrt(eps(Tel)) * (m_scale + one(Tel))
    @inbounds for i in axes(M_save, 1)
        M_save[i, i] += delta
    end
    F = lu!(M_save; check = false)
    ldiv!(F, rhsv)
    return rhsv
end

function qzblocks(S)
    n = size(S, 1)
    nb = 0
    i = 1
    while i <= n
        nb += 1
        i += i < n && abs(S[i + 1, i]) > 1.0e-14 ? 2 : 1
    end

    starts = Vector{Int}(undef, nb)
    sizes = Vector{Int}(undef, nb)
    i = 1
    b = 0
    while i <= n
        b += 1
        starts[b] = i
        if i < n && abs(S[i + 1, i]) > 1.0e-14
            sizes[b] = 2
            i += 2
        else
            sizes[b] = 1
            i += 1
        end
    end
    return starts, sizes
end

function ordqz_tangent!(dS, dT, dQ, dZ, S, T, Q, Z, dA, dB)
    n = size(S, 1)
    starts, sizes = qzblocks(S)
    nb = length(starts)

    Tel = eltype(S)
    E = zeros(Tel, n, n)
    F = zeros(Tel, n, n)
    OmegaQ = zeros(Tel, n, n)
    OmegaZ = zeros(Tel, n, n)
    tmp1 = zeros(Tel, n, n)
    tmp2 = zeros(Tel, n, n)

    # Per-block-pair scratch: blocks are 1 or 2, so n_unknowns ≤ 8.
    M_buf = Matrix{Tel}(undef, 8, 8)
    M_save = Matrix{Tel}(undef, 8, 8)
    rhs_buf = Vector{Tel}(undef, 8)

    mul!(tmp1, transpose(Q), dA)
    mul!(E, tmp1, Z)
    mul!(tmp1, transpose(Q), dB)
    mul!(F, tmp1, Z)

    @inbounds for jb in 1:(nb - 1)
        j_start = starts[jb]
        qj = sizes[jb]
        j_range = j_start:(j_start + qj - 1)

        for ib in nb:-1:(jb + 1)
            i_start = starts[ib]
            pi = sizes[ib]
            if pi == 1 && qj == 1
                # Fast inline 2x2 path: the common DSGE case (1x1 blocks on
                # both sides). M = [-S_jj S_ii; -T_jj T_ii] and rhs has two
                # entries. Direct Cramer-rule solve, with a closed-form
                # minimum-norm pseudoinverse for the rank-deficient case.
                ii = i_start
                jj = j_start
                S_ii = S[ii, ii]; T_ii = T[ii, ii]
                S_jj = S[jj, jj]; T_jj = T[jj, jj]
                rhs_S = -E[ii, jj]
                rhs_T = -F[ii, jj]
                for k in (i_start + 1):n
                    rhs_S -= S[ii, k] * OmegaZ[k, jj]
                    rhs_T -= T[ii, k] * OmegaZ[k, jj]
                end
                for k in 1:(j_start - 1)
                    rhs_S += OmegaQ[ii, k] * S[k, jj]
                    rhs_T += OmegaQ[ii, k] * T[k, jj]
                end
                # Original M = [-S_jj S_ii; -T_jj T_ii]; det(M) = S_ii*T_jj -
                # S_jj*T_ii vanishes when blocks share a generalized eigenvalue
                # (including the common DSGE case S_ii ≈ S_jj ≈ 0). Tikhonov-
                # regularize by adding δ to the diagonal. The "is this matrix
                # well-conditioned?" test compares |det| against √eps · ‖M‖²_∞;
                # for our pencil this catches both exact singularity and the
                # ill-conditioned case where det is tiny but not zero.
                m_scale = max(abs(S_jj), abs(S_ii), abs(T_jj), abs(T_ii))
                a = -S_jj
                d = T_ii
                det = a * d - S_ii * (-T_jj)  # = S_ii*T_jj - S_jj*T_ii
                tol = sqrt(eps(Tel)) * m_scale * m_scale
                if abs(det) <= tol
                    delta = sqrt(eps(Tel)) * (m_scale + one(Tel))
                    a += delta
                    d += delta
                    det = a * d - S_ii * (-T_jj)
                end
                inv_det = inv(det)
                sol_1 = (d * rhs_S - S_ii * rhs_T) * inv_det
                sol_2 = (a * rhs_T + T_jj * rhs_S) * inv_det
                OmegaQ[ii, jj] = sol_1
                OmegaQ[jj, ii] = -sol_1
                OmegaZ[ii, jj] = sol_2
                OmegaZ[jj, ii] = -sol_2
                continue
            end

            i_range = i_start:(i_start + pi - 1)
            n_unknowns = 2 * pi * qj
            Mv = view(M_buf, 1:n_unknowns, 1:n_unknowns)
            rhsv = view(rhs_buf, 1:n_unknowns)
            Msv = view(M_save, 1:n_unknowns, 1:n_unknowns)
            fill!(Mv, zero(Tel))

            for (jj_loc, jj) in enumerate(j_range)
                for (ii_loc, ii) in enumerate(i_range)
                    eq_S = (jj_loc - 1) * pi + ii_loc
                    eq_T = pi * qj + eq_S
                    rhs_S = -E[ii, jj]
                    rhs_T = -F[ii, jj]

                    for k in (i_start + pi):n
                        rhs_S -= S[ii, k] * OmegaZ[k, jj]
                        rhs_T -= T[ii, k] * OmegaZ[k, jj]
                    end
                    for k in 1:(j_start - 1)
                        rhs_S += OmegaQ[ii, k] * S[k, jj]
                        rhs_T += OmegaQ[ii, k] * T[k, jj]
                    end

                    rhsv[eq_S] = rhs_S
                    rhsv[eq_T] = rhs_T

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

            _ordqz_block_solve!(Mv, rhsv, Msv)
            for (jj_loc, jj) in enumerate(j_range)
                for (ii_loc, ii) in enumerate(i_range)
                    idx = (jj_loc - 1) * pi + ii_loc
                    OmegaQ[ii, jj] = rhsv[idx]
                    OmegaQ[jj, ii] = -rhsv[idx]
                    OmegaZ[ii, jj] = rhsv[pi * qj + idx]
                    OmegaZ[jj, ii] = -rhsv[pi * qj + idx]
                end
            end
        end
    end

    mul!(tmp1, OmegaQ, S)
    mul!(tmp2, S, OmegaZ)
    @inbounds for j in 1:n, i in 1:n
        dS[i, j] = E[i, j] - tmp1[i, j] + tmp2[i, j]
    end
    for b in 1:nb
        b_start = starts[b]
        b_end = b_start + sizes[b] - 1
        for j in b_start:b_end, i in (b_end + 1):n
            dS[i, j] = zero(eltype(dS))
        end
    end

    mul!(tmp1, OmegaQ, T)
    mul!(tmp2, T, OmegaZ)
    @inbounds for j in 1:n, i in 1:n
        dT[i, j] = F[i, j] - tmp1[i, j] + tmp2[i, j]
    end
    @inbounds for j in 1:n, i in (j + 1):n
        dT[i, j] = zero(eltype(dT))
    end

    mul!(dQ, Q, OmegaQ)
    mul!(dZ, Z, OmegaZ)
    return nothing
end
