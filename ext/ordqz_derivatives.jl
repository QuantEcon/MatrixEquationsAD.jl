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

    E = zeros(eltype(S), n, n)
    F = zeros(eltype(S), n, n)
    OmegaQ = zeros(eltype(S), n, n)
    OmegaZ = zeros(eltype(S), n, n)
    tmp1 = zeros(eltype(S), n, n)
    tmp2 = zeros(eltype(S), n, n)

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
            i_range = i_start:(i_start + pi - 1)
            n_unknowns = 2 * pi * qj
            rhs = zeros(eltype(S), n_unknowns)
            M = zeros(eltype(S), n_unknowns, n_unknowns)

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

                    rhs[eq_S] = rhs_S
                    rhs[eq_T] = rhs_T

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

            sol = M \ rhs
            for (jj_loc, jj) in enumerate(j_range)
                for (ii_loc, ii) in enumerate(i_range)
                    idx = (jj_loc - 1) * pi + ii_loc
                    OmegaQ[ii, jj] = sol[idx]
                    OmegaQ[jj, ii] = -sol[idx]
                    OmegaZ[ii, jj] = sol[pi * qj + idx]
                    OmegaZ[jj, ii] = -sol[pi * qj + idx]
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
