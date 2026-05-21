function _klein_bigk_plan(A, B, g_x, h_x)
    T = promote_type(eltype(A), eltype(B), eltype(g_x), eltype(h_x))
    n = size(A, 1)
    n_x = size(h_x, 1)
    n_y = n - n_x

    A_mat = Matrix{T}(A)
    B_mat = Matrix{T}(B)
    g_mat = Matrix{T}(g_x)
    h_mat = Matrix{T}(h_x)

    I_x = Matrix{T}(LinearAlgebra.I, n_x, n_x)
    I_y = Matrix{T}(LinearAlgebra.I, n_y, n_y)
    G = vcat(I_x, g_mat)
    E_y = vcat(zeros(T, n_x, n_y), I_y)

    AG = A_mat * G
    AE_y = A_mat * E_y
    BE_y = B_mat * E_y

    n_h = n_x * n_x
    K = zeros(T, n * n_x, n_h + n_y * n_x)
    @inbounds for j in 1:n_x
        row0 = (j - 1) * n
        hcol0 = (j - 1) * n_x
        for k in 1:n_x
            for i in 1:n
                K[row0 + i, hcol0 + k] = AG[i, k]
            end
        end

        for q in 1:n_x
            gcol0 = n_h + (q - 1) * n_y
            hqj = h_mat[q, j]
            for p in 1:n_y
                for i in 1:n
                    K[row0 + i, gcol0 + p] += hqj * AE_y[i, p]
                end
            end
        end

        gcol0 = n_h + (j - 1) * n_y
        for p in 1:n_y
            for i in 1:n
                K[row0 + i, gcol0 + p] += BE_y[i, p]
            end
        end
    end

    return (; F = LinearAlgebra.lu(K), G, h_x = h_mat, n, n_x, n_y)
end

function _klein_bigk_jvp(plan, dA, dB)
    (; F, G, h_x, n_x, n_y) = plan
    T = eltype(G)
    dA_mat = Matrix{T}(dA)
    dB_mat = Matrix{T}(dB)

    R = -((dA_mat * G) * h_x + dB_mat * G)
    x = F \ vec(R)

    n_h = n_x * n_x
    dh = copy(reshape(view(x, 1:n_h), n_x, n_x))
    dg = copy(reshape(view(x, (n_h + 1):length(x)), n_y, n_x))
    return (; g_x = dg, h_x = dh)
end

function _klein_bigk_vjp(plan, g_bar, h_bar)
    (; F, G, h_x, n, n_x) = plan
    T = eltype(G)
    rhs = vcat(vec(Matrix{T}(h_bar)), vec(Matrix{T}(g_bar)))
    lambda = F' \ rhs
    Lambda = reshape(lambda, n, n_x)

    A_bar = -(Lambda * transpose(h_x)) * transpose(G)
    B_bar = -Lambda * transpose(G)
    return (; A = A_bar, B = B_bar)
end

function _klein_structured_plan(A, B, g_x, h_x)
    T = promote_type(eltype(A), eltype(B), eltype(g_x), eltype(h_x))
    n = size(A, 1)
    n_x = size(h_x, 1)
    n_y = n - n_x

    A_mat = Matrix{T}(A)
    B_mat = Matrix{T}(B)
    g_mat = Matrix{T}(g_x)
    h_mat = Matrix{T}(h_x)

    I_x = Matrix{T}(LinearAlgebra.I, n_x, n_x)
    I_y = Matrix{T}(LinearAlgebra.I, n_y, n_y)
    G = vcat(I_x, g_mat)
    E_y = vcat(zeros(T, n_x, n_y), I_y)

    AG = A_mat * G
    AE_y = A_mat * E_y
    BE_y = B_mat * E_y
    C0 = hcat(AG, BE_y)
    F0 = LinearAlgebra.lu(C0)

    J = F0 \ AE_y
    Jx = copy(view(J, 1:n_x, :))
    Jy = copy(view(J, (n_x + 1):n, :))

    Fy = LinearAlgebra.schur(Jy)
    Fh = LinearAlgebra.schur(h_mat)

    return (;
        G, h_x = h_mat, F0, Jx,
        Sy = Fy.T, Qy = Fy.Z, Sh = Fh.T, Qh = Fh.Z,
        n, n_x, n_y,
    )
end

function _klein_structured_jvp(plan, dA, dB)
    (; G, h_x, F0, Jx, Sy, Qy, Sh, Qh, n, n_x) = plan
    T = eltype(G)
    dA_mat = Matrix{T}(dA)
    dB_mat = Matrix{T}(dB)

    R = -((dA_mat * G) * h_x + dB_mat * G)
    Y = F0 \ R

    Yx = view(Y, 1:n_x, :)
    Yy = view(Y, (n_x + 1):n, :)

    Xtilde = transpose(Qy) * Yy * Qh
    sylvds!(Sy, Sh, Xtilde)
    dg = Qy * Xtilde * transpose(Qh)
    dh = copy(Yx)
    dh .-= (Jx * dg) * h_x

    return (; g_x = dg, h_x = dh)
end

function _klein_structured_vjp(plan, g_bar, h_bar)
    (; G, h_x, F0, Jx, Sy, Qy, Sh, Qh, n, n_x) = plan
    T = eltype(G)
    g_bar_mat = Matrix{T}(g_bar)
    h_bar_mat = Matrix{T}(h_bar)

    X_bar = g_bar_mat - (transpose(Jx) * h_bar_mat) * transpose(h_x)
    Ztilde = transpose(Qy) * X_bar * Qh
    sylvds!(Sy, Sh, Ztilde; adjA = true, adjB = true)
    Z = Qy * Ztilde * transpose(Qh)

    Y_bar = Matrix{T}(undef, n, n_x)
    copyto!(view(Y_bar, 1:n_x, :), h_bar_mat)
    copyto!(view(Y_bar, (n_x + 1):n, :), Z)

    Lambda = F0' \ Y_bar
    A_bar = -(Lambda * transpose(h_x)) * transpose(G)
    B_bar = -Lambda * transpose(G)
    return (; A = A_bar, B = B_bar)
end
