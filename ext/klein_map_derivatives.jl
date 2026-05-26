# Big-K factorisation (OOP `klein_map`) — see docs § Big-K JVP / VJP for
# the K block layout and vec ordering. The triple loop assembles K with
# the h_x-block first (cols 1:n_h), then the g_x-block (cols n_h+1:end).

function _klein_bigk_plan(A, B, g_x, h_x)
    T = promote_type(eltype(A), eltype(B), eltype(g_x), eltype(h_x))
    n = size(A, 1)
    n_x = size(h_x, 1)
    n_y = n - n_x

    A_mat = Matrix{T}(A)
    B_mat = Matrix{T}(B)
    g_mat = Matrix{T}(g_x)
    h_mat = Matrix{T}(h_x)

    # Ψ = G = [I_{n_x}; g_x],   E_y = [0; I_{n_y}].
    I_x = Matrix{T}(LinearAlgebra.I, n_x, n_x)
    I_y = Matrix{T}(LinearAlgebra.I, n_y, n_y)
    G = vcat(I_x, g_mat)
    E_y = vcat(zeros(T, n_x, n_y), I_y)

    # Three shared products used to fill K — and reused later in the JVP
    # RHS and VJP parameter outer products (via the cached `G`, `h_x`).
    AG = A_mat * G            # M  = A · Ψ            (n × n_x)
    AE_y = A_mat * E_y        # N  = A · E_y          (n × n_y)
    BE_y = B_mat * E_y        # P  = B · E_y          (n × n_y)

    # Assemble K with h_x-block first, then g_x-block. Each j ∈ 1:n_x
    # corresponds to one of the n_x "h_x columns" of the vec ordering.
    n_h = n_x * n_x
    K = zeros(T, n * n_x, n_h + n_y * n_x)
    @inbounds for j in 1:n_x
        row0 = (j - 1) * n          # start row of K[:, j-th n-block]
        hcol0 = (j - 1) * n_x       # start col of I_{n_x} ⊗ M  block

        # K[row0+(1:n), hcol0+(1:n_x)] = M ; i.e. the I_{n_x} ⊗ M block.
        for k in 1:n_x
            for i in 1:n
                K[row0 + i, hcol0 + k] = AG[i, k]
            end
        end

        # K[row0+(1:n), n_h + (q-1)·n_y + (1:n_y)] += h_x[q, j] · N
        # — this is the `h_x' ⊗ N` term in the g_x-block of K.
        for q in 1:n_x
            gcol0 = n_h + (q - 1) * n_y
            hqj = h_mat[q, j]
            for p in 1:n_y
                for i in 1:n
                    K[row0 + i, gcol0 + p] += hqj * AE_y[i, p]
                end
            end
        end

        # K[row0+(1:n), n_h + (j-1)·n_y + (1:n_y)] += P
        # — this is the `I_{n_x} ⊗ P` term in the g_x-block of K.
        gcol0 = n_h + (j - 1) * n_y
        for p in 1:n_y
            for i in 1:n
                K[row0 + i, gcol0 + p] += BE_y[i, p]
            end
        end
    end

    return (; F = LinearAlgebra.lu(K), G, h_x = h_mat, n, n_x, n_y)
end

# Docs § Big-K JVP: v = K⁻¹ · (−vec(dA·Ψ·h_x + dB·Ψ)),
# then split v = [vec(d h_x); vec(d g_x)].

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

# Docs § Big-K VJP: Λ = reshape(K⁻ᵀ · [vec(h̄); vec(ḡ)], n, n_x);
# Ā += −Λ · h_x' · Ψ',   B̄ += −Λ · Ψ'.

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

# Docs § Reduced-Sylvester factorisation (in-place `klein_map!`).
#
# Avoids the dense (n n_x) × (n n_x) K factorisation entirely. Build
# instead the n × n block
#   C₀ = [M | P] = [A·Ψ | B·E_y].
# LU-factorise C₀ (one n × n LU), solve once for the auxiliary
#   J = C₀⁻¹ · N = C₀⁻¹ · (A · E_y) ∈ ℝ^{n × n_y},
# split J = [J_x; J_y], and Schur-factorise J_y and h_x:
#   J_y = Q_y · S_y · Q_y',     h_x = Q_h · S_h · Q_h'.
# These caches are shared across the JVP, the VJP, and all tangent /
# cotangent directions of either.

function _klein_structured_plan(A, B, g_x, h_x)
    T = promote_type(eltype(A), eltype(B), eltype(g_x), eltype(h_x))
    n = size(A, 1)
    n_x = size(h_x, 1)
    n_y = n - n_x

    A_mat = Matrix{T}(A)
    B_mat = Matrix{T}(B)
    g_mat = Matrix{T}(g_x)
    h_mat = Matrix{T}(h_x)

    # Ψ = G = [I_{n_x}; g_x],   E_y = [0; I_{n_y}].
    I_x = Matrix{T}(LinearAlgebra.I, n_x, n_x)
    I_y = Matrix{T}(LinearAlgebra.I, n_y, n_y)
    G = vcat(I_x, g_mat)
    E_y = vcat(zeros(T, n_x, n_y), I_y)

    # M = A·Ψ, N = A·E_y, P = B·E_y; C₀ = [M | P].
    AG = A_mat * G
    AE_y = A_mat * E_y
    BE_y = B_mat * E_y
    C0 = hcat(AG, BE_y)
    F0 = LinearAlgebra.lu(C0)               # cached LU of C₀.

    # J = C₀⁻¹ · N, split into J_x (top n_x rows) and J_y (bottom n_y).
    J = F0 \ AE_y
    Jx = copy(view(J, 1:n_x, :))
    Jy = copy(view(J, (n_x + 1):n, :))

    # Real Schur of J_y and h_x — used by `sylvds!` in JVP and VJP.
    Fy = LinearAlgebra.schur(Jy)
    Fh = LinearAlgebra.schur(h_mat)

    return (;
        G, h_x = h_mat, F0, Jx,
        Sy = Fy.T, Qy = Fy.Z, Sh = Fh.T, Qh = Fh.Z,
        n, n_x, n_y,
    )
end

# Docs § Reduced-Sylvester JVP: Y = C₀⁻¹·R with R = −(dA·Ψ·h_x + dB·Ψ);
# bottom block gives the Stein S_y·X̃·S_h + X̃ = Q_y'·Y_y·Q_h in the
# (Q_y, Q_h) Schur frame; top block gives d h_x = Y_x − J_x · d g_x · h_x.

function _klein_structured_jvp(plan, dA, dB)
    (; G, h_x, F0, Jx, Sy, Qy, Sh, Qh, n, n_x) = plan
    T = eltype(G)
    dA_mat = Matrix{T}(dA)
    dB_mat = Matrix{T}(dB)

    # R = −(dA·Ψ·h_x + dB·Ψ);  Y = C₀⁻¹ · R.
    R = -((dA_mat * G) * h_x + dB_mat * G)
    Y = F0 \ R

    Yx = view(Y, 1:n_x, :)               # top block: drives d h_x
    Yy = view(Y, (n_x + 1):n, :)         # bottom block: drives d g_x

    # Reduced Stein in Schur coordinates, solved via `sylvds!`.
    Xtilde = transpose(Qy) * Yy * Qh     # Q_y' · Y_y · Q_h
    sylvds!(Sy, Sh, Xtilde)              # solves S_y·X̃·S_h + X̃ = rhs
    dg = Qy * Xtilde * transpose(Qh)     # d g_x = Q_y · X̃ · Q_h'

    # d h_x = Y_x − J_x · d g_x · h_x.
    dh = copy(Yx)
    dh .-= (Jx * dg) * h_x

    return (; g_x = dg, h_x = dh)
end

# Docs § Reduced-Sylvester VJP: corrected g̃̄_x = ḡ_x − J_x'·h̄_x·h_x'
# drives the adjoint Stein J_y'·Z·h_x' + Z = g̃̄_x; Λ = C₀⁻ᵀ · [h̄_x; Z].
# Same Ā / B̄ outer products as the big-K rule.

function _klein_structured_vjp(plan, g_bar, h_bar)
    (; G, h_x, F0, Jx, Sy, Qy, Sh, Qh, n, n_x) = plan
    T = eltype(G)
    g_bar_mat = Matrix{T}(g_bar)
    h_bar_mat = Matrix{T}(h_bar)

    # Corrected g̃̄_x = ḡ_x − J_x' · h̄_x · h_x'.
    X_bar = g_bar_mat - (transpose(Jx) * h_bar_mat) * transpose(h_x)
    # Adjoint Stein in Schur coordinates.
    Ztilde = transpose(Qy) * X_bar * Qh
    sylvds!(Sy, Sh, Ztilde; adjA = true, adjB = true)
    Z = Qy * Ztilde * transpose(Qh)      # Ȳ_y = Z

    # Reassemble Ȳ = [h̄_x; Z], solve Λ = C₀⁻ᵀ · Ȳ.
    Y_bar = Matrix{T}(undef, n, n_x)
    copyto!(view(Y_bar, 1:n_x, :), h_bar_mat)
    copyto!(view(Y_bar, (n_x + 1):n, :), Z)
    Lambda = F0' \ Y_bar

    # Parameter cotangents (identical to big-K formula — see docs).
    A_bar = -(Lambda * transpose(h_x)) * transpose(G)
    B_bar = -Lambda * transpose(G)
    return (; A = A_bar, B = B_bar)
end
