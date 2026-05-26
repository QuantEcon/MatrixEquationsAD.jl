# Docs § Big-K factorisation (OOP `klein_map`).
#
# Build the dense `(n n_x) × (n n_x)` Jacobian K of the implicit equation
# F(A, B, g_x, h_x) = A·Ψ·h_x + B·Ψ = 0 in the column-major vec ordering
#   v = [vec(d h_x); vec(d g_x)]    (h_x block first, then g_x).
# From docs § "Jacobian of the implicit equation",
#   K = [ I_{n_x} ⊗ M  |  h_x' ⊗ N + I_{n_x} ⊗ P ],
#   M = A·Ψ,   N = A·E_y,   P = B·E_y,   E_y = [0; I_{n_y}],   Ψ = [I_{n_x}; g_x].
# The triple loop below fills K in this column order block-by-block. The
# plan caches the LU `F = lu(K)`, plus the inputs `G` (= Ψ) and `h_x`
# needed by both JVP RHS construction and VJP outer products.

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

# Docs § Big-K JVP, Steps 1-3:
#   1. Differentiate (F): tangent is the linear system (K) for
#        v = [vec(d h_x); vec(d g_x)]
#      with rhs = −vec(d A·Ψ·h_x + d B·Ψ).
#   2. Cached LU of K (built by `_klein_bigk_plan`).
#   3. v = K⁻¹ · rhs, then reshape into (d h_x, d g_x).
# Block order matches the K assembly in the plan: h_x block first, g_x block second.

function _klein_bigk_jvp(plan, dA, dB)
    (; F, G, h_x, n_x, n_y) = plan
    T = eltype(G)
    dA_mat = Matrix{T}(dA)
    dB_mat = Matrix{T}(dB)

    # rhs = −(dA·Ψ·h_x + dB·Ψ) — `G = Ψ` in the plan.
    R = -((dA_mat * G) * h_x + dB_mat * G)
    x = F \ vec(R)

    # Split v back into (d h_x, d g_x).
    n_h = n_x * n_x
    dh = copy(reshape(view(x, 1:n_h), n_x, n_x))
    dg = copy(reshape(view(x, (n_h + 1):length(x)), n_y, n_x))
    return (; g_x = dg, h_x = dh)
end

# Docs § Big-K VJP, Steps 1-3:
#   1. Pack output cotangents in K's block order:  u = [vec(h̄); vec(ḡ)].
#   2. Λ = reshape(K⁻ᵀ · u, n, n_x).
#   3. Parameter cotangents (chain rule against the rhs = −(dA·Ψ·h_x + dB·Ψ)):
#        Ā += −Λ · h_x' · Ψ',     B̄ += −Λ · Ψ'.

function _klein_bigk_vjp(plan, g_bar, h_bar)
    (; F, G, h_x, n, n_x) = plan
    T = eltype(G)
    # Step 1: stack cotangents in K's vec ordering.
    rhs = vcat(vec(Matrix{T}(h_bar)), vec(Matrix{T}(g_bar)))
    # Step 2: transposed LU solve, then reshape into Λ.
    lambda = F' \ rhs
    Lambda = reshape(lambda, n, n_x)

    # Step 3: parameter outer products.
    A_bar = -(Lambda * transpose(h_x)) * transpose(G)   # −Λ · h_x' · Ψ'
    B_bar = -Lambda * transpose(G)                       # −Λ · Ψ'
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

# Docs § Reduced-Sylvester JVP, Steps 1-3:
#   1. Tangent equation: C₀ · [d h_x; d g_x] + N · d g_x · h_x = R,
#      with R = −(dA·Ψ·h_x + dB·Ψ). Premultiply by C₀⁻¹ and use J = C₀⁻¹·N:
#      let Y = C₀⁻¹·R, Y = [Y_x; Y_y]; the bottom block becomes the
#      reduced Sylvester / Stein equation
#        d g_x + J_y · d g_x · h_x = Y_y.
#   2. Cached LU of C₀ plus Schurs J_y = Q_y·S_y·Q_y' and h_x = Q_h·S_h·Q_h'.
#   3. In the Schur frame X̃ = Q_y' · d g_x · Q_h the equation becomes
#        S_y · X̃ · S_h + X̃ = Q_y' · Y_y · Q_h    (`sylvds!` convention),
#      then untransform: d g_x = Q_y · X̃ · Q_h'.  Finally
#        d h_x = Y_x − J_x · d g_x · h_x        (top-block equation).

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

# Docs § Reduced-Sylvester VJP, Steps 1-3:
#   1. Adjoint of `d h_x = Y_x − J_x · d g_x · h_x` gives
#        Ȳ_x = h̄_x,    g̃̄_x = ḡ_x − J_x' · h̄_x · h_x'.
#      The corrected g̃̄_x drives the adjoint Stein equation
#        J_y' · Z · h_x' + Z = g̃̄_x,    Ȳ_y = Z,
#      solved via `sylvds!(Sy, Sh, ·; adjA=true, adjB=true)` in the
#      Schur frame.
#   2. Reassemble  Ȳ = [Ȳ_x; Ȳ_y]  and solve  Λ = C₀⁻ᵀ · Ȳ.
#   3. Parameter cotangents (same as the big-K rule):
#        Ā += −Λ · h_x' · Ψ',    B̄ += −Λ · Ψ'.

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
