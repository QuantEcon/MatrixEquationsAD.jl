# Klein (2000) / Sims policy-function extraction from an ordered
# generalised Schur decomposition of (A, B), using stock LinearAlgebra
# routines.
#
# Public API:
#   klein_map(A, B; threshold)          → (; g_x, h_x)
#   klein_map!(g_x, h_x, A, B; threshold)
#   klein_map(A::SMatrix, B::SMatrix, Val(n_x); threshold)
#       (StaticArrays ext, typed output sizes)
#
# n_x is inferred from the BK selection at runtime for heap and in-place APIs.
# StaticArrays inputs without explicit Val sizes intentionally fall through to
# the heap AbstractMatrix method.

# Apply the Klein/Sims algebra given an already-ordered generalized-Schur
# factorization. Mutates `g_x` and `h_x`; allocates blob/temp scratch.
#
# Implements docs § Algorithm step 3 with the partitioning
#   b = 1:n_x  (unstable / "big"),    l = (n_x+1):n  (stable),
# and the closed forms (where `S` and `T` here use the docs convention,
# i.e. `A = Q S Z'`, `B = Q T Z'`):
#   g_x =  −(Z_{ll})⁻ᵀ · (Z_{bl})ᵀ                    ∈ ℝ^{n_y × n_x}
#   h_x =  −W⁻¹ · S_{bb}⁻¹ · T_{bb} · W,
#     with  W = (Z_{bb})ᵀ + (Z_{lb})ᵀ · g_x          ∈ ℝ^{n_x × n_x}.
# The implementation writes into `g_x` and `h_x` in place and only
# allocates the n_x × n_x `blob` (= W) and `temp` (= T_{bb} · W) scratch
# matrices.

function _klein_extract!(
        g_x::AbstractMatrix, h_x::AbstractMatrix,
        S::AbstractMatrix, T::AbstractMatrix, Z::AbstractMatrix,
        n_x::Integer,
    )
    n = size(S, 1)
    b = 1:n_x
    l = (n_x + 1):n

    # ── g_x = −(Z_{ll})⁻ᵀ · (Z_{bl})ᵀ ────────────────────────────────────
    # Solve (Z_{ll})ᵀ · g_x = (Z_{bl})ᵀ via LU, then flip sign.
    Zll_T = Matrix(transpose(@view Z[l, l]))
    g_x .= transpose(@view Z[b, l])
    F_Zll = lu!(Zll_T)
    ldiv!(F_Zll, g_x)
    g_x .*= -1

    # ── W = (Z_{bb})ᵀ + (Z_{lb})ᵀ · g_x ──────────────────────────────────
    # Initialise blob with (Z_{bb})ᵀ, accumulate the (Z_{lb})ᵀ · g_x term.
    blob = Matrix(transpose(@view Z[b, b]))
    mul!(blob, transpose(@view Z[l, b]), g_x, 1.0, 1.0)

    # ── temp = T_{bb} · W ─────────────────────────────────────────────────
    # T_{bb} is the leading n_x × n_x block of the upper-triangular T.
    temp = Matrix{eltype(blob)}(undef, n_x, n_x)
    mul!(temp, UpperTriangular(@view T[b, b]), blob)

    # ── h_x = −W⁻¹ · S_{bb}⁻¹ · temp ─────────────────────────────────────
    # Stage 1: solve S_{bb} · X = temp in place via LU of S_{bb}. The
    # h_x buffer is repurposed as scratch for the S_{bb} factorisation
    # (its final value is computed in stage 2 below).
    @inbounds for j in 1:n_x, i in 1:n_x
        h_x[i, j] = S[i, j]
    end
    F_Sbb = lu!(h_x)
    ldiv!(F_Sbb, temp)
    # Stage 2: solve W · X = (S_{bb}⁻¹ · temp) and negate.
    F_blob = lu!(blob)
    ldiv!(F_blob, temp)
    @. h_x = -temp

    return (; g_x, h_x)
end

# Generalised real Schur factorisation `A = Q·S·Z'`, `B = Q·T·Z'` of the
# pencil (A, B), with eigenvalues `(α_i, β_i)`. Reorder so the *unstable*
# block — docs § Algorithm step 2: |α_i| ≥ (1 − τ)|β_i| — comes first.
# Returns the reordered factorisation (we only need `S`, `T`, `Z`) and
# the BK count. `n_x` here is the docs' BK count; the caller must check
# 0 < n_x < n and matching `size(h_x, 1)`.

function _klein_ordered_schur(A::AbstractMatrix, B::AbstractMatrix, threshold)
    F = schur(A, B)
    inds = abs.(F.α) .>= (1 - threshold) .* abs.(F.β)
    n_x = count(inds)
    ordschur!(F, inds)
    return F, n_x
end

function klein_map(
        A::AbstractMatrix, B::AbstractMatrix;
        threshold = 1.0e-6,
    )
    n = checksquare(A)
    checksquare(B) == n ||
        throw(DimensionMismatch("A and B must have matching square sizes"))
    F, n_x = _klein_ordered_schur(A, B, threshold)
    if n_x == 0 || n_x == n
        throw(
            ErrorException(
                "Blanchard-Kahn condition not satisfied: " *
                    "BK selection returned $n_x stable eigenvalues for an " *
                    "$(n)x$(n) gschur input (need 0 < n_x < n)",
            ),
        )
    end
    Tel = promote_type(eltype(A), eltype(B))
    g_x = zeros(Tel, n - n_x, n_x)
    h_x = zeros(Tel, n_x, n_x)
    return _klein_extract!(g_x, h_x, F.S, F.T, F.Z, n_x)
end

function klein_map!(
        g_x::AbstractMatrix, h_x::AbstractMatrix,
        A::AbstractMatrix, B::AbstractMatrix;
        threshold = 1.0e-6,
    )
    n = checksquare(A)
    checksquare(B) == n ||
        throw(DimensionMismatch("A and B must have matching square sizes"))
    n_x = checksquare(h_x)
    size(g_x) == (n - n_x, n_x) ||
        throw(DimensionMismatch("g_x must be $(n - n_x)×$n_x for h_x of size $(n_x)×$(n_x)"))
    F, sdim = _klein_ordered_schur(A, B, threshold)
    if sdim != n_x
        throw(
            ErrorException(
                "Blanchard-Kahn condition not satisfied: " *
                    "expected $n_x stable eigenvalues (from size(h_x, 1)), found $sdim",
            ),
        )
    end
    return _klein_extract!(g_x, h_x, F.S, F.T, F.Z, n_x)
end
