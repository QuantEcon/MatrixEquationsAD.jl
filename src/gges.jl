# This wrapper is adapted from FastLapackInterface.jl's generalized Schur
# implementation:
# https://github.com/DynareJulia/FastLapackInterface.jl/blob/main/src/schur.jl
# Switch back to that implementation if it becomes fully maintained for this
# narrow ordered-QZ use case.

const GGES_FLOAT = Union{Float32, Float64}
const DEFAULT_GGES_CRITERIUM = (1 - DEFAULT_BK_THRESHOLD)^2
const GGES_CRITERIUM = Ref{Float64}(DEFAULT_GGES_CRITERIUM)
const GGES_LOCK = ReentrantLock()

function _gges_ed_select32(αr_::Ptr{Float32}, αi_::Ptr{Float32}, β_::Ptr{Float32})::Cint
    αr = unsafe_load(αr_)
    αi = unsafe_load(αi_)
    β = unsafe_load(β_)
    criterium = Float32(GGES_CRITERIUM[])
    return αr^2 + αi^2 >= criterium * β^2 ? Cint(1) : Cint(0)
end

function _gges_ed_select64(αr_::Ptr{Float64}, αi_::Ptr{Float64}, β_::Ptr{Float64})::Cint
    αr = unsafe_load(αr_)
    αi = unsafe_load(αi_)
    β = unsafe_load(β_)
    criterium = GGES_CRITERIUM[]
    return αr^2 + αi^2 >= criterium * β^2 ? Cint(1) : Cint(0)
end

function gges(
        A::AbstractMatrix, B::AbstractMatrix;
        select::Symbol = :ed, criterium = DEFAULT_GGES_CRITERIUM
    )
    n = _gges_check_pair(A, B)
    T = promote_type(eltype(A), eltype(B))
    T <: GGES_FLOAT ||
        throw(ArgumentError("gges only supports Float32 and Float64 matrices"))

    S = Matrix{T}(undef, n, n)
    Tmat = Matrix{T}(undef, n, n)
    Q = Matrix{T}(undef, n, n)
    Z = Matrix{T}(undef, n, n)
    return gges!(S, Tmat, Q, Z, A, B; select, criterium)
end

function gges!(
        S::AbstractMatrix, Tmat::AbstractMatrix,
        Q::AbstractMatrix, Z::AbstractMatrix,
        A::AbstractMatrix, B::AbstractMatrix;
        select::Symbol = :ed, criterium = DEFAULT_GGES_CRITERIUM
    )
    n = _gges_check_pair(A, B)
    _gges_check_output(:S, S, n)
    _gges_check_output(:T, Tmat, n)
    _gges_check_output(:Q, Q, n)
    _gges_check_output(:Z, Z, n)

    return _gges!(S, Tmat, Q, Z, A, B, select, criterium)
end

function _gges!(
        S::StridedMatrix{T}, Tmat::StridedMatrix{T},
        Q::StridedMatrix{T}, Z::StridedMatrix{T},
        A::AbstractMatrix, B::AbstractMatrix,
        select::Symbol, criterium
    ) where {T <: GGES_FLOAT}
    _gges_check_select(select)
    _gges_check_criterium(criterium)
    Base.require_one_based_indexing(S, Tmat, Q, Z, A, B)
    chkstride1(S, Tmat, Q, Z)

    copyto!(S, A)
    copyto!(Tmat, B)
    n_explosive = _gges_lapack_ed!(S, Tmat, Q, Z, T(criterium))
    return (; S, T = Tmat, Q, Z, n_explosive)
end

function _gges_ordschur!(
        S::StridedMatrix, Tmat::StridedMatrix,
        Q::StridedMatrix, Z::StridedMatrix,
        A::AbstractMatrix, B::AbstractMatrix,
        select::Symbol, criterium
    )
    _gges_check_select(select)
    _gges_check_criterium(criterium)

    F = schur(A, B)
    n = length(F.α)
    selection = Vector{Bool}(undef, n)
    @inbounds for i in 1:n
        selection[i] = abs2(F.α[i]) >= criterium * abs2(F.β[i])
    end
    n_explosive = count(selection)
    ordschur!(F, selection)
    copyto!(S, F.S)
    copyto!(Tmat, F.T)
    copyto!(Q, F.Q)
    copyto!(Z, F.Z)
    return (; S, T = Tmat, Q, Z, n_explosive)
end

function _gges_check_pair(A::AbstractMatrix, B::AbstractMatrix)
    n = checksquare(A)
    m = checksquare(B)
    n == m ||
        throw(DimensionMismatch("dimensions of A, ($n,$n), and B, ($m,$m), must match"))
    return n
end

function _gges_check_output(name::Symbol, A::AbstractMatrix, n::Integer)
    size(A) == (n, n) ||
        throw(DimensionMismatch("output $name has size $(size(A)); expected ($n, $n)"))
    return nothing
end

function _gges_check_select(select::Symbol)
    select === :ed ||
        throw(ArgumentError("unsupported generalized Schur selector :$select; only :ed is supported"))
    return nothing
end

function _gges_check_criterium(criterium)
    criterium isa Real || throw(ArgumentError("criterium must be real"))
    isfinite(criterium) || throw(ArgumentError("criterium must be finite"))
    criterium >= zero(criterium) || throw(ArgumentError("criterium must be nonnegative"))
    return nothing
end

for (gges_lapack, elty, select_func) in (
        (:dgges_, :Float64, :_gges_ed_select64),
        (:sgges_, :Float32, :_gges_ed_select32),
    )
    @eval begin
        function _gges_lapack_ed!(
                S::StridedMatrix{$elty}, Tmat::StridedMatrix{$elty},
                Q::StridedMatrix{$elty}, Z::StridedMatrix{$elty},
                criterium::$elty
            )
            n = checksquare(S)
            checksquare(Tmat) == n ||
                throw(DimensionMismatch("dimensions of S and T must match"))
            _gges_check_output(:Q, Q, n)
            _gges_check_output(:Z, Z, n)
            Base.require_one_based_indexing(S, Tmat, Q, Z)
            chkstride1(S, Tmat, Q, Z)

            αr = Vector{$elty}(undef, n)
            αi = Vector{$elty}(undef, n)
            β = Vector{$elty}(undef, n)
            bwork = Vector{BlasInt}(undef, n)
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            sdim = Ref{BlasInt}(0)
            info = Ref{BlasInt}()
            sel_func = @cfunction(
                $select_func, Cint, (Ptr{$elty}, Ptr{$elty}, Ptr{$elty})
            )

            lock(GGES_LOCK)
            old_criterium = GGES_CRITERIUM[]
            try
                GGES_CRITERIUM[] = Float64(criterium)
                ccall(
                    (@blasfunc($gges_lapack), liblapack), Cvoid,
                    (
                        Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid},
                        Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                        Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty},
                        Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                        Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{Cvoid},
                        Ref{BlasInt}, Clong, Clong, Clong,
                    ),
                    'V', 'V', 'S', sel_func,
                    n, S, max(1, stride(S, 2)), Tmat, max(1, stride(Tmat, 2)),
                    sdim, αr, αi, β, Q, max(1, stride(Q, 2)),
                    Z, max(1, stride(Z, 2)), work, lwork, bwork, info, 1, 1, 1
                )
                chklapackerror(info[])

                lwork = max(BlasInt(1), BlasInt(real(work[1])))
                resize!(work, lwork)
                ccall(
                    (@blasfunc($gges_lapack), liblapack), Cvoid,
                    (
                        Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid},
                        Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                        Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty},
                        Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                        Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{Cvoid},
                        Ref{BlasInt}, Clong, Clong, Clong,
                    ),
                    'V', 'V', 'S', sel_func,
                    n, S, max(1, stride(S, 2)), Tmat, max(1, stride(Tmat, 2)),
                    sdim, αr, αi, β, Q, max(1, stride(Q, 2)),
                    Z, max(1, stride(Z, 2)), work, lwork, bwork, info, 1, 1, 1
                )
                chklapackerror(info[])
            finally
                GGES_CRITERIUM[] = old_criterium
                unlock(GGES_LOCK)
            end

            return Int(sdim[])
        end
    end
end
