module MatrixEquationsADEnzymeStaticArraysExt

# Enzyme forward rule for the static-native `lyapdkr(::SMatrix, ::SMatrix)`
# path. Reuses one `lu(M)` factorization across the primal solve and every
# tangent solve. Per-tangent solves rather than batched — there's no BLAS-3
# to exploit on SMatrix and the static `F \ vec(rhs)` is already very fast.
#
# No reverse rule / augmented_primal: callers who need reverse on SMatrix
# inputs can use the heap path (or convert via `Matrix(A)` themselves).

using Enzyme.EnzymeRules: EnzymeRules
using Enzyme:
    Annotation, BatchDuplicated, BatchDuplicatedNoNeed, Const,
    Duplicated, DuplicatedNoNeed
using LinearAlgebra: lu
using MatrixEquationsAD: build_M!!, lyapdkr, symmetrize!!
using StaticArrays: SMatrix

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(lyapdkr)},
        ::Type{RT},
        A::Annotation{<:SMatrix{N, N, T}},
        C::Annotation{<:SMatrix{N, N, T}};
        M_ws = nothing,   # accepted for API parity with the heap rule; ignored
    ) where {RT <: Union{Const, Duplicated, DuplicatedNoNeed, BatchDuplicated, BatchDuplicatedNoNeed},
        N, T,
    }
    M = build_M!!(nothing, A.val)
    F = lu(M)
    X = symmetrize!!(SMatrix{N, N, T}(F \ vec(C.val)))

    Nw = EnzymeRules.width(config)
    dXs = ntuple(Val(Nw)) do i
        Base.@_inline_meta
        dC_i = if typeof(C) <: Const
            zero(SMatrix{N, N, T})
        else
            Nw == 1 ? C.dval : C.dval[i]
        end
        rhs = dC_i
        if !(typeof(A) <: Const)
            dA = Nw == 1 ? A.dval : A.dval[i]
            rhs = rhs + dA * X * A.val' + A.val * X * dA'
        end
        symmetrize!!(SMatrix{N, N, T}(F \ vec(rhs)))
    end

    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return Nw == 1 ? Duplicated(X, dXs[1]) : BatchDuplicated(X, dXs)
    elseif EnzymeRules.needs_shadow(config)
        return Nw == 1 ? dXs[1] : dXs
    elseif EnzymeRules.needs_primal(config)
        return X
    else
        return nothing
    end
end

end
