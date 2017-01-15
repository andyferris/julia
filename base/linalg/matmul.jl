# This file is a part of Julia. License is MIT: http://julialang.org/license

# matmul.jl: Everything to do with generic matrix multiplication.
#            (see blas.jl for dense implementations)

# for output element type inference
matprod(x, y) = x*y + x*y

# We support the follow vector/matrix multiplication patterns
#
# 1) rowvector * vector   (inner product -> scalar)   (BLAS level 1)
# 2) vector * rowvector   (outer product -> matrix)   (BLAS level 2) (also vector * 1xn matrix, and nx1 matrix * rowvector)
# 3) matrix * vector      (-> vector)                 (BLAS level 2)
# 4) rowvector * matrix   (-> rowvector)              (BLAS level 2)
# 5) matrix * matrix      (-> matrix)                 (BLAS level 3)

*(a::AbstractArray, b::AbstractArray) = error("Cannot multiply arrays of dimension $(ndims(a)) and $(ndims(b)). Try `.*` for elementwise/broadcasting multiplication.")

# ============================
# Generic non-mutating methods
# ============================

# inner product
@inline function *(rowvec::RowVector, vec::AbstractVector)
    if length(rowvec) != length(vec)
        throw(DimensionMismatch("A has dimensions $(size(rowvec)) but B has dimensions $(size(vec))"))
    end
    sum(@inbounds(return rowvec[i]*vec[i]) for i = 1:length(vec))
end

# outer product
@inline *(vec::AbstractVector, rowvec::RowVector) = vec .* rowvec
@inline *(vec::AbstractVector, mat::AbstractMatrix) = reshape(vec, (length(vec), 1)) * mat
@inline *(mat::AbstractMatrix, rowvec::RowVector) = mat * reshape(rowvec.', (1, length(rowvec)))

# matrix-vector product
@inline function (*){T,S}(mat::AbstractMatrix{T}, vec::AbstractVector{S})
    TS = promote_op(matprod, S)
    A_mul_B!(similar(vec, TS, (size(mat,1),)), mat, vec)
end

# rowvector-matrix product
@inline *(rowvec::RowVector, mat::AbstractMatrix) = transpose(transpose(mat)*transpose(rowvec))

# matrix-matrix product
"""
```
*(A::AbstractMatrix, B::AbstractMatrix)
```

Matrix multiplication.

# Example

```jldoctest
julia> [1 1; 0 1] * [1 0; 1 1]
2×2 Array{Int64,2}:
 2  1
 1  1
```
"""
@inline function (*){T,S}(A::AbstractMatrix{T}, B::AbstractMatrix{S})
    TS = promote_op(matprod, T, S)
    A_mul_B!(similar(B, TS, (size(A,1), size(B,2))), A, B)
end


# ========================
# Generic mutating methods
# ========================

# We currently support this - factor it to a generic method to simplify other methods
function A_mul_B!(z::AbstractVector, x::AbstractVector, y::AbstractVector)
    if length(x) != 1 || length(y) != 1
        throw(DimensionMismatch("input A has dimensions ($(length(x))), vector B has length $(length(y))"))
    end
    if length(z) != 1
        throw(DimensionMismatch("result C has length $(length(z)), needs length 1"))
    end
    @inbounds z[1] = x[1] * y[1]
    z
end

# Matrix-vector product
A_mul_B!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector) = generic_matvecmul!(y, 'N', A, x)
A_mul_B!(y::AbstractVector, A::TransposedMatrix, x::AbstractVector) = generic_matvecmul!(y, 'T', A.', x)
A_mul_B!(y::AbstractVector, A::ConjTransposedMatrix, x::AbstractVector) = generic_matvecmul!(y, 'C', A', x)

"""
    A_mul_B!(C, A, B) -> C

Calculates the matrix-matrix or matrix-vector product ``A⋅B`` and stores the result in `C`,
overwriting the existing value of `C`. Note that `C` must not be aliased with either `A` or
`B`.

# Example

```jldoctest
julia> A=[1.0 2.0; 3.0 4.0]; B=[1.0 1.0; 1.0 1.0]; Y = similar(B); A_mul_B!(Y, A, B);

julia> Y
2×2 Array{Float64,2}:
 3.0  3.0
 7.0  7.0
```
"""
A_mul_B!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix) = generic_matmatmul!(C, 'N', 'N', A, B)
A_mul_B!(C::AbstractMatrix, A::TransposedMatrix, B::AbstractMatrix) = generic_matmatmul!(C, 'T', 'N', A.', B)
A_mul_B!(C::AbstractMatrix, A::ConjTransposedMatrix, B::AbstractMatrix) = generic_matmatmul!(C, 'C', 'N', A', B)
A_mul_B!(C::AbstractMatrix, A::AbstractMatrix, B::TransposedMatrix) = generic_matmatmul!(C, 'N', 'T', A, B.')
A_mul_B!(C::AbstractMatrix, A::AbstractMatrix, B::ConjTransposedMatrix) = generic_matmatmul!(C, 'N', 'C', A, B')
A_mul_B!(C::AbstractMatrix, A::TransposedMatrix, B::TransposedMatrix) = generic_matmatmul!(C, 'T', 'T', A.', B.')
A_mul_B!(C::AbstractMatrix, A::ConjTransposedMatrix, B::ConjTransposedMatrix) = generic_matmatmul!(C, 'C', 'C', A', B')




# TODO make sure these work out:

# inner product -> dot product specializations
#@inline *{T<:Real}(rowvec::RowVector{T}, vec::AbstractVector{T}) = dot(parent(rowvec), vec)
#@inline *(rowvec::ConjRowVector, vec::AbstractVector) = dot(rowvec', vec)


# Supporting functions for matrix multiplication

function copytri!(A::AbstractMatrix, uplo::Char, conjugate::Bool=false)
    n = checksquare(A)
    if uplo == 'U'
        for i = 1:(n-1), j = (i+1):n
            A[j,i] = conjugate ? conj(A[i,j]) : A[i,j]
        end
    elseif uplo == 'L'
        for i = 1:(n-1), j = (i+1):n
            A[i,j] = conjugate ? conj(A[j,i]) : A[j,i]
        end
    else
        throw(ArgumentError("uplo argument must be 'U' (upper) or 'L' (lower), got $uplo"))
    end
    A
end

lapack_size(t::Char, M::AbstractVecOrMat) = (size(M, t=='N' ? 1:2), size(M, t=='N' ? 2:1))

function copy!(B::AbstractVecOrMat, ir_dest::UnitRange{Int}, jr_dest::UnitRange{Int}, tM::Char, M::AbstractVecOrMat, ir_src::UnitRange{Int}, jr_src::UnitRange{Int})
    if tM == 'N'
        copy!(B, ir_dest, jr_dest, M, ir_src, jr_src)
    else
        Base.copy_transpose!(B, ir_dest, jr_dest, M, jr_src, ir_src)
        tM == 'C' && conj!(B)
    end
    B
end

function copy_transpose!(B::AbstractMatrix, ir_dest::UnitRange{Int}, jr_dest::UnitRange{Int}, tM::Char, M::AbstractVecOrMat, ir_src::UnitRange{Int}, jr_src::UnitRange{Int})
    if tM == 'N'
        Base.copy_transpose!(B, ir_dest, jr_dest, M, ir_src, jr_src)
    else
        copy!(B, ir_dest, jr_dest, M, jr_src, ir_src)
        tM == 'C' && conj!(B)
    end
    B
end

# TODO: It will be faster for large matrices to convert to float,
# call BLAS, and convert back to required type.

# NOTE: the generic version is called from BLAS methods as fallback for strides != 1 cases

function generic_matvecmul!{R}(C::AbstractVector{R}, tA, A::AbstractVecOrMat, B::AbstractVector)
    mB = length(B)
    mA, nA = lapack_size(tA, A)
    if mB != nA
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA), vector B has length $mB"))
    end
    if mA != length(C)
        throw(DimensionMismatch("result C has length $(length(C)), needs length $mA"))
    end

    Astride = size(A, 1)

    if tA == 'T'  # fastest case
        for k = 1:mA
            aoffs = (k-1)*Astride
            if mB == 0
                s = zero(R)
            else
                s = zero(A[aoffs + 1]*B[1] + A[aoffs + 1]*B[1])
            end
            for i = 1:nA
                s += A[aoffs+i].'B[i]
            end
            C[k] = s
        end
    elseif tA == 'C'
        for k = 1:mA
            aoffs = (k-1)*Astride
            if mB == 0
                s = zero(R)
            else
                s = zero(A[aoffs + 1]*B[1] + A[aoffs + 1]*B[1])
            end
            for i = 1:nA
                s += A[aoffs + i]'B[i]
            end
            C[k] = s
        end
    else # tA == 'N'
        for i = 1:mA
            if mB == 0
                C[i] = zero(R)
            else
                C[i] = zero(A[i]*B[1] + A[i]*B[1])
            end
        end
        for k = 1:mB
            aoffs = (k-1)*Astride
            b = B[k]
            for i = 1:mA
                C[i] += A[aoffs + i] * b
            end
        end
    end
    C
end

function generic_matmatmul{T,S}(tA, tB, A::AbstractVecOrMat{T}, B::AbstractMatrix{S})
    mA, nA = lapack_size(tA, A)
    mB, nB = lapack_size(tB, B)
    C = similar(B, promote_op(matprod, T, S), mA, nB)
    generic_matmatmul!(C, tA, tB, A, B)
end

const tilebufsize = 10800  # Approximately 32k/3
const Abuf = Array{UInt8}(tilebufsize)
const Bbuf = Array{UInt8}(tilebufsize)
const Cbuf = Array{UInt8}(tilebufsize)

function generic_matmatmul!(C::AbstractMatrix, tA, tB, A::AbstractMatrix, B::AbstractMatrix)
    mA, nA = lapack_size(tA, A)
    mB, nB = lapack_size(tB, B)
    mC, nC = size(C)

    if mA == nA == mB == nB == mC == nC == 2
        return matmul2x2!(C, tA, tB, A, B)
    end
    if mA == nA == mB == nB == mC == nC == 3
        return matmul3x3!(C, tA, tB, A, B)
    end
    _generic_matmatmul!(C, tA, tB, A, B)
end

generic_matmatmul!(C::AbstractVecOrMat, tA, tB, A::AbstractVecOrMat, B::AbstractVecOrMat) = _generic_matmatmul!(C, tA, tB, A, B)

function _generic_matmatmul!{T,S,R}(C::AbstractVecOrMat{R}, tA, tB, A::AbstractVecOrMat{T}, B::AbstractVecOrMat{S})
    mA, nA = lapack_size(tA, A)
    mB, nB = lapack_size(tB, B)
    if mB != nA
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA), matrix B has dimensions ($mB,$nB)"))
    end
    if size(C,1) != mA || size(C,2) != nB
        throw(DimensionMismatch("result C has dimensions $(size(C)), needs ($mA,$nB)"))
    end
    if isempty(A) || isempty(B)
        return fill!(C, zero(R))
    end

    tile_size = 0
    if isbits(R) && isbits(T) && isbits(S) && (tA == 'N' || tB != 'N')
        tile_size = floor(Int,sqrt(tilebufsize/max(sizeof(R),sizeof(S),sizeof(T))))
    end
    @inbounds begin
    if tile_size > 0
        sz = (tile_size, tile_size)
        Atile = unsafe_wrap(Array, convert(Ptr{T}, pointer(Abuf)), sz)
        Btile = unsafe_wrap(Array, convert(Ptr{S}, pointer(Bbuf)), sz)

        z1 = zero(A[1, 1]*B[1, 1] + A[1, 1]*B[1, 1])
        z = convert(promote_type(typeof(z1), R), z1)

        if mA < tile_size && nA < tile_size && nB < tile_size
            Base.copy_transpose!(Atile, 1:nA, 1:mA, tA, A, 1:mA, 1:nA)
            copy!(Btile, 1:mB, 1:nB, tB, B, 1:mB, 1:nB)
            for j = 1:nB
                boff = (j-1)*tile_size
                for i = 1:mA
                    aoff = (i-1)*tile_size
                    s = z
                    for k = 1:nA
                        s += Atile[aoff+k] * Btile[boff+k]
                    end
                    C[i,j] = s
                end
            end
        else
            Ctile = unsafe_wrap(Array, convert(Ptr{R}, pointer(Cbuf)), sz)
            for jb = 1:tile_size:nB
                jlim = min(jb+tile_size-1,nB)
                jlen = jlim-jb+1
                for ib = 1:tile_size:mA
                    ilim = min(ib+tile_size-1,mA)
                    ilen = ilim-ib+1
                    fill!(Ctile, z)
                    for kb = 1:tile_size:nA
                        klim = min(kb+tile_size-1,mB)
                        klen = klim-kb+1
                        Base.copy_transpose!(Atile, 1:klen, 1:ilen, tA, A, ib:ilim, kb:klim)
                        copy!(Btile, 1:klen, 1:jlen, tB, B, kb:klim, jb:jlim)
                        for j=1:jlen
                            bcoff = (j-1)*tile_size
                            for i = 1:ilen
                                aoff = (i-1)*tile_size
                                s = z
                                for k = 1:klen
                                    s += Atile[aoff+k] * Btile[bcoff+k]
                                end
                                Ctile[bcoff+i] += s
                            end
                        end
                    end
                    copy!(C, ib:ilim, jb:jlim, Ctile, 1:ilen, 1:jlen)
                end
            end
        end
    else
        # Multiplication for non-plain-data uses the naive algorithm

        if tA == 'N'
            if tB == 'N'
                for i = 1:mA, j = 1:nB
                    z2 = zero(A[i, 1]*B[1, j] + A[i, 1]*B[1, j])
                    Ctmp = convert(promote_type(R, typeof(z2)), z2)
                    for k = 1:nA
                        Ctmp += A[i, k]*B[k, j]
                    end
                    C[i,j] = Ctmp
                end
            elseif tB == 'T'
                for i = 1:mA, j = 1:nB
                    z2 = zero(A[i, 1]*B[j, 1] + A[i, 1]*B[j, 1])
                    Ctmp = convert(promote_type(R, typeof(z2)), z2)
                    for k = 1:nA
                        Ctmp += A[i, k]*B[j, k].'
                    end
                    C[i,j] = Ctmp
                end
            else
                for i = 1:mA, j = 1:nB
                    z2 = zero(A[i, 1]*B[j, 1] + A[i, 1]*B[j, 1])
                    Ctmp = convert(promote_type(R, typeof(z2)), z2)
                    for k = 1:nA
                        Ctmp += A[i, k]*B[j, k]'
                    end
                    C[i,j] = Ctmp
                end
            end
        elseif tA == 'T'
            if tB == 'N'
                for i = 1:mA, j = 1:nB
                    z2 = zero(A[1, i]*B[1, j] + A[1, i]*B[1, j])
                    Ctmp = convert(promote_type(R, typeof(z2)), z2)
                    for k = 1:nA
                        Ctmp += A[k, i].'B[k, j]
                    end
                    C[i,j] = Ctmp
                end
            elseif tB == 'T'
                for i = 1:mA, j = 1:nB
                    z2 = zero(A[1, i]*B[j, 1] + A[1, i]*B[j, 1])
                    Ctmp = convert(promote_type(R, typeof(z2)), z2)
                    for k = 1:nA
                        Ctmp += A[k, i].'B[j, k].'
                    end
                    C[i,j] = Ctmp
                end
            else
                for i = 1:mA, j = 1:nB
                    z2 = zero(A[1, i]*B[j, 1] + A[1, i]*B[j, 1])
                    Ctmp = convert(promote_type(R, typeof(z2)), z2)
                    for k = 1:nA
                        Ctmp += A[k, i].'B[j, k]'
                    end
                    C[i,j] = Ctmp
                end
            end
        else
            if tB == 'N'
                for i = 1:mA, j = 1:nB
                    z2 = zero(A[1, i]*B[1, j] + A[1, i]*B[1, j])
                    Ctmp = convert(promote_type(R, typeof(z2)), z2)
                    for k = 1:nA
                        Ctmp += A[k, i]'B[k, j]
                    end
                    C[i,j] = Ctmp
                end
            elseif tB == 'T'
                for i = 1:mA, j = 1:nB
                    z2 = zero(A[1, i]*B[j, 1] + A[1, i]*B[j, 1])
                    Ctmp = convert(promote_type(R, typeof(z2)), z2)
                    for k = 1:nA
                        Ctmp += A[k, i]'B[j, k].'
                    end
                    C[i,j] = Ctmp
                end
            else
                for i = 1:mA, j = 1:nB
                    z2 = zero(A[1, i]*B[j, 1] + A[1, i]*B[j, 1])
                    Ctmp = convert(promote_type(R, typeof(z2)), z2)
                    for k = 1:nA
                        Ctmp += A[k, i]'B[j, k]'
                    end
                    C[i,j] = Ctmp
                end
            end
        end
    end
    end # @inbounds
    C
end


# multiply 2x2 matrices
function matmul2x2{T,S}(tA, tB, A::AbstractMatrix{T}, B::AbstractMatrix{S})
    matmul2x2!(similar(B, promote_op(matprod, T, S), 2, 2), tA, tB, A, B)
end

function matmul2x2!(C::AbstractMatrix, tA, tB, A::AbstractMatrix, B::AbstractMatrix)
    if !(size(A) == size(B) == size(C) == (2,2))
        throw(DimensionMismatch("A has size $(size(A)), B has size $(size(B)), C has size $(size(C))"))
    end
    @inbounds begin
    if tA == 'T'
        A11 = transpose(A[1,1]); A12 = transpose(A[2,1]); A21 = transpose(A[1,2]); A22 = transpose(A[2,2])
    elseif tA == 'C'
        A11 = ctranspose(A[1,1]); A12 = ctranspose(A[2,1]); A21 = ctranspose(A[1,2]); A22 = ctranspose(A[2,2])
    else
        A11 = A[1,1]; A12 = A[1,2]; A21 = A[2,1]; A22 = A[2,2]
    end
    if tB == 'T'
        B11 = transpose(B[1,1]); B12 = transpose(B[2,1]); B21 = transpose(B[1,2]); B22 = transpose(B[2,2])
    elseif tB == 'C'
        B11 = ctranspose(B[1,1]); B12 = ctranspose(B[2,1]); B21 = ctranspose(B[1,2]); B22 = ctranspose(B[2,2])
    else
        B11 = B[1,1]; B12 = B[1,2]; B21 = B[2,1]; B22 = B[2,2]
    end
    C[1,1] = A11*B11 + A12*B21
    C[1,2] = A11*B12 + A12*B22
    C[2,1] = A21*B11 + A22*B21
    C[2,2] = A21*B12 + A22*B22
    end # inbounds
    C
end

# Multiply 3x3 matrices
function matmul3x3{T,S}(tA, tB, A::AbstractMatrix{T}, B::AbstractMatrix{S})
    matmul3x3!(similar(B, promote_op(matprod, T, S), 3, 3), tA, tB, A, B)
end

function matmul3x3!(C::AbstractMatrix, tA, tB, A::AbstractMatrix, B::AbstractMatrix)
    if !(size(A) == size(B) == size(C) == (3,3))
        throw(DimensionMismatch("A has size $(size(A)), B has size $(size(B)), C has size $(size(C))"))
    end
    @inbounds begin
    if tA == 'T'
        A11 = transpose(A[1,1]); A12 = transpose(A[2,1]); A13 = transpose(A[3,1])
        A21 = transpose(A[1,2]); A22 = transpose(A[2,2]); A23 = transpose(A[3,2])
        A31 = transpose(A[1,3]); A32 = transpose(A[2,3]); A33 = transpose(A[3,3])
    elseif tA == 'C'
        A11 = ctranspose(A[1,1]); A12 = ctranspose(A[2,1]); A13 = ctranspose(A[3,1])
        A21 = ctranspose(A[1,2]); A22 = ctranspose(A[2,2]); A23 = ctranspose(A[3,2])
        A31 = ctranspose(A[1,3]); A32 = ctranspose(A[2,3]); A33 = ctranspose(A[3,3])
    else
        A11 = A[1,1]; A12 = A[1,2]; A13 = A[1,3]
        A21 = A[2,1]; A22 = A[2,2]; A23 = A[2,3]
        A31 = A[3,1]; A32 = A[3,2]; A33 = A[3,3]
    end

    if tB == 'T'
        B11 = transpose(B[1,1]); B12 = transpose(B[2,1]); B13 = transpose(B[3,1])
        B21 = transpose(B[1,2]); B22 = transpose(B[2,2]); B23 = transpose(B[3,2])
        B31 = transpose(B[1,3]); B32 = transpose(B[2,3]); B33 = transpose(B[3,3])
    elseif tB == 'C'
        B11 = ctranspose(B[1,1]); B12 = ctranspose(B[2,1]); B13 = ctranspose(B[3,1])
        B21 = ctranspose(B[1,2]); B22 = ctranspose(B[2,2]); B23 = ctranspose(B[3,2])
        B31 = ctranspose(B[1,3]); B32 = ctranspose(B[2,3]); B33 = ctranspose(B[3,3])
    else
        B11 = B[1,1]; B12 = B[1,2]; B13 = B[1,3]
        B21 = B[2,1]; B22 = B[2,2]; B23 = B[2,3]
        B31 = B[3,1]; B32 = B[3,2]; B33 = B[3,3]
    end

    C[1,1] = A11*B11 + A12*B21 + A13*B31
    C[1,2] = A11*B12 + A12*B22 + A13*B32
    C[1,3] = A11*B13 + A12*B23 + A13*B33

    C[2,1] = A21*B11 + A22*B21 + A23*B31
    C[2,2] = A21*B12 + A22*B22 + A23*B32
    C[2,3] = A21*B13 + A22*B23 + A23*B33

    C[3,1] = A31*B11 + A32*B21 + A33*B31
    C[3,2] = A31*B12 + A32*B22 + A33*B32
    C[3,3] = A31*B13 + A32*B23 + A33*B33
    end # inbounds
    C
end
