## Multiplication ##

@inline function *(rowvec::RowVector, vec::AbstractVector)
    length(vec) == length(rowvec) || throw(DimensionMismatch("A has dimensions $(size(rowvec)) but B has dimensions $(size(vec))"))
    sum( (@inbounds return rowvec[i] * vec[i]) for i = 1:length(vec) )
end
@inline *(rowvec::RowVector, mat::AbstractMatrix) = transpose(mat.' * transpose(rowvec))
*(vec::AbstractVector, mat::AbstractMatrix) = error("Cannot left-multiply a matrix by a vector") # Should become a deprecation
*(::RowVector, ::RowVector) = error("Cannot multiply two transposed vectors")
@inline *(vec::AbstractVector, rowvec::RowVector) = kron(vec, rowvec)
*(vec::AbstractVector, rowvec::AbstractVector) = error("Cannot multiply two vectors")
*(mat::AbstractMatrix, rowvec::RowVector) = error("Cannot right-multiply matrix by transposed vector")

# Transposed forms
A_mul_Bt(::RowVector, ::AbstractVector) = error("Cannot multiply two transposed vectors")
@inline A_mul_Bt(rowvec::RowVector, mat::AbstractMatrix) = transpose(mat * transpose(rowvec))
A_mul_Bt(vec::AbstractVector, mat::AbstractMatrix) = error("Cannot left-multiply a matrix by a vector")
@inline A_mul_Bt(rowvec1::RowVector, rowvec2::RowVector) = reduce(+, map(At_mul_B, transpose(rowvec1), transpose(rowvec2)))
A_mul_Bt(vec::AbstractVector, rowvec::RowVector) = error("Cannot multiply two vectors")
@inline A_mul_Bt(vec1::AbstractVector, vec2::AbstractVector) = kron(vec1, transpose(vec2))
@inline A_mul_Bt(mat::AbstractMatrix, rowvec::RowVector) = mat * transpose(rowvec)

@inline At_mul_Bt(rowvec::RowVector, vec::AbstractVector) = kron(transpose(vec), transpose(rowvec))
At_mul_Bt(rowvec::RowVector, mat::AbstractMatrix) = error("Cannot left-multiply matrix by vector")
@inline At_mul_Bt(vec::AbstractVector, mat::AbstractMatrix) = transpose(mat * vec)
At_mul_Bt(rowvec1::RowVector, rowvec2::RowVector) = error("Cannot multiply two vectors")
@inline At_mul_Bt(vec::AbstractVector, rowvec::RowVector) = reduce(+, map(At_mul_B, vec, transpose(rowvec)))
At_mul_Bt(vec::AbstractVector, rowvec::AbstractVector) = error("Cannot multiply two transposed vectors")
@inline At_mul_Bt(mat::AbstractMatrix, rowvec::RowVector) = mat.' * transpose(rowvec)

At_mul_B(::RowVector, ::AbstractVector) = error("Cannot multiply two vectors")
At_mul_B(rowvec::RowVector, mat::AbstractMatrix) = error("Cannot left-multiply matrix by vector")
@inline At_mul_B(vec::AbstractVector, mat::AbstractMatrix) = transpose(At_mul_B(mat,vec))
@inline At_mul_B(rowvec1::RowVector, rowvec2::RowVector) = kron(transpose(rowvec1), rowvec2)
At_mul_B(vec::AbstractVector, rowvec::RowVector) = error("Cannot multiply two transposed vectors")
@inline At_mul_B{T<:Real}(vec1::AbstractVector{T}, vec2::AbstractVector{T}) = reduce(+, map(At_mul_B, vec1, vec2)) # Seems to be overloaded...
@inline At_mul_B(vec1::AbstractVector, vec2::AbstractVector) = reduce(+, map(At_mul_B, vec1, vec2))
At_mul_B(mat::AbstractMatrix, rowvec::RowVector) = error("Cannot right-multiply matrix by transposed vector")

# Conjugated forms
A_mul_Bc(::RowVector, ::AbstractVector) = error("Cannot multiply two transposed vectors")
@inline A_mul_Bc(rowvec::RowVector, mat::AbstractMatrix) = ctranspose(mat * ctranspose(rowvec))
A_mul_Bc(vec::AbstractVector, mat::AbstractMatrix) = error("Cannot left-multiply a matrix by a vector")
@inline A_mul_Bc(rowvec1::RowVector, rowvec2::RowVector) = reduce(+, map(A_mul_Bc, rowvec1, rowvec2))
A_mul_Bc(vec::AbstractVector, rowvec::RowVector) = error("Cannot multiply two vectors")
@inline A_mul_Bc(vec1::AbstractVector, vec2::AbstractVector) = kron(vec1, ctranspose(vec2))
@inline A_mul_Bc(mat::AbstractMatrix, rowvec::RowVector) = mat * ctranspose(rowvec)

@inline Ac_mul_Bc(rowvec::RowVector, vec::AbstractVector) = kron(ctranspose(vec), ctranspose(rowvec))
Ac_mul_Bc(rowvec::RowVector, mat::AbstractMatrix) = error("Cannot left-multiply matrix by vector")
@inline Ac_mul_Bc(vec::AbstractVector, mat::AbstractMatrix) = ctranspose(mat * vec)
Ac_mul_Bc(rowvec1::RowVector, rowvec2::RowVector) = error("Cannot multiply two vectors")
@inline Ac_mul_Bc(vec::AbstractVector, rowvec::RowVector) = reduce(+, map(Ac_mul_Bc, vec, transpose(rowvec)))
Ac_mul_Bc(vec::AbstractVector, rowvec::AbstractVector) = error("Cannot multiply two transposed vectors")
@inline Ac_mul_Bc(mat::AbstractMatrix, rowvec::RowVector) = mat' * ctranspose(rowvec)

Ac_mul_B(::RowVector, ::AbstractVector) = error("Cannot multiply two vectors")
Ac_mul_B(rowvec::RowVector, mat::AbstractMatrix) = error("Cannot left-multiply matrix by vector")
@inline Ac_mul_B(vec::AbstractVector, mat::AbstractMatrix) = ctranspose(Ac_mul_B(mat,vec))
@inline Ac_mul_B(rowvec1::RowVector, rowvec2::RowVector) = kron(ctranspose(rowvec1), rowvec2)
Ac_mul_B(vec::AbstractVector, rowvec::RowVector) = error("Cannot multiply two transposed vectors")
@inline Ac_mul_B(vec1::AbstractVector, vec2::AbstractVector) = reduce(+, map(Ac_mul_B, vec1, vec2))
Ac_mul_B(mat::AbstractMatrix, rowvec::RowVector) = error("Cannot right-multiply matrix by transposed vector")

## Left Division ##

\(mat::AbstractMatrix, rowvec::RowVector) = error("Cannot left-divide transposed vector by matrix")
At_ldiv_B(mat::AbstractMatrix, rowvec::RowVector) = error("Cannot left-divide transposed vector by matrix")
Ac_ldiv_B(mat::AbstractMatrix, rowvec::RowVector) = error("Cannot left-divide transposed vector by matrix")

## Right Division ##

@inline /(rowvec::RowVector, mat::AbstractMatrix) = transpose(transpose(mat) \ transpose(rowvec))
@inline A_rdiv_Bt(rowvec::RowVector, mat::AbstractMatrix) = transpose(mat \ transpose(rowvec))
@inline A_rdiv_Bc(rowvec::RowVector, mat::AbstractMatrix) = ctranspose(mat  \ ctranspose(rowvec))
