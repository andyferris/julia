immutable RowVector{T,V<:AbstractVector} <: AbstractMatrix{T}
    vec::V
    function RowVector(v::V)
        check_types(T,v)
        new(v)
    end
end

@inline check_types{T1,T2}(::Type{T1},::AbstractVector{T2}) = check_types(T1, T2)
@pure check_types{T1,T2}(::Type{T1},::Type{T2}) = T1 === transpose_type(T2) ? nothing : error("Element type mismatch. Tried to create a `RowVector{$T1}` from an `AbstractVector{$T2}`")

# The element type is transformed as transpose is recursive
@inline transpose_type{T}(::Type{T}) = promote_op(transpose, T)

# Constructors that take a vector
@inline RowVector{T}(vec::AbstractVector{T}) = RowVector{transpose_type(T),typeof(vec)}(vec)
@inline (::Type{RowVector{T}}){T}(vec::AbstractVector{T}) = RowVector{T,typeof(vec)}(vec)

# Constructors that take a size and default to Array
@inline (::Type{RowVector{T}}){T}(n::Int) = RowVector{T}(Vector{transpose_type(T)}(n))
@inline (::Type{RowVector{T}}){T}(n1::Int, n2::Int) = n1 == 1 ? RowVector{T}(Vector{transpose_type(T)}(n2)) : error("RowVector expects 1×N size, got ($n1,$n2)")
@inline (::Type{RowVector{T}}){T}(n::Tuple{Int}) = RowVector{T}(Vector{transpose_type(T)}(n[1]))
@inline (::Type{RowVector{T}}){T}(n::Tuple{Int,Int}) = n[1] == 1 ? RowVector{T}(Vector{transpose_type(T)}(n[2])) : error("RowVector expects 1×N size, got $n")

# similar()
@inline similar(rowvec::RowVector) = RowVector(similar(rowvec.vec))
@inline similar{T}(rowvec::RowVector, ::Type{T}) = RowVector(similar(rowvec.vec, transpose_type(T)))
# There is no resizing similar() because it would be ambiguous if the result were a Matrix or a RowVector

# Basic methods
@inline transpose(vec::AbstractVector) = RowVector(vec)
@inline ctranspose{T}(vec::AbstractVector{T}) = RowVector(conj(vec))
@inline ctranspose{T<:Real}(vec::AbstractVector{T}) = RowVector(vec)

@inline transpose(rowvec::RowVector) = rowvec.vec
@inline ctranspose{T}(rowvec::RowVector{T}) = conj(rowvec.vec)
@inline ctranspose{T<:Real}(rowvec::RowVector{T}) = rowvec.vec

# Some overloads from Base
@inline transpose(r::Range) = RowVector(r)
@inline ctranspose(r::Range) = RowVector(conj(r)) # is there such a thing as a complex range?

# Strictly, these are unnecessary but will make things stabler if we introduce
# a "view" for conj(::AbstractArray)
@inline conj(rowvec::RowVector) = RowVector(conj(rowvec.vec))
@inline conj{T<:Real}(rowvec::RowVector{T}) = rowvec

# AbstractArray interface
@inline length(rowvec::RowVector) =  length(rowvec.vec)
@inline size(rowvec::RowVector) = (1, length(rowvec.vec))
@inline size(rowvec::RowVector, d) = ifelse(d==2, length(rowvec.vec), 1)
linearindexing{V<:RowVector}(::Union{V,Type{V}}) = LinearFast()

@propagate_inbounds getindex(rowvec::RowVector, i) = transpose(rowvec.vec[i])
@propagate_inbounds setindex!(rowvec::RowVector, v, i) = setindex!(rowvec.vec, transpose(v), i)

# Cartesian indexing is distorted by getindex
# Furthermore, Cartesian indexes don't have to match shape, apparently!
@inline function getindex(rowvec::RowVector, i::CartesianIndex)
    @boundscheck if !(i.I[1] == 1 && i.I[2] ∈ indices(rowvec.vec)[1] && check_tail_indices(i.I...))
        throw(BoundsError(rowvec, i.I))
    end
    @inbounds return transpose(rowvec.vec[i.I[2]])
end
@inline function setindex!(rowvec::RowVector, v, i::CartesianIndex)
    @boundscheck if !(i.I[1] == 1 && i.I[2] ∈ indices(rowvec.vec)[1] && check_tail_indices(i.I...))
        throw(BoundsError(rowvec, i.I))
    end
    @inbounds rowvec.vec[i.I[2]] = transpose(v)
end

@propagate_inbounds getindex(rowvec::RowVector, ::CartesianIndex{0}) = getindex(rowvec)
@propagate_inbounds getindex(rowvec::RowVector, i::CartesianIndex{1}) = getindex(rowvec, i.I[1])

@propagate_inbounds setindex!(rowvec::RowVector, v, ::CartesianIndex{0}) = setindex!(rowvec, v)
@propagate_inbounds setindex!(rowvec::RowVector, v, i::CartesianIndex{1}) = setindex!(rowvec, v, i.I[1])

@inline check_tail_indices(i1, i2) = true
@inline check_tail_indices(i1, i2, i3, is...) = i3 == 1 ? check_tail_indices(i1, i2, is...) :  false

# Some conversions
convert(::Type{AbstractVector}, rowvec::RowVector) = rowvec.vec
convert{V<:AbstractVector}(::Type{V}, rowvec::RowVector) = convert(V, rowvec.vec)
convert{T,V}(::Type{RowVector{T,V}}, rowvec::RowVector) = RowVector(convert(V, rowvec.vec))

# helper function for below
@inline to_vec(rowvec::RowVector) = transpose(rowvec)
@inline to_vec(x::Number) = x
@inline to_vecs(rowvecs...) = (map(to_vec, rowvecs)...)

# map
@inline map(f, rowvecs::RowVector...) = RowVector(map(f, to_vecs(rowvecs...)...))

# broacast (other combinations default to higher-dimensional array)
@inline broadcast(f, rowvecs::Union{Number,RowVector}...) = RowVector(broadcast(f, to_vecs(rowvecs...)...))

# Horizontal concatenation #

# Empty hcat defaults to row vector
hcat() = transpose([])
@inline hcat{T}(X::T...)         = transpose(transpose_type(T)[ transpose(X[j]) for j=1:length(X) ])
@inline hcat{T<:Number}(X::T...) = transpose(T[ X[j] for j=1:length(X) ])
@inline hcat(X::Number...) = transpose(hvcat_fill(Array{promote_typeof(X...),1}(length(X)), X))
@inline hcat(X::RowVector...) = transpose(vcat(map(transpose, X)...))
@inline hcat(X::Union{RowVector,Number}...) = transpose(vcat(map(transpose, X)...))

typed_hcat{T}(::Type{T}) = transpose(typed_vcat(T))
@inline typed_hcat{T}(::Type{T}, X::Number...) = transpose(hvcat_fill(Array{T,1}(length(X)), X))
@inline typed_hcat{T}(::Type{T}, X::RowVector...) = transpose(typed_vcat(T, map(transpose, X)...))
@inline typed_hcat{T}(::Type{T}, X::Union{RowVector,Number}...) = transpose(typed_vcat(T, map(transpose, X)...))

# Multiplication #

@inline *(rowvec::RowVector, vec::AbstractVector) = reduce(+, map(At_mul_B, transpose(rowvec), vec))
@inline *(rowvec::RowVector, mat::AbstractMatrix) = transpose(mat.' * transpose(rowvec))
@inline *(vec::AbstractVector, mat::AbstractMatrix) = error("Cannot left-multiply a matrix by a vector") # Should become a deprecation
@inline *(::RowVector, ::RowVector) = error("Cannot multiply two transposed vectors")
@inline *(vec::AbstractVector, rowvec::RowVector) = kron(vec, rowvec)
@inline *(vec::AbstractVector, rowvec::AbstractVector) = error("Cannot multiply two vectors")
@inline *(mat::AbstractMatrix, rowvec::RowVector) = error("Cannot right-multiply matrix by transposed vector")

# Transposed forms
@inline A_mul_Bt(::RowVector, ::AbstractVector) = error("Cannot multiply two transposed vectors")
@inline A_mul_Bt(rowvec::RowVector, mat::AbstractMatrix) = transpose(mat * transpose(rowvec))
@inline A_mul_Bt(vec::AbstractVector, mat::AbstractMatrix) = error("Cannot left-multiply a matrix by a vector")
@inline A_mul_Bt(rowvec1::RowVector, rowvec2::RowVector) = reduce(+, map(At_mul_B, transpose(rowvec1), transpose(rowvec2)))
@inline A_mul_Bt(vec::AbstractVector, rowvec::RowVector) = error("Cannot multiply two vectors")
@inline A_mul_Bt(vec1::AbstractVector, vec2::AbstractVector) = kron(vec1, transpose(vec2))
@inline A_mul_Bt(mat::AbstractMatrix, rowvec::RowVector) = mat * transpose(rowvec)

@inline At_mul_Bt(rowvec::RowVector, vec::AbstractVector) = kron(transpose(vec), transpose(rowvec))
@inline At_mul_Bt(rowvec::RowVector, mat::AbstractMatrix) = error("Cannot left-multiply matrix by vector")
@inline At_mul_Bt(vec::AbstractVector, mat::AbstractMatrix) = transpose(mat * vec)
@inline At_mul_Bt(rowvec1::RowVector, rowvec2::RowVector) = error("Cannot multiply two vectors")
@inline At_mul_Bt(vec::AbstractVector, rowvec::RowVector) = reduce(+, map(At_mul_B, vec, transpose(rowvec)))
@inline At_mul_Bt(vec::AbstractVector, rowvec::AbstractVector) = error("Cannot multiply two transposed vectors")
@inline At_mul_Bt(mat::AbstractMatrix, rowvec::RowVector) = mat.' * transpose(rowvec)

@inline At_mul_B(::RowVector, ::AbstractVector) = error("Cannot multiply two vectors")
@inline At_mul_B(rowvec::RowVector, mat::AbstractMatrix) = error("Cannot left-multiply matrix by vector")
@inline At_mul_B(vec::AbstractVector, mat::AbstractMatrix) = transpose(At_mul_B(mat,vec))
@inline At_mul_B(rowvec1::RowVector, rowvec2::RowVector) = kron(transpose(rowvec1), rowvec2)
@inline At_mul_B(vec::AbstractVector, rowvec::RowVector) = error("Cannot multiply two transposed vectors")
@inline At_mul_B{T<:Real}(vec1::AbstractVector{T}, vec2::AbstractVector{T}) = reduce(+, map(At_mul_B, vec1, vec2)) # Seems to be overloaded...
@inline At_mul_B(vec1::AbstractVector, vec2::AbstractVector) = reduce(+, map(At_mul_B, vec1, vec2))
@inline At_mul_B(mat::AbstractMatrix, rowvec::RowVector) = error("Cannot right-multiply matrix by transposed vector")

# Conjugated forms
@inline A_mul_Bc(::RowVector, ::AbstractVector) = error("Cannot multiply two transposed vectors")
@inline A_mul_Bc(rowvec::RowVector, mat::AbstractMatrix) = ctranspose(mat * ctranspose(rowvec))
@inline A_mul_Bc(vec::AbstractVector, mat::AbstractMatrix) = error("Cannot left-multiply a matrix by a vector")
@inline A_mul_Bc(rowvec1::RowVector, rowvec2::RowVector) = reduce(+, map(A_mul_Bc, rowvec1, rowvec2))
@inline A_mul_Bc(vec::AbstractVector, rowvec::RowVector) = error("Cannot multiply two vectors")
@inline A_mul_Bc(vec1::AbstractVector, vec2::AbstractVector) = kron(vec1, ctranspose(vec2))
@inline A_mul_Bc(mat::AbstractMatrix, rowvec::RowVector) = mat * ctranspose(rowvec)

@inline Ac_mul_Bc(rowvec::RowVector, vec::AbstractVector) = kron(ctranspose(vec), ctranspose(rowvec))
@inline Ac_mul_Bc(rowvec::RowVector, mat::AbstractMatrix) = error("Cannot left-multiply matrix by vector")
@inline Ac_mul_Bc(vec::AbstractVector, mat::AbstractMatrix) = ctranspose(mat * vec)
@inline Ac_mul_Bc(rowvec1::RowVector, rowvec2::RowVector) = error("Cannot multiply two vectors")
@inline Ac_mul_Bc(vec::AbstractVector, rowvec::RowVector) = reduce(+, map(Ac_mul_Bc, vec, transpose(rowvec)))
@inline Ac_mul_Bc(vec::AbstractVector, rowvec::AbstractVector) = error("Cannot multiply two transposed vectors")
@inline Ac_mul_Bc(mat::AbstractMatrix, rowvec::RowVector) = mat' * ctranspose(rowvec)

@inline Ac_mul_B(::RowVector, ::AbstractVector) = error("Cannot multiply two vectors")
@inline Ac_mul_B(rowvec::RowVector, mat::AbstractMatrix) = error("Cannot left-multiply matrix by vector")
@inline Ac_mul_B(vec::AbstractVector, mat::AbstractMatrix) = ctranspose(Ac_mul_B(mat,vec))
@inline Ac_mul_B(rowvec1::RowVector, rowvec2::RowVector) = kron(ctranspose(rowvec1), rowvec2)
@inline Ac_mul_B(vec::AbstractVector, rowvec::RowVector) = error("Cannot multiply two transposed vectors")
@inline Ac_mul_B(vec1::AbstractVector, vec2::AbstractVector) = reduce(+, map(Ac_mul_B, vec1, vec2))
@inline Ac_mul_B(mat::AbstractMatrix, rowvec::RowVector) = error("Cannot right-multiply matrix by transposed vector")

# Left Division #

@inline \(mat::AbstractMatrix, rowvec::RowVector) = error("Cannot left-divide transposed vector by matrix")
@inline At_ldiv_B(mat::AbstractMatrix, rowvec::RowVector) = error("Cannot left-divide transposed vector by matrix")
@inline Ac_ldiv_B(mat::AbstractMatrix, rowvec::RowVector) = error("Cannot left-divide transposed vector by matrix")

# Right Division #

@inline /(rowvec::RowVector, mat::AbstractMatrix) = transpose(transpose(mat) \ transpose(rowvec))
@inline A_rdiv_Bt(rowvec::RowVector, mat::AbstractMatrix) = transpose(mat \ transpose(rowvec))
@inline A_rdiv_Bc(rowvec::RowVector, mat::AbstractMatrix) = ctranspose(mat  \ ctranspose(rowvec))
