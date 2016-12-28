# This file is a part of Julia. License is MIT: http://julialang.org/license

## Array conjugation ##

"""
    ConjArray(array)

A lazy-view wrapper of an `AbstractArray`, taking the elementwise complex
conjugate. This type is usually constructed (and unwrapped) via the `conj()`
function (or related `ctranspose()`).
"""
immutable ConjArray{T, N, A <: AbstractArray} <: AbstractArray{T, N}
    parent::A
end

@inline ConjArray{T,N}(a::AbstractArray{T,N}) = ConjArray{conj_type(T), N, typeof(a)}(a)

@inline conj_type(x) = conj_type(typeof(x))
@inline conj_type{T}(::Type{T}) = promote_op(conj, T)

@inline parent_type(c::ConjArray) = parent_type(typeof(c))
@inline parent_type{T,N,A}(::Type{ConjArray{T,N,A}}) = A

@inline size(a::ConjArray) = size(a.parent)
linearindexing{CA <: ConjArray}(::Union{CA,Type{CA}}) = linearindexing(parent_type(CA))

@propagate_inbounds getindex{T,N}(a::ConjArray{T,N}, i::Int) = conj(getindex(a.parent, i))
@propagate_inbounds getindex{T,N}(a::ConjArray{T,N}, i::Vararg{Int,N}) = conj(getindex(a.parent, i...))
@propagate_inbounds setindex!{T,N}(a::ConjArray{T,N}, v, i::Int) = setindex!(a.parent, conj(v), i)
@propagate_inbounds setindex!{T,N}(a::ConjArray{T,N}, v, i::Vararg{Int,N}) = setindex!(a.parent, conj(v), i...)

@inline conj(a::AbstractArray) = ConjArray(a)
@inline conj{T<:Real}(a::AbstractArray{T}) = a
@inline conj(a::ConjArray) = a.parent
@inline conj{T<:Real}(a::ConjArray{T}) = a.parent  # disambiguate (usually shouldn't happen)

## Matrix transposition ##

transpose(a::AbstractArray) = error("transpose not defined for $(typeof(a)). Consider using `permutedims` for higher-dimensional arrays.")

"""
    TransposedMatrix(matrix)

A lazy-view wrapper of an `AbstractMatrix`, which reverses the dimensions of the
input `matrix` (and presents a view of the transpose of it's elements, as
transposition applies recursively). This type is usually constructed (and
unwrapped) via the `transpose()` function (or related `ctranspose()`).
"""
immutable TransposedMatrix{T, M <: AbstractMatrix} <: AbstractMatrix{T}
    parent::M
end

@inline TransposedMatrix(m::AbstractMatrix) = TransposedMatrix{transpose_type(eltype(m)), typeof(m)}(m)

@inline transpose_type(x) = conj_type(typeof(x))
@inline transpose_type{T}(::Type{T}) = promote_op(conj, T)

@inline parent_type(m::TransposedMatrix) = parent_type(typeof(m))
@inline parent_type{T,M}(::Type{TransposedMatrix{T,M}}) = M

@inline size(m::TransposedMatrix) = (tmp = size(m.parent); return (tmp[2], tmp[1]))
linearindexing{TM <: TransposedMatrix}(::Union{TM,Type{TM}}) = LinearSlow()

@propagate_inbounds getindex(m::TransposedMatrix, i...) = transpose(getindex(m.parent, i...))
@propagate_inbounds setindex!(m::TransposedMatrix, v, i...) = setindex!(m.parent, transpose(v), i...)


"""
    transpose(A)

The transposition operator (`.'`). Creates a transposed "view" of `A` (use
`copy(transpose(A))` to allocate a new matrix).

# Example

```jldoctest
julia> A = [1 2 3; 4 5 6; 7 8 9]
3×3 Array{Int64,2}:
 1  2  3
 4  5  6
 7  8  9

julia> transpose(A)
3×3 TransposedMatrix{Int64,Array{Int64,2}}:
 1  4  7
 2  5  8
 3  6  9
```
"""
@inline transpose(m::AbstractMatrix) = TransposedMatrix(m)
@inline transpose(m::TransposedMatrix) = m.parent

function copy{T,M}(m::TransposedMatrix{T,M})
    m_out = similar(m.parent, eltype(m), indices(m))
    transpose!(m_out, m)
end


# Define how conj and transpose nest
@inline conj(m::TransposedMatrix) = TransposedMatrix(conj(m.parent))

## Mutating implementations of matrix transposition ##

"""
    transpose!(dest,src)

Transpose array `src` and store the result in the preallocated array `dest`, which should
have a size corresponding to `(size(src,2),size(src,1))`. No in-place transposition is
supported and unexpected results will happen if `src` and `dest` have overlapping memory
regions.
"""
transpose!(B::AbstractMatrix, A::AbstractMatrix) = transpose_f!(transpose, B, A)

"""
    ctranspose!(dest,src)

Conjugate transpose array `src` and store the result in the preallocated array `dest`, which
should have a size corresponding to `(size(src,2),size(src,1))`. No in-place transposition
is supported and unexpected results will happen if `src` and `dest` have overlapping memory
regions.
"""
ctranspose!(B::AbstractMatrix, A::AbstractMatrix) = transpose_f!(ctranspose, B, A)

const transposebaselength=64
function transpose_f!(f,B::AbstractMatrix,A::AbstractMatrix)
    inds = indices(A)
    indices(B,1) == inds[2] && indices(B,2) == inds[1] || throw(DimensionMismatch(string(f)))

    m, n = length(inds[1]), length(inds[2])
    if m*n<=4*transposebaselength
        @inbounds begin
            for j = inds[2]
                for i = inds[1]
                    B[j,i] = f(A[i,j])
                end
            end
        end
    else
        transposeblock!(f,B,A,m,n,first(inds[1])-1,first(inds[2])-1)
    end
    return B
end
function transposeblock!(f,B::AbstractMatrix,A::AbstractMatrix,m::Int,n::Int,offseti::Int,offsetj::Int)
    if m*n<=transposebaselength
        @inbounds begin
            for j = offsetj+(1:n)
                for i = offseti+(1:m)
                    B[j,i] = f(A[i,j])
                end
            end
        end
    elseif m>n
        newm=m>>1
        transposeblock!(f,B,A,newm,n,offseti,offsetj)
        transposeblock!(f,B,A,m-newm,n,offseti+newm,offsetj)
    else
        newn=n>>1
        transposeblock!(f,B,A,m,newn,offseti,offsetj)
        transposeblock!(f,B,A,m,n-newn,offseti,offsetj+newn)
    end
    return B
end

function ccopy!(B, A)
    RB, RA = eachindex(B), eachindex(A)
    if RB == RA
        for i = RB
            B[i] = ctranspose(A[i])
        end
    else
        for (i,j) = zip(RB, RA)
            B[i] = ctranspose(A[j])
        end
    end
end

function copy_transpose!{R,S}(B::AbstractVecOrMat{R}, ir_dest::Range{Int}, jr_dest::Range{Int},
                              A::AbstractVecOrMat{S}, ir_src::Range{Int}, jr_src::Range{Int})
    if length(ir_dest) != length(jr_src)
        throw(ArgumentError(string("source and destination must have same size (got ",
                                   length(jr_src)," and ",length(ir_dest),")")))
    end
    if length(jr_dest) != length(ir_src)
        throw(ArgumentError(string("source and destination must have same size (got ",
                                   length(ir_src)," and ",length(jr_dest),")")))
    end
    @boundscheck checkbounds(B, ir_dest, jr_dest)
    @boundscheck checkbounds(A, ir_src, jr_src)
    idest = first(ir_dest)
    for jsrc in jr_src
        jdest = first(jr_dest)
        for isrc in ir_src
            B[idest,jdest] = A[isrc,jsrc]
            jdest += step(jr_dest)
        end
        idest += step(ir_dest)
    end
    return B
end


## Row vectors ##
"""
    RowVector(vector)

A lazy-view wrapper of an `AbstractVector`, which turns a length-`n` vector into
a `1×n` shaped row vector and represents the transpose of a vector (the elements
are also transposed recursively). This type is usually constructed (and
unwrapped) via the `transpose()` function (or related `ctranspose()`).

By convention, a vector can be multiplied by a matrix on it's left (`A * v`)
whereas a row vector can be multiplied by a matrix on it's right (such that
`v.' * A = (A.' * v).'`). It differs from a `1×n`-sized matrix by the facts that
its transpose returns a vector and the inner product `v1.' * v2` returns a
scalar, but will otherwise behave similarly.
"""
immutable RowVector{T,V<:AbstractVector} <: AbstractMatrix{T}
    parent::V
    function RowVector(v::V)
        check_types(T,v)
        new(v)
    end
end

@inline check_types{T1,T2}(::Type{T1},::AbstractVector{T2}) = check_types(T1, T2)
@pure check_types{T1,T2}(::Type{T1},::Type{T2}) = T1 === transpose_type(T2) ? nothing : error("Element type mismatch. Tried to create a `RowVector{$T1}` from an `AbstractVector{$T2}`")

# Constructors that take a vector (the element type is transformed as transpose is recursive)
@inline RowVector{T}(vec::AbstractVector{T}) = RowVector{transpose_type(T),typeof(vec)}(vec)
@inline (::Type{RowVector{T}}){T}(vec::AbstractVector{T}) = RowVector{T,typeof(vec)}(vec)

# Constructors that take a size and default to Vector
@inline (::Type{RowVector{T}}){T}(n::Int) = RowVector{T}(Vector{transpose_type(T)}(n))
@inline (::Type{RowVector{T}}){T}(n1::Int, n2::Int) = n1 == 1 ? RowVector{T}(Vector{transpose_type(T)}(n2)) : error("RowVector expects 1×N size, got ($n1,$n2)")
@inline (::Type{RowVector{T}}){T}(n::Tuple{Int}) = RowVector{T}(Vector{transpose_type(T)}(n[1]))
@inline (::Type{RowVector{T}}){T}(n::Tuple{Int,Int}) = n[1] == 1 ? RowVector{T}(Vector{transpose_type(T)}(n[2])) : error("RowVector expects 1×N size, got $n")

# similar()
@inline similar(rowvec::RowVector) = RowVector(similar(rowvec.parent))
@inline similar{T}(rowvec::RowVector, ::Type{T}) = RowVector(similar(rowvec.parent, transpose_type(T)))
# There is no resizing similar() because it would be ambiguous if the result were a Matrix or a RowVector

# Basic methods
@inline transpose(vec::AbstractVector) = RowVector(vec)
@inline ctranspose(vec::AbstractVector) = RowVector(conj(vec))

@inline transpose(rowvec::RowVector) = rowvec.parent
@inline ctranspose(rowvec::RowVector) = conj(rowvec.parent)

# Define how conj and transpose nest
@inline conj(rowvec::RowVector) = RowVector(conj(rowvec.parent))

# AbstractArray interface
@inline length(rowvec::RowVector) =  length(rowvec.parent)
@inline size(rowvec::RowVector) = (1, length(rowvec.parent))
@inline size(rowvec::RowVector, d) = ifelse(d==2, length(rowvec.parent), 1)
linearindexing{V<:RowVector}(::Union{V,Type{V}}) = LinearFast()

@propagate_inbounds getindex(rowvec::RowVector, i) = transpose(rowvec.parent[i])
@propagate_inbounds setindex!(rowvec::RowVector, v, i) = setindex!(rowvec.parent, transpose(v), i)

# Cartesian indexing is distorted by getindex
# Furthermore, Cartesian indexes don't have to match shape, apparently!
@inline function getindex(rowvec::RowVector, i::CartesianIndex)
    @boundscheck if !(i.I[1] == 1 && i.I[2] ∈ indices(rowvec.parent)[1] && check_tail_indices(i.I...))
        throw(BoundsError(rowvec, i.I))
    end
    @inbounds return transpose(rowvec.parent[i.I[2]])
end
@inline function setindex!(rowvec::RowVector, v, i::CartesianIndex)
    @boundscheck if !(i.I[1] == 1 && i.I[2] ∈ indices(rowvec.parent)[1] && check_tail_indices(i.I...))
        throw(BoundsError(rowvec, i.I))
    end
    @inbounds rowvec.parent[i.I[2]] = transpose(v)
end

@propagate_inbounds getindex(rowvec::RowVector, ::CartesianIndex{0}) = getindex(rowvec)
@propagate_inbounds getindex(rowvec::RowVector, i::CartesianIndex{1}) = getindex(rowvec, i.I[1])

@propagate_inbounds setindex!(rowvec::RowVector, v, ::CartesianIndex{0}) = setindex!(rowvec, v)
@propagate_inbounds setindex!(rowvec::RowVector, v, i::CartesianIndex{1}) = setindex!(rowvec, v, i.I[1])

@inline check_tail_indices(i1, i2) = true
@inline check_tail_indices(i1, i2, i3, is...) = i3 == 1 ? check_tail_indices(i1, i2, is...) :  false

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
