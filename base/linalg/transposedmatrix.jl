immutable TransposedMatrix{T, M <: AbstractMatrix} <: AbstractMatrix{T}
    parent::M
end

ConjTransposedMatrix{T, CM <: ConjMatrix} = TransposedMatrix{T, CM}

TransposedMatrix(m::AbstractMatrix) = TransposedMatrix{transpose_type(eltype(m)), typeof(m)}(m)

parent(m::TransposedMatrix) = m.parent

transpose(m::AbstractMatrix) = TransposedMatrix(m)
transpose(m::TransposedMatrix) = parent(m)

ctranspose(m::AbstractMatrix) = TransposedMatrix(conj(m))
ctranspose(m::TransposedMatrix) = conj(parent(m))

ctranspose(m::AbstractMatrix{T} where T <: Real) = TransposedMatrix(m)
ctranspose(m::TransposedMatrix{T} where T <: Real) = parent(m)

linearindexing(::TransposedMatrix) = Base.LinearSlow()
linearindexing(::Type{M} where M <: TransposedMatrix) = Base.LinearSlow()

@inline function size(m::TransposedMatrix)
    s = size(parent(m))
    return (s[2], s[1])
end

@inline function indices(m::TransposedMatrix)
    inds = indices(parent(m))
    return (inds[2], inds[1])
end

@propagate_inbounds getindex(m::TransposedMatrix, i1::Int, i2::Int, is::Int...) = parent(m)[i2, i1, is...]
@propagate_inbounds setindex!(m::TransposedMatrix, v, i1::Int, i2::Int, is::Int...) = ((parent(m))[i2, i1, is...] = v)

# TODO CartesianIndex indexing might escape the above

@inline similar{T,N}(m::TransposedMatrix, ::Type{T}, dims::Dims{N}) = similar(parent(m), T, dims)
