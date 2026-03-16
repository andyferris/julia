# This file is a part of Julia. License is MIT: https://julialang.org/license

length(::SizedMemory{T,n}) where {T,n} = n
size(::SizedMemory{T,n}) where {T,n} = (n,)
sizeof(m::SizedMemory) = Core.sizeof(m)

IndexStyle(::Type{<:SizedMemory}) = IndexLinear()

function getindex(m::SizedMemory{T,n}, i::Int) where {T,n}
    @_noub_if_noinbounds_meta
    @boundscheck checkbounds(m, i)
    ccall(:jl_sizedmemoryget, Any, (Any, Csize_t), m, i - 1)::T
end

function setindex!(m::SizedMemory{T,n}, v, i::Int) where {T,n}
    @_noub_if_noinbounds_meta
    val = v isa T ? v : convert(T, v)::T
    @boundscheck checkbounds(m, i)
    ccall(:jl_sizedmemoryset, Cvoid, (Any, Any, Csize_t), m, val, i - 1)
    m
end

function isassigned(m::SizedMemory, i::Int)
    @boundscheck checkbounds(m, i)
    ccall(:jl_sizedmemory_isassigned, Cint, (Any, Csize_t), m, i - 1) != 0
end

function copy(a::SizedMemory{T,n}) where {T,n}
    b = SizedMemory{T,n}()
    @inbounds for i in 1:n
        if isassigned(a, i)
            b[i] = a[i]
        end
    end
    b
end

similar(::SizedMemory{T,n}) where {T,n} = SizedMemory{T,n}()
similar(::SizedMemory, ::Type{S}, dims::Dims{1}) where {S} = Memory{S}(undef, dims[1])
