# This file is a part of Julia. License is MIT: https://julialang.org/license

## sizedmemory.jl: Statically-sized managed memory

"""
    SizedMemory{T, N} <: DenseVector{T}

Fixed-size mutable array with statically-known length `N`.

Data is stored inline (not heap-allocated separately) with a maximum total
size of approximately 2KB. The length `N` is a compile-time constant encoded
in the type, enabling aggressive compiler optimizations.

# Examples
```jldoctest
julia> sm = SizedMemory{Int, 3}()
3-element SizedMemory{Int64, 3}

julia> sm[1] = 10; sm[2] = 20; sm[3] = 30;

julia> sm[1]
10

julia> length(sm)
3
```

!!! compat "Julia 1.14"
    This type requires Julia 1.14 or later.
"""
Core.SizedMemory

# Basic properties
length(sm::SizedMemory{T, N}) where {T, N} = N
size(sm::SizedMemory{T, N}) where {T, N} = (N,)
axes(sm::SizedMemory{T, N}) where {T, N} = (Base.OneTo(N),)
axes(sm::SizedMemory{T, N}, d::Integer) where {T, N} = d == 1 ? Base.OneTo(N) : Base.OneTo(1)

# Element access with proper bounds checking and inlining
@inline function getindex(sm::SizedMemory{T, N}, i::Int) where {T, N}
    @boundscheck checkbounds(sm, i)
    return ccall(:jl_sizedmemory_ref, Any, (Any, Int), sm, i - 1)::T
end

@inline function setindex!(sm::SizedMemory{T, N}, val, i::Int) where {T, N}
    @boundscheck checkbounds(sm, i)
    val_T = convert(T, val)
    ccall(:jl_sizedmemory_set, Cvoid, (Any, Int, Any), sm, i - 1, val_T)
    return val
end

# Iteration
@inline function iterate(sm::SizedMemory{T, N}, state::Int=1) where {T, N}
    state > N && return nothing
    return (@inbounds sm[state], state + 1)
end

# Type-preserving operations
similar(sm::SizedMemory{T, N}) where {T, N} = SizedMemory{T, N}()
similar(sm::SizedMemory{T, N}, ::Type{S}) where {T, N, S} = SizedMemory{S, N}()

# Copy
function copy(sm::SizedMemory{T, N}) where {T, N}
    new_sm = SizedMemory{T, N}()
    for i in 1:N
        @inbounds new_sm[i] = sm[i]
    end
    return new_sm
end

# Fill
function fill!(sm::SizedMemory{T, N}, val) where {T, N}
    val_T = convert(T, val)
    for i in 1:N
        @inbounds sm[i] = val_T
    end
    return sm
end

# Show methods for better REPL display
function show(io::IO, sm::SizedMemory{T, N}) where {T, N}
    print(io, N, "-element ", typeof(sm))
end
