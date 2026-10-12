# This file is a part of Julia. License is MIT: https://julialang.org/license

## sizedmemory.jl: Statically-sized managed memory

"""
    SizedMemory{T, S} <: DenseVector{T}

Fixed-size mutable array with statically-known length `S`.

Data is stored inline (not heap-allocated separately) with a maximum total
size of approximately 2KB. The length `S` is a compile-time constant encoded
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
length(sm::SizedMemory{T, S}) where {T, S} = S
size(sm::SizedMemory{T, S}) where {T, S} = (S,)
axes(sm::SizedMemory{T, S}) where {T, S} = (Base.OneTo(S),)
axes(sm::SizedMemory{T, S}, d::Integer) where {T, S} = d == 1 ? Base.OneTo(S) : Base.OneTo(1)

# Element access with proper bounds checking and inlining
@inline function getindex(sm::SizedMemory{T, S}, i::Int) where {T, S}
    @boundscheck checkbounds(sm, i)
    # Call C function - it will throw UndefRefError for uninitialized boxed elements
    result = ccall(:jl_sizedmemory_ref, Any, (Any, Int), sm, i - 1)
    # For isbitstype, result is always valid
    # For boxed types, the C function handles UndefRefError
    # We need to ensure proper type here for inference
    return result::T
end

@inline function setindex!(sm::SizedMemory{T, S}, val, i::Int) where {T, S}
    @boundscheck checkbounds(sm, i)
    val_T = convert(T, val)
    ccall(:jl_sizedmemory_set, Cvoid, (Any, Int, Any), sm, i - 1, val_T)
    return val
end

# Check if element is assigned (for boxed types)
@inline function isassigned(sm::SizedMemory{T, S}, i::Int) where {T, S}
    @boundscheck checkbounds(sm, i)
    if isbitstype(T)
        # Bits types are always assigned
        return true
    else
        # For boxed types, check if the element is initialized
        # We need to catch UndefRefError and return false
        try
            ccall(:jl_sizedmemory_ref, Any, (Any, Int), sm, i - 1)
            return true
        catch e
            if isa(e, UndefRefError)
                return false
            else
                rethrow(e)
            end
        end
    end
end

# Iteration
@inline function iterate(sm::SizedMemory{T, S}, state::Int=1) where {T, S}
    state > S && return nothing
    return (@inbounds sm[state], state + 1)
end

# Type-preserving operations
similar(sm::SizedMemory{T, S}) where {T, S} = SizedMemory{T, S}()
similar(sm::SizedMemory{T, S}, ::Type{R}) where {T, S, R} = SizedMemory{R, S}()

# Copy
function copy(sm::SizedMemory{T, S}) where {T, S}
    new_sm = SizedMemory{T, S}()
    for i in 1:S
        # For boxed types, check if element is initialized
        if !isbitstype(T)
            try
                @inbounds new_sm[i] = sm[i]
            catch e
                if isa(e, UndefRefError)
                    # Leave uninitialized
                    continue
                else
                    rethrow(e)
                end
            end
        else
            @inbounds new_sm[i] = sm[i]
        end
    end
    return new_sm
end

# Fill
function fill!(sm::SizedMemory{T, S}, val) where {T, S}
    val_T = convert(T, val)
    for i in 1:S
        @inbounds sm[i] = val_T
    end
    return sm
end

# Show methods for better REPL display
function show(io::IO, sm::SizedMemory{T, S}) where {T, S}
    print(io, S, "-element ", typeof(sm))
end
