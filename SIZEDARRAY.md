# SizedMemory{T, n} Implementation Plan

## Overview

`SizedMemory{T, n}` is a new Core type that provides statically-sized, mutable arrays where:
- `n` is a compile-time constant `Int` (type parameter)
- `T` can be any type (doesn't need to be `isbitstype`)
- Data is stored inline in the object
- Size is limited to `GC_MAX_SZCLASS` (~2KB) for MVP

**Goal**: Replace `StaticArrays.MArray{(n,), T}` with a robust Core type.

## Design Decisions

1. **Size Source**: Type parameter (not runtime field) - true compile-time size
2. **Storage**: Always inline (error if total size > GC_MAX_SZCLASS)
3. **Mutability**: Mutable (reference type) for MVP
4. **Type Parameters**: `SizedMemory{T, n}` where `n::Int`
5. **Layout**: Opaque layout with fixed size per type instantiation

## Implementation Steps

---

### Step 1: Add C Type Definition ✅ COMPLETED

**Status**: Successfully compiled! All C type definitions added and Julia builds without errors.

**Changes Made**:
- Added `jl_sizedmemory_t` typedef to `src/julia.h:200`
- Added helper macros to `src/julia.h:1449-1455`
- Added exports to `src/jl_exported_data.inc:86-87`
- Added global defines to `src/jl_data_globals_defs.inc:82-83`

**File**: `src/julia.h` (around line 200, after `jl_genericmemory_t`)

```c
// SizedMemory{T, N}: fixed-size, statically-known memory block
// Data is stored inline immediately after this header
// Size and layout determined by type parameters at instantiation
typedef struct _jl_sizedmemory_t {
    JL_DATA_TYPE
    // No fields - data stored inline after this struct
    // Length is encoded in the type, not stored at runtime
} jl_sizedmemory_t;
```

**Add global declarations** (around line 1900):
```c
extern JL_DLLIMPORT jl_typename_t *jl_sizedmemory_typename JL_GLOBALLY_ROOTED;
extern JL_DLLIMPORT jl_unionall_t *jl_sizedmemory_type JL_GLOBALLY_ROOTED;
```

**Add helper macros** (around line 1430):
```c
// Get length from SizedMemory type parameter
#define jl_sizedmemory_length(smtype) (jl_unbox_long(jl_tparam1(smtype)))
// Get element type from SizedMemory type parameter
#define jl_sizedmemory_eltype(smtype) (jl_tparam0(smtype))
// Get pointer to data
#define jl_sizedmemory_data(sm) ((void*)((char*)(sm) + sizeof(jl_sizedmemory_t)))
```

---

### Step 2: Add Type System Integration ✅ COMPLETED

**Status**: Successfully compiled! SizedMemory type registered in Core type system.

**Changes Made**:
- Added SizedMemory type creation in `src/jltypes.c:3384-3393`
- Type parameters: `SizedMemory{T, N}` where T is element type, N is count
- Registered as mutable type inheriting from Any
- Type globals `jl_sizedmemory_typename` and `jl_sizedmemory_type` initialized

**File**: `src/jltypes.c` (around line 3380, after GenericMemory setup)

```c
// SizedMemory{T, N} - statically sized memory with N elements of type T
tv = jl_svec2(tvar("T"), tvar("N"));
jl_datatype_t *sizedmemory_datatype =
    jl_new_datatype(jl_symbol("SizedMemory"), core, jl_any_type, tv,
                    jl_emptysvec,  // No named fields (opaque inline storage)
                    jl_emptysvec,  // No field types
                    jl_emptysvec, 0, 1, 0);  // abstract=0, mutabl=1, ninitialized=0
jl_sizedmemory_typename = sizedmemory_datatype->name;
jl_sizedmemory_type = (jl_unionall_t*)jl_sizedmemory_typename->wrapper;
```

**File**: `src/julia_internal.h` (add declaration around line 800)

```c
void jl_get_sizedmemory_layout(jl_datatype_t *st);
```

**File**: `src/gf.c` (add to exports, around line 2300)

```c
jl_sizedmemory_typename = NULL;
jl_sizedmemory_type = NULL;
```

---

### Step 3: Add Layout Computation ✅ COMPLETED

**Status**: Successfully compiled! Layout computation handles alignment, GC requirements, and pointer tracking.

**Changes Made**:
- Created `src/datatype_sizedmemory.c` with layout computation function
- Included in `src/datatype.c:632`
- Hooked into `jl_compute_field_offsets` at `src/datatype.c:649-652`
- Handles zero-length arrays (singleton creation)
- Validates size constraints (<= GC_MAX_SZCLASS)
- Computes pointer maps for GC marking when elements contain pointers
- Proper alignment for GC heap-allocated objects

**Key Design Points**:
- Data stored inline immediately after `jl_sizedmemory_t` header
- Pointer offsets computed as words from object start (including header)
- Uses `nfields=1` to indicate inline (non-opaque) layout
- Alignment at least `sizeof(void*)` for GC compatibility
- Replicates element pointer pattern for each array position

**File**: `src/datatype.c` (around line 645, in `jl_compute_field_offsets`)

```c
if (st->name == jl_sizedmemory_typename) {
    jl_get_sizedmemory_layout(st);
    return;
}
```

**Add new function** (around line 630, after `jl_get_genericmemory_layout`):

```c
void jl_get_sizedmemory_layout(jl_datatype_t *st)
{
    jl_value_t *eltype = jl_tparam0(st);  // T
    jl_value_t *n_param = jl_tparam1(st); // n

    if (!st->isconcretetype) {
        // Abstract SizedMemory{T, n} with free variables - use opaque layout
        static const jl_datatype_layout_t opaque_ptr_layout = {0, 0, 1, -1, sizeof(void*), {0}};
        st->layout = &opaque_ptr_layout;
        st->has_concrete_subtype = 0;
        return;
    }

    // Extract n - must be a constant Int
    if (!jl_is_long(n_param))
        jl_error("SizedMemory size parameter must be a constant Int");
    ssize_t n = jl_unbox_long(n_param);

    if (n < 0)
        jl_errorf("SizedMemory size must be non-negative, got %zd", n);

    // Handle zero-length case
    if (n == 0) {
        static const jl_datatype_layout_t zero_layout = {0, 1, 0, -1, 1, {.isbitsegal=1}};
        st->layout = &zero_layout;
        st->zeroinit = 0;
        st->has_concrete_subtype = 1;

        // Create singleton zero-length instance
        jl_task_t *ct = jl_current_task;
        jl_sizedmemory_t *zeroinst = (jl_sizedmemory_t*)jl_gc_permobj(
            ct->ptls, sizeof(jl_sizedmemory_t), st, 0);
        assert(!st->instance);
        st->instance = (jl_value_t*)zeroinst;
        return;
    }

    // Check that element type is inlineable
    if (!jl_is_type(eltype))
        jl_error("SizedMemory element type must be a type");

    size_t elsz = 0, al = 1;
    int isunboxed = jl_islayout_inline(eltype, &elsz, &al);

    if (!isunboxed)
        jl_error("SizedMemory element type must be inlineable (for now)");

    // Align element size
    elsz = LLT_ALIGN(elsz, al);

    // Compute total data size
    size_t total_data_size = 0;
    int overflow = __builtin_mul_overflow(n, elsz, &total_data_size);
    if (overflow || total_data_size >= MAXINTVAL)
        jl_errorf("SizedMemory size too large: %zd * %zu bytes", n, elsz);

    // Check against GC_MAX_SZCLASS limit for inline allocation
    size_t total_obj_size = sizeof(jl_sizedmemory_t) + total_data_size;
    if (total_obj_size > GC_MAX_SZCLASS)
        jl_errorf("SizedMemory size %zu bytes exceeds maximum inline size %d bytes",
                  total_obj_size, GC_MAX_SZCLASS);

    // Compute pointer map for GC marking
    uint32_t npointers = 0;
    int32_t first_ptr = -1;
    uint32_t *pointers = &first_ptr;
    int zi = 0;  // zero-init flag
    int haspadding = 0;
    int isbitsegal = 1;

    if (jl_is_datatype(eltype)) {
        jl_datatype_t *eldt = (jl_datatype_t*)eltype;
        const jl_datatype_layout_t *el_layout = eldt->layout;
        zi = eldt->zeroinit;

        if (el_layout->first_ptr >= 0) {
            // Element type contains pointers - need to replicate pattern for each element
            npointers = n * el_layout->npointers;

            // Allocate pointer map
            size_t ptr_map_size = npointers * sizeof(uint32_t);
            int should_malloc = ptr_map_size >= jl_page_size;
            if (should_malloc)
                pointers = (uint32_t*)malloc_s(ptr_map_size);
            else
                pointers = (uint32_t*)alloca(ptr_map_size);

            // Fill pointer map - replicate element pattern for each array element
            size_t words_per_elem = elsz / sizeof(void*);
            for (size_t i = 0; i < (size_t)n; i++) {
                for (int j = 0; j < el_layout->npointers; j++) {
                    uint32_t elem_ptr_offset = jl_ptr_offset(eldt, j);
                    pointers[i * el_layout->npointers + j] =
                        (uint32_t)(i * words_per_elem + elem_ptr_offset);
                }
            }
            first_ptr = pointers[0];
            isbitsegal = 0;  // Contains pointers, not bits-comparable
            haspadding = el_layout->flags.haspadding;
        }
        else {
            // Pure bits type
            isbitsegal = el_layout->flags.isbitsegal;
            haspadding = el_layout->flags.haspadding;
        }
    }
    else if (jl_is_uniontype(eltype)) {
        // Union types not supported yet
        jl_error("SizedMemory does not yet support Union element types");
    }

    // Create layout
    // Use nfields=1 to indicate inline storage (not opaque variable-size)
    st->layout = jl_get_layout(
        total_data_size,  // size = total data size (not per-element)
        1,                // nfields = 1 (inline storage)
        npointers,
        al,
        haspadding,
        isbitsegal,
        0,                // arrayelem flags (not applicable)
        NULL,             // no field descriptors
        pointers
    );

    if (pointers != &first_ptr && (size_t)(npointers * sizeof(uint32_t)) >= jl_page_size)
        free(pointers);

    st->zeroinit = zi;
    st->has_concrete_subtype = 1;
    st->isbitstype = (npointers == 0 && !zi);
}
```

---

### Step 4: Add GC Support ✅ COMPLETED

**File**: `src/gc-stock.c` (around line 2450, in `gc_mark_outrefs`)

Add after the GenericMemory case:

```c
else if (vt->name == jl_sizedmemory_typename) {
    // SizedMemory has fixed inline size determined by type
    // Use layout-based marking for interior pointers
    const jl_datatype_layout_t *layout = vt->layout;
    gc_setmark_pool(ptls, o, bits);

    // Mark interior pointers if present
    if (layout->first_ptr >= 0) {
        char *data = (char*)new_obj;
        uint32_t npointers = layout->npointers;

        if (layout->fielddesc_type == 3) {
            // Foreign type - shouldn't happen for SizedMemory
            // Fall through to generic marking
        }
        else {
            // Use pre-computed pointer map
            const void *ptrs = jl_dt_layout_ptrs(layout);
            for (uint32_t i = 0; i < npointers; i++) {
                uint32_t fld_offset;
                if (layout->fielddesc_type == 0) {
                    fld_offset = ((const uint8_t*)ptrs)[i];
                }
                else if (layout->fielddesc_type == 1) {
                    fld_offset = ((const uint16_t*)ptrs)[i];
                }
                else {
                    assert(layout->fielddesc_type == 2);
                    fld_offset = ((const uint32_t*)ptrs)[i];
                }

                jl_value_t **slot = &((jl_value_t**)data)[fld_offset];
                jl_value_t *fld = jl_atomic_load_relaxed((_Atomic(jl_value_t*)*)slot);

                if (fld != NULL) {
                    verify_parent2("SizedMemory", new_obj, slot, "elem(%d)", i);
                    gc_try_claim_and_push(mq, fld, &nptr);
                }
            }
        }
    }
    // If no pointers (isbits), nothing to mark
    goto ret;
}
```

**File**: `src/gc-mmtk.c` (around line 642, similar addition needed for MMTk GC)

```c
if (vt->name == jl_sizedmemory_typename) {
    // SizedMemory inline storage
    // Scan based on layout pointer map
    mmtk_scan_memory(object_reference, vt);
}
```

---

### Step 5: Add Allocation and Accessor Functions ✅ COMPLETED

**File**: `src/sizedmemory.c` (new file)

```c
// This file is a part of Julia. License is MIT: https://julialang.org/license

/*
  SizedMemory{T, N} constructors and primitives
  Fixed-size arrays with compile-time length
*/
#include <stdlib.h>
#include <string.h>
#include "julia.h"
#include "julia_internal.h"
#include "julia_assert.h"

#ifdef __cplusplus
extern "C" {
#endif

// Allocate a new SizedMemory of the given type
// The type must be a concrete SizedMemory{T, N} type
JL_DLLEXPORT jl_sizedmemory_t *jl_alloc_sizedmemory(jl_value_t *smtype)
{
    assert(jl_is_datatype(smtype));
    jl_datatype_t *dt = (jl_datatype_t*)smtype;

    if (dt->name != jl_sizedmemory_typename)
        jl_error("jl_alloc_sizedmemory: type must be a SizedMemory type");

    if (!dt->isconcretetype)
        jl_error("jl_alloc_sizedmemory: type must be concrete");

    // Get allocation size from layout
    size_t allocsize = jl_datatype_size(dt);

    jl_task_t *ct = jl_current_task;
    jl_sizedmemory_t *sm = (jl_sizedmemory_t*)jl_gc_alloc(
        ct->ptls,
        allocsize,
        smtype
    );

    // Zero-initialize if needed
    if (dt->zeroinit) {
        memset(jl_sizedmemory_data(sm), 0, allocsize);
    }

    return sm;
}

// Get element at index i (0-based)
JL_DLLEXPORT jl_value_t *jl_sizedmemory_ref(jl_sizedmemory_t *sm, size_t i)
{
    jl_datatype_t *dt = (jl_datatype_t*)jl_typetagof(sm);
    jl_value_t *eltype = jl_tparam0(dt);
    size_t n = jl_unbox_long(jl_tparam1(dt));

    if (i >= n)
        jl_bounds_error_int((jl_value_t*)sm, i + 1);  // Julia uses 1-based indexing

    const jl_datatype_layout_t *layout = dt->layout;
    size_t elsz = jl_datatype_size((jl_datatype_t*)eltype);

    char *data = (char*)jl_sizedmemory_data(sm);
    char *elem_ptr = data + i * elsz;

    // If element type is boxed or has pointers, return the value
    if (jl_is_datatype(eltype)) {
        jl_datatype_t *eldt = (jl_datatype_t*)eltype;
        if (eldt->layout->first_ptr >= 0) {
            // Contains pointers - need to box
            return jl_new_bits(eltype, elem_ptr);
        }
    }

    // Pure bits type
    return jl_new_bits(eltype, elem_ptr);
}

// Set element at index i (0-based)
JL_DLLEXPORT void jl_sizedmemory_set(jl_sizedmemory_t *sm, size_t i, jl_value_t *val)
{
    jl_datatype_t *dt = (jl_datatype_t*)jl_typetagof(sm);
    jl_value_t *eltype = jl_tparam0(dt);
    size_t n = jl_unbox_long(jl_tparam1(dt));

    if (i >= n)
        jl_bounds_error_int((jl_value_t*)sm, i + 1);  // Julia uses 1-based indexing

    // Type check
    if (eltype != (jl_value_t*)jl_any_type && !jl_typeis(val, eltype)) {
        if (!jl_isa(val, eltype))
            jl_type_error("jl_sizedmemory_set", eltype, val);
    }

    size_t elsz = jl_datatype_size((jl_datatype_t*)eltype);
    char *data = (char*)jl_sizedmemory_data(sm);
    char *elem_ptr = data + i * elsz;

    // Check if element type has pointers
    int hasptr = 0;
    if (jl_is_datatype(eltype)) {
        jl_datatype_t *eldt = (jl_datatype_t*)eltype;
        hasptr = eldt->layout->first_ptr >= 0;
    }

    // Copy data
    size_t fsz = jl_datatype_size((jl_datatype_t*)jl_typeof(val));
    memassign_safe(hasptr, elem_ptr, val, fsz);

    // Write barrier if needed
    if (hasptr)
        jl_gc_wb(sm, val);
}

// Get length from SizedMemory type
JL_DLLEXPORT size_t jl_sizedmemory_len(jl_value_t *smtype)
{
    assert(jl_is_datatype(smtype));
    jl_datatype_t *dt = (jl_datatype_t*)smtype;

    if (dt->name != jl_sizedmemory_typename)
        jl_error("jl_sizedmemory_len: type must be a SizedMemory type");

    return jl_unbox_long(jl_tparam1(dt));
}

#ifdef __cplusplus
}
#endif
```

**Update build system**: Add to `src/Makefile` (around line 50):

```makefile
SRCS := \
	... \
	genericmemory \
	sizedmemory \
	...
```

---

### Step 6: Add Julia Core Interface ✅ COMPLETED

**File**: `base/boot.jl` (around line 450, after Array constructors)

Add Core interface for SizedMemory:

```julia
# SizedMemory{T, N} - statically-sized mutable array
# Exported from Core for early use, full Base interface comes later

# Constructor
function SizedMemory{T, N}() where {T, N}
    ccall(:jl_alloc_sizedmemory, Ref{SizedMemory{T, N}}, (Any,), SizedMemory{T, N})
end

# Basic accessors
function length(sm::SizedMemory{T, N}) where {T, N}
    # Length is compile-time constant from type parameter
    return N::Int
end

function size(sm::SizedMemory{T, N}) where {T, N}
    (N,)
end

# Bounds checking helper
@inline function checkbounds(::Type{Bool}, sm::SizedMemory{T, N}, i::Int) where {T, N}
    return 1 <= i <= N
end

# getindex - 1-based indexing
@inline function getindex(sm::SizedMemory{T, N}, i::Int) where {T, N}
    @boundscheck checkbounds(Bool, sm, i) || throw(BoundsError(sm, i))
    # Convert to 0-based for C call
    return ccall(:jl_sizedmemory_ref, Any, (Any, Csize_t), sm, i - 1)::T
end

# setindex! - 1-based indexing
@inline function setindex!(sm::SizedMemory{T, N}, val, i::Int) where {T, N}
    @boundscheck checkbounds(Bool, sm, i) || throw(BoundsError(sm, i))
    # Type assertion for better inference
    val_T = convert(T, val)
    # Convert to 0-based for C call
    ccall(:jl_sizedmemory_set, Cvoid, (Any, Csize_t, Any), sm, i - 1, val_T)
    return val
end

# Iteration support
function iterate(sm::SizedMemory{T, N}, state::Int=1) where {T, N}
    state > N && return nothing
    return (sm[state], state + 1)
end

# Similar for type stability
function similar(sm::SizedMemory{T, N}) where {T, N}
    SizedMemory{T, N}()
end

function similar(sm::SizedMemory{T, N}, ::Type{S}) where {T, N, S}
    SizedMemory{S, N}()
end
```

**File**: `base/builtins.c` (around line 2630, in `jl_init_primitives_1`)

Add to exported builtins:

```c
// SizedMemory primitives
add_builtin("SizedMemory", (jl_value_t*)jl_sizedmemory_type);
```

**Considerations**:

1. **Type parameter `N` must be known at compile time** for `length()` to be constant-folded
2. **`@inline` annotations** are critical for performance
3. **Bounds checking** uses `@boundscheck` for conditional compilation
4. **Type conversion** in `setindex!` ensures type stability
5. **C calls use 0-based indexing**, Julia uses 1-based

**Optional builtins** (if needed for better codegen):

Add to `src/builtins.c`:

```c
// Get compile-time length of SizedMemory type
JL_CALLABLE(jl_f_sizedmemory_length)
{
    JL_NARGS(sizedmemory_length, 1, 1);
    JL_TYPECHK(sizedmemory_length, datatype, args[0]);
    jl_datatype_t *dt = (jl_datatype_t*)args[0];
    if (dt->name != jl_sizedmemory_typename)
        jl_error("sizedmemory_length: argument must be a SizedMemory type");
    return jl_tparam1(dt);  // Return N as Int
}
```

Then expose in boot.jl if better codegen is needed:

```julia
# Alternative implementation using builtin
length(::Type{SizedMemory{T, N}}) where {T, N} = N
length(sm::SizedMemory{T, N}) where {T, N} = N  # Inferred as constant
```

---

### Step 7: Type Inference and Constant Propagation

**Goal**: Ensure `length(sm)` and `getindex(sm, i)` have optimal inference and codegen.

**File**: `base/compiler/tfuncs.jl` (around line 2400, near Array inference rules)

Add inference rules for SizedMemory:

```julia
# SizedMemory length is always a constant from the type parameter
function return_type_tfunc(interp::AbstractInterpreter, argtypes::Vector{Any}, si::StmtInfo, sv::AbsIntState)
    # ... existing code ...

    # Special case for length(::SizedMemory{T, N})
    if length(argtypes) == 2 && argtypes[1] === typeof(length)
        at = widenconst(argtypes[2])
        if at <: SizedMemory
            # Extract N from SizedMemory{T, N}
            if isa(at, DataType) && length(at.parameters) >= 2
                N = at.parameters[2]
                if isa(N, Int)
                    return Const(N)
                end
            end
        end
    end

    # ... rest of function ...
end
```

**However**, this might not be needed! The current approach should work:

1. **Type parameter `N` is part of the type**: `SizedMemory{Float64, 3}`
2. **`length()` returns `N` directly**: `length(sm::SizedMemory{T, N}) where {T, N} = N::Int`
3. **Type inference should see**:
   - Input type: `SizedMemory{Float64, 3}`
   - Type parameters: `T = Float64, N = 3`
   - Return value: `N::Int` → constant `3`

**Verification Steps**:

1. **Test constant folding**:
   ```julia
   function test_const_prop()
       sm = SizedMemory{Float64, 3}()
       n = length(sm)
       # Check if n is inferred as Const(3)
       return n + 1
   end

   # Should show Const(4) in inference
   @code_warntype test_const_prop()
   ```

2. **Test bounds check elimination**:
   ```julia
   function test_bounds_elim(sm::SizedMemory{Float64, 3})
       # This should eliminate bounds check via constant propagation
       return sm[1] + sm[2] + sm[3]
   end

   # Should show no bounds checks in LLVM IR
   @code_llvm test_bounds_elim(SizedMemory{Float64, 3}())
   ```

3. **Test loop unrolling**:
   ```julia
   function sum_sizedmem(sm::SizedMemory{Float64, N}) where {N}
       s = 0.0
       for i in 1:length(sm)
           s += sm[i]
       end
       return s
   end

   # Should unroll for small N
   @code_llvm sum_sizedmem(SizedMemory{Float64, 3}())
   ```

**If inference needs help**, add to `base/compiler/abstractinterpretation.jl`:

```julia
# Constant prop for SizedMemory operations
function abstract_call_known(interp::AbstractInterpreter, @nospecialize(f),
                            fargs::Union{Nothing,Vector{Any}}, argtypes::Argtypes,
                            sv::InferenceState, max_methods::Int = InferenceParams(interp).MAX_METHODS)
    # ... existing code ...

    # Special handling for length(::SizedMemory)
    if f === length && length(argtypes) == 2
        at = argtypes[2]
        if isa(at, Const)
            val = at.val
            if isa(val, SizedMemory)
                smtype = typeof(val)
                if isa(smtype, DataType) && length(smtype.parameters) >= 2
                    N = smtype.parameters[2]
                    if isa(N, Int)
                        return CallMeta(Const(N), EFFECTS_TOTAL, NoCallInfo())
                    end
                end
            end
        elseif isa(at, DataType) && at.name === SizedMemory.name
            if length(at.parameters) >= 2
                N = at.parameters[2]
                if isa(N, Int)
                    return CallMeta(Const(N), EFFECTS_TOTAL, NoCallInfo())
                end
            end
        end
    end

    # ... rest of function ...
end
```

**Likely Outcome**: No changes needed! The type system should handle this automatically because:
- `N` is a type parameter, visible to the compiler
- `length()` just returns `N`, which is known statically
- The compiler's existing constant propagation should handle it

**Validation Plan**:

1. Implement Steps 1-6
2. Write simple tests
3. Check `@code_warntype` and `@code_llvm` output
4. Only add Step 7 inference rules if needed

---

## Testing Checklist

**Basic Functionality**:
- [ ] Can construct `SizedMemory{Int, 3}()`
- [ ] Can construct `SizedMemory{Float64, 10}()`
- [ ] `length()` returns correct value
- [ ] `getindex` works with valid indices
- [ ] `setindex!` works and updates values
- [ ] Bounds checking throws errors

**Type System**:
- [ ] Zero-length `SizedMemory{T, 0}` works
- [ ] Large sizes (near `GC_MAX_SZCLASS`) work
- [ ] Error for sizes > `GC_MAX_SZCLASS`
- [ ] Error for non-Int `N` parameter
- [ ] Error for negative `N`

**GC Correctness**:
- [ ] `SizedMemory{Any, N}` (boxed elements)
- [ ] `SizedMemory{Vector{Int}, N}` (elements with pointers)
- [ ] No memory leaks under GC pressure
- [ ] Proper marking during GC cycles

**Performance**:
- [ ] `length()` constant-folds
- [ ] Bounds checks eliminated for constant indices
- [ ] Small loops unroll
- [ ] Comparable performance to `MArray` for isbits types

**Edge Cases**:
- [ ] Mixed types: `SizedMemory{Union{Int, Float64}, 3}` (should error for now)
- [ ] Nested: `SizedMemory{SizedMemory{Int, 2}, 3}`
- [ ] Abstract element type (should error if not concrete)

---

## Future Enhancements (Post-MVP)

1. **Support for Union element types** with selector bytes
2. **Immutable variant** (`SizedArray{T, N}`) that can live on stack
3. **Sizes > GC_MAX_SZCLASS** with malloc'd storage
4. **Multi-dimensional** `SizedArray{T, (M, N)}`
5. **Full Base interface**: broadcasting, linear algebra, etc.
6. **Compiler optimizations**: Stack allocation, SIMD
7. **Atomic operations** support

---

## Dependencies

**Build order**:
1. julia.h changes
2. jltypes.c (type creation)
3. datatype.c (layout computation)
4. sizedmemory.c (allocation and accessors)
5. gc-stock.c and gc-mmtk.c (GC support)
6. boot.jl (Julia interface)
7. Makefile updates

**Testing requirements**:
- GC test suite
- Type inference tests
- Performance benchmarks
- Comparison with StaticArrays.jl

**Documentation**:
- Docstrings in boot.jl
- CLAUDE.md entry for contributors
- Future: Manual entry when stabilized

---

## Notes

- **Alignment**: Element alignment handled automatically via `jl_islayout_inline`
- **Zero-init**: Follows element type's `zeroinit` flag
- **Type parameters**: Must be `Int`, not `Integer` subtypes
- **GC_MAX_SZCLASS**: Currently ~2032 bytes on 64-bit systems
- **Pointer map**: Pre-computed at type instantiation, reused for all instances
- **Performance**: Should match or exceed `StaticArrays.MArray` for isbits types

---

## Implementation Timeline Estimate

- Step 1: 30 minutes (C type definition)
- Step 2: 1 hour (type system integration)
- Step 3: 2-3 hours (layout computation with all edge cases)
- Step 4: 2 hours (GC support for both gc-stock and gc-mmtk)
- Step 5: 1-2 hours (allocation and accessors)
- Step 6: 1 hour (Julia Core interface)
- Step 7: 0-2 hours (inference rules, if needed)
- Testing: 2-4 hours (comprehensive tests)

**Total**: ~10-15 hours for complete MVP implementation

---

## Priority 2: Non-isbitstype T Support

### Current Blocker

**File**: `src/datatype_sizedmemory.c:43-51`

```c
size_t elsz = 0, al = 1;
int isunboxed = jl_islayout_inline(eltype, &elsz, &al);

if (!isunboxed)
    jl_error("SizedMemory element type must be inlineable (for now)");
```

This check prevents using types like `Any`, `String`, `Vector{Int}`, or any non-isbitstype as elements.

### How GenericMemory Handles Boxed Elements

**Key Insight**: For non-isbitstype T (boxed elements):
- Elements are stored as **pointers** (type `jl_value_t*`)
- Element size is **always** `sizeof(void*)` (8 bytes on 64-bit)
- Storage contains pointers to heap-allocated objects
- All N positions are pointers (simple pointer map: [0, 1, 2, ..., N-1])
- Uses `arrayelem_isboxed` flag to distinguish boxed vs inline

**GenericMemory Layout** (from `src/datatype.c:590-611`):
```c
else {
    elsz = sizeof(void*);  // Boxed: store pointers, not actual data
    al = elsz;
    zi = 1;  // Zero-init (NULL pointers)
}

int arrayelem = 0;
if (!isunboxed)
    arrayelem |= 1;  // Set arrayelem_isboxed flag

st->layout = jl_get_layout(elsz, nfields, npointers, al, haspadding,
                           isbitsegal, arrayelem, NULL, pointers);
```

**GenericMemory Accessors** (from `src/genericmemory.c`):

**Get** (lines 332-348):
```c
static jl_value_t *jl_ptrmemrefget(jl_genericmemoryref_t m, int isatomic) {
    assert(layout->flags.arrayelem_isboxed);
    _Atomic(jl_value_t*) *ptr = (_Atomic(jl_value_t*)*)m.ptr_or_offset;
    jl_value_t *elt = isatomic ? jl_atomic_load(ptr) : jl_atomic_load_relaxed(ptr);
    if (elt == NULL)
        jl_throw(jl_undefref_exception);  // Uninitialized
    return elt;  // Return the heap object, not the pointer
}

JL_DLLEXPORT jl_value_t *jl_memoryrefget(jl_genericmemoryref_t m, int isatomic) {
    const jl_datatype_layout_t *layout = ((jl_datatype_t*)jl_typetagof(m.mem))->layout;
    if (layout->flags.arrayelem_isboxed)
        return jl_ptrmemrefget(m, isatomic);  // Dereference pointer
    // ... inline case ...
}
```

**Set** (lines 420-428):
```c
if (layout->flags.arrayelem_isboxed) {
    assert((char*)m.ptr_or_offset - (char*)m.mem->ptr < sizeof(jl_value_t*) * m.mem->length);
    if (isatomic)
        jl_atomic_store((_Atomic(jl_value_t*)*)m.ptr_or_offset, rhs);
    else
        jl_atomic_store_release((_Atomic(jl_value_t*)*)m.ptr_or_offset, rhs);
    jl_gc_wb(jl_genericmemory_owner(m.mem), rhs);  // Write barrier for GC
    return;
}
```

### Required Changes for SizedMemory

#### 1. Layout Computation (`src/datatype_sizedmemory.c`)

**Change lines 43-68** from:
```c
size_t elsz = 0, al = 1;
int isunboxed = jl_islayout_inline(eltype, &elsz, &al);

if (!isunboxed)
    jl_error("SizedMemory element type must be inlineable (for now)");

// Align element size
elsz = LLT_ALIGN(elsz, al);

// Compute total data size
size_t total_data_size = 0;
int overflow = __builtin_mul_overflow(n, elsz, &total_data_size);
```

**To**:
```c
size_t elsz = 0, al = 1;
int isunboxed = jl_islayout_inline(eltype, &elsz, &al);

// Handle boxed (non-inlineable) element types
if (!isunboxed) {
    elsz = sizeof(void*);  // Store pointers, not data
    al = sizeof(void*);
    zi = 1;  // Zero-init (NULL pointers indicate uninitialized)
}
else {
    // Align element size for inline storage
    elsz = LLT_ALIGN(elsz, al);
}

// Compute total data size
size_t total_data_size = 0;
int overflow = __builtin_mul_overflow(n, elsz, &total_data_size);
```

**Change pointer map computation (lines 77-118)** to handle boxed case:
```c
if (!isunboxed) {
    // Boxed elements: ALL positions are pointers
    npointers = n;

    // Allocate pointer map
    size_t ptr_map_size = npointers * sizeof(uint32_t);
    int should_malloc = ptr_map_size >= jl_page_size;
    if (should_malloc)
        pointers = (uint32_t*)malloc_s(ptr_map_size);
    else
        pointers = (uint32_t*)alloca(ptr_map_size);

    // Simple sequential map: [0, 1, 2, ..., N-1] in word offsets
    size_t header_words = 0;  // SizedMemory has no header
    for (size_t i = 0; i < (size_t)n; i++) {
        pointers[i] = (uint32_t)(header_words + i);
    }
    first_ptr = pointers[0];
    isbitsegal = 0;  // Pointers are not bits-comparable
    haspadding = 0;  // Pointers have no padding
}
else if (jl_is_datatype(eltype)) {
    // Inline elements with potential pointers inside
    // ... existing code ...
}
```

**Note**: We don't need to set `arrayelem_isboxed` flag because SizedMemory doesn't use it. The type itself knows whether T is boxed, and we can check at accessor time.

#### 2. Accessor Functions (`src/sizedmemory.c`)

**Update `jl_sizedmemory_ref` (lines 391-417)** to handle boxed elements:

```c
JL_DLLEXPORT jl_value_t *jl_sizedmemory_ref(jl_sizedmemory_t *sm, size_t i)
{
    jl_datatype_t *dt = (jl_datatype_t*)jl_typetagof(sm);
    jl_value_t *eltype = jl_tparam0(dt);
    size_t n = jl_unbox_long(jl_tparam1(dt));

    if (i >= n)
        jl_bounds_error_int((jl_value_t*)sm, i + 1);

    // Check if element type is boxed
    size_t elsz_check = 0, al_check = 1;
    int isunboxed = jl_islayout_inline(eltype, &elsz_check, &al_check);

    char *data = (char*)jl_sizedmemory_data(sm);

    if (!isunboxed) {
        // Boxed elements: dereference pointer
        jl_value_t **elem_ptr = (jl_value_t**)(data + i * sizeof(void*));
        jl_value_t *elt = jl_atomic_load_relaxed((_Atomic(jl_value_t*)*)elem_ptr);
        if (elt == NULL)
            jl_throw(jl_undefref_exception);  // Uninitialized
        return elt;
    }
    else {
        // Inline elements: copy bits to new object
        size_t elsz = jl_datatype_size((jl_datatype_t*)eltype);
        char *elem_ptr = data + i * elsz;
        return jl_new_bits(eltype, elem_ptr);
    }
}
```

**Update `jl_sizedmemory_set` (lines 420-453)** to handle boxed elements:

```c
JL_DLLEXPORT void jl_sizedmemory_set(jl_sizedmemory_t *sm, size_t i, jl_value_t *val)
{
    jl_datatype_t *dt = (jl_datatype_t*)jl_typetagof(sm);
    jl_value_t *eltype = jl_tparam0(dt);
    size_t n = jl_unbox_long(jl_tparam1(dt));

    if (i >= n)
        jl_bounds_error_int((jl_value_t*)sm, i + 1);

    // Type check
    if (eltype != (jl_value_t*)jl_any_type && !jl_typeis(val, eltype)) {
        if (!jl_isa(val, eltype))
            jl_type_error("jl_sizedmemory_set", eltype, val);
    }

    // Check if element type is boxed
    size_t elsz_check = 0, al_check = 1;
    int isunboxed = jl_islayout_inline(eltype, &elsz_check, &al_check);

    char *data = (char*)jl_sizedmemory_data(sm);

    if (!isunboxed) {
        // Boxed elements: store pointer with write barrier
        jl_value_t **elem_ptr = (jl_value_t**)(data + i * sizeof(void*));
        jl_atomic_store_release((_Atomic(jl_value_t*)*)elem_ptr, val);
        jl_gc_wb(sm, val);  // Write barrier for GC
    }
    else {
        // Inline elements: copy bits
        size_t elsz = jl_datatype_size((jl_datatype_t*)jl_typeof(val));
        char *elem_ptr = data + i * elsz;

        // Check if element type has pointers
        int hasptr = 0;
        if (jl_is_datatype(eltype)) {
            jl_datatype_t *eldt = (jl_datatype_t*)eltype;
            hasptr = eldt->layout->first_ptr >= 0;
        }

        // Copy data safely
        memassign_safe(hasptr, elem_ptr, val, elsz);

        // Write barrier if needed
        if (hasptr)
            jl_gc_wb(sm, val);
    }
}
```

#### 3. GC Support (Already Complete!)

The existing GC code should work because:
- **Stock GC** (`src/gc-stock.c:2453-2509`): Uses pre-computed pointer maps
  - For boxed elements: pointer map = [0, 1, 2, ..., N-1]
  - Correctly marks all pointer positions
- **MMTk GC** (`src/gc-mmtk.c:654-658`): Handles inline data
  - Copies entire object including inline pointers

**No changes needed** to GC code!

#### 4. Allocation (Already Complete!)

The existing `jl_alloc_sizedmemory` already handles zero-init correctly:
- For boxed types: `zi = 1`, so storage is zeroed (NULL pointers)
- For inline types: follows element type's `zeroinit` flag

**No changes needed** to allocation code!

### Testing Plan for Non-isbitstype Support

Once changes are implemented, test with:

```julia
# Test boxed elements
sm_any = Core.SizedMemory{Any, 3}()
sm_any[1] = "hello"
sm_any[2] = 42
sm_any[3] = [1, 2, 3]
println(sm_any[1], " ", sm_any[2], " ", sm_any[3])

# Test uninitialized access
sm_str = Core.SizedMemory{String, 3}()
try
    sm_str[1]  # Should throw UndefRefError
    println("ERROR: Should have thrown!")
catch e
    println("Caught expected error: ", typeof(e))
end

# Test GC integration
sm_vec = Core.SizedMemory{Vector{Int}, 2}()
sm_vec[1] = [1, 2, 3]
sm_vec[2] = [4, 5, 6]
GC.gc()  # Force GC
println("After GC: ", sm_vec[1], " ", sm_vec[2])

# Test nested SizedMemory
sm_nested = Core.SizedMemory{Core.SizedMemory{Int, 2}, 3}()
sm_nested[1] = Core.SizedMemory{Int, 2}()
sm_nested[1][1] = 10
sm_nested[1][2] = 20
println("Nested: ", sm_nested[1][1], " ", sm_nested[1][2])
```

### Summary of Changes

**What's blocking non-isbitstype T**: Single check at `datatype_sizedmemory.c:50`

**Required changes**:
1. Remove error check for `!isunboxed`
2. Handle boxed case in layout: `elsz = sizeof(void*)`, simple pointer map
3. Update ref: dereference pointer if boxed, check NULL
4. Update set: store pointer + write barrier if boxed

**No changes needed**:
- GC marking (already uses pointer maps correctly)
- Allocation (already handles zero-init)
- Julia interface (ccalls are type-agnostic)

**Estimated effort**: 1-2 hours to implement + 1 hour testing = ~3 hours total

**Risk level**: Low - pattern is well-established in GenericMemory, just needs adaptation
