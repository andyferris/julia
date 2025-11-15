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

    // Check for zero-length singleton
    if (dt->instance != NULL)
        return (jl_sizedmemory_t*)dt->instance;

    // Get allocation size from layout (includes header + data)
    size_t allocsize = jl_datatype_size(dt);

    jl_task_t *ct = jl_current_task;
    jl_sizedmemory_t *sm = (jl_sizedmemory_t*)jl_gc_alloc(
        ct->ptls,
        allocsize,
        smtype
    );

    // Zero-initialize data portion if needed
    if (dt->zeroinit) {
        char *data = (char*)jl_sizedmemory_data(sm);
        size_t datasize = allocsize - sizeof(jl_sizedmemory_t);
        memset(data, 0, datasize);
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
        if (!jl_is_datatype(eltype))
            jl_error("jl_sizedmemory_ref: element type must be a concrete datatype");

        jl_datatype_t *eldt = (jl_datatype_t*)eltype;
        const jl_datatype_layout_t *el_layout = eldt->layout;
        size_t elsz = el_layout->size;

        char *elem_ptr = data + i * elsz;
        return jl_new_bits(eltype, elem_ptr);
    }
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
        if (!jl_is_datatype(eltype))
            jl_error("jl_sizedmemory_set: element type must be a concrete datatype");

        jl_datatype_t *eldt = (jl_datatype_t*)eltype;
        const jl_datatype_layout_t *el_layout = eldt->layout;
        size_t elsz = el_layout->size;

        char *elem_ptr = data + i * elsz;

        // Check if element type has pointers
        int hasptr = el_layout->first_ptr >= 0;

        // Copy data safely
        size_t fsz = jl_datatype_size((jl_datatype_t*)jl_typeof(val));
        memassign_safe(hasptr, elem_ptr, val, fsz);

        // Write barrier if needed
        if (hasptr)
            jl_gc_wb(sm, val);
    }
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
