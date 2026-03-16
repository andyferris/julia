// This file is a part of Julia. License is MIT: https://julialang.org/license

/*
  SizedMemory{T, n} constructors and primitives
*/
#include <stdlib.h>
#include <string.h>
#include "julia.h"
#include "julia_internal.h"
#include "julia_assert.h"

#ifdef __cplusplus
extern "C" {
#endif

// Compute the total allocation size for a SizedMemory type
static size_t jl_sizedmemory_nbytes(jl_datatype_t *ty) JL_NOTSAFEPOINT
{
    size_t n = jl_unbox_long(jl_tparam1(ty));
    const jl_datatype_layout_t *layout = ty->layout;
    size_t elsz = layout->size;
    size_t total = n * elsz;
    if (layout->flags.arrayelem_isunion)
        total += n; // union selector bytes
    return total;
}

JL_DLLEXPORT jl_value_t *jl_sizedmemory_alloc(jl_datatype_t *mtype)
{
    if (!jl_is_sizedmemory_type(mtype) || !jl_is_concrete_type((jl_value_t*)mtype) || mtype->layout == NULL) {
        jl_type_error_rt("SizedMemory", "element type", (jl_value_t*)jl_type_type,
                         jl_svec_len(mtype->parameters) >= 1 ? jl_tparam0(mtype) : (jl_value_t*)jl_nothing);
    }
    // Validate T is a type and n is a non-negative integer
    jl_value_t *eltype = jl_tparam0(mtype);
    jl_value_t *n_param = jl_tparam1(mtype);
    if (!jl_is_type(eltype) && !jl_is_typevar(eltype))
        jl_type_error_rt("SizedMemory", "element type", (jl_value_t*)jl_type_type, eltype);
    if (!jl_is_long(n_param) || jl_unbox_long(n_param) < 0)
        jl_type_error_rt("SizedMemory", "size parameter", (jl_value_t*)jl_long_type, n_param);
    if (mtype->instance != NULL)
        return mtype->instance;
    size_t total = jl_sizedmemory_nbytes(mtype);
    jl_task_t *ct = jl_current_task;
    jl_value_t *m = jl_gc_alloc(ct->ptls, total, mtype);
    if (mtype->zeroinit)
        memset((char*)m, 0, total);
    return m;
}

JL_DLLEXPORT jl_value_t *jl_sizedmemoryget(jl_value_t *m, size_t i)
{
    jl_datatype_t *ty = (jl_datatype_t*)jl_typeof(m);
    const jl_datatype_layout_t *layout = ty->layout;
    jl_value_t *eltype = jl_tparam0(ty);
    char *data = (char*)m;

    if (layout->flags.arrayelem_isboxed) {
        _Atomic(jl_value_t*) *slot = (_Atomic(jl_value_t*) *)data + i;
        jl_value_t *elt = jl_atomic_load_relaxed(slot);
        if (elt == NULL)
            jl_throw(jl_undefref_exception);
        return elt;
    }
    if (layout->flags.arrayelem_isunion) {
        assert(jl_is_uniontype(eltype));
        size_t n = jl_unbox_long(jl_tparam1(ty));
        // selector bytes stored after all element data
        uint8_t sel = ((uint8_t*)(data + n * layout->size))[i];
        eltype = jl_nth_union_component(eltype, sel);
        data = data + i * layout->size;
        if (layout->size == 0) {
            assert(jl_is_datatype_singleton((jl_datatype_t*)eltype));
            return ((jl_datatype_t*)eltype)->instance;
        }
        jl_value_t *r = jl_new_bits(eltype, data);
        return r;
    }
    if (layout->size == 0) {
        assert(jl_is_datatype_singleton((jl_datatype_t*)eltype));
        return ((jl_datatype_t*)eltype)->instance;
    }
    data = data + i * layout->size;
    jl_value_t *r = jl_new_bits(eltype, data);
    r = undefref_check((jl_datatype_t*)eltype, r);
    if (__unlikely(r == NULL))
        jl_throw(jl_undefref_exception);
    return r;
}

JL_DLLEXPORT void jl_sizedmemoryset(jl_value_t *m JL_ROOTING_ARGUMENT, jl_value_t *rhs JL_ROOTED_ARGUMENT JL_MAYBE_UNROOTED, size_t i)
{
    jl_datatype_t *ty = (jl_datatype_t*)jl_typeof(m);
    const jl_datatype_layout_t *layout = ty->layout;
    jl_value_t *eltype = jl_tparam0(ty);
    char *data = (char*)m;

    if (eltype != (jl_value_t*)jl_any_type && !jl_typeis(rhs, eltype)) {
        JL_GC_PUSH1(&rhs);
        if (!jl_isa(rhs, eltype))
            jl_type_error("sizedmemoryset!", eltype, rhs);
        JL_GC_POP();
    }

    if (layout->flags.arrayelem_isboxed) {
        jl_atomic_store_release((_Atomic(jl_value_t*) *)data + i, rhs);
        jl_gc_wb(m, rhs);
        return;
    }
    if (layout->flags.arrayelem_isunion) {
        assert(jl_is_uniontype(eltype));
        size_t n = jl_unbox_long(jl_tparam1(ty));
        uint8_t *psel = (uint8_t*)(data + n * layout->size) + i;
        unsigned nth = 0;
        if (!jl_find_union_component(eltype, jl_typeof(rhs), &nth))
            assert(0 && "invalid sizedmemoryset to isbits union");
        *psel = nth;
        data = data + i * layout->size;
        if (layout->size != 0) {
            size_t fsz = jl_datatype_size((jl_datatype_t*)jl_typeof(rhs));
            memassign_safe(0, data, rhs, fsz);
        }
        return;
    }
    if (layout->size != 0) {
        data = data + i * layout->size;
        int hasptr = layout->first_ptr >= 0;
        size_t fsz = jl_datatype_size((jl_datatype_t*)jl_typeof(rhs));
        memassign_safe(hasptr, data, rhs, fsz);
        if (hasptr)
            jl_gc_multi_wb(m, rhs);
    }
}

JL_DLLEXPORT int jl_sizedmemory_isassigned(jl_value_t *m, size_t i)
{
    jl_datatype_t *ty = (jl_datatype_t*)jl_typeof(m);
    const jl_datatype_layout_t *layout = ty->layout;
    char *data = (char*)m;

    if (layout->flags.arrayelem_isboxed) {
        _Atomic(jl_value_t*) *slot = (_Atomic(jl_value_t*) *)data + i;
        return jl_atomic_load_relaxed(slot) != NULL;
    }
    else if (layout->first_ptr >= 0) {
        char *eldata = data + i * layout->size;
        jl_value_t **firstptr = (jl_value_t**)(eldata + layout->first_ptr * sizeof(jl_value_t*));
        return *firstptr != NULL;
    }
    return 1; // isbits with no pointers are always assigned
}

#ifdef __cplusplus
}
#endif
