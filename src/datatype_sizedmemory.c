// This file is a part of Julia. License is MIT: https://julialang.org/license

// Layout computation for SizedMemory{T, S} type
// This is included into datatype.c

void jl_get_sizedmemory_layout(jl_datatype_t *st)
{
    jl_value_t *eltype = jl_tparam0(st);  // T (element type)
    jl_value_t *n_param = jl_tparam1(st); // S (size - number of elements)

    if (!st->isconcretetype) {
        // Abstract SizedMemory{T, S} with free variables - use opaque layout
        static const jl_datatype_layout_t opaque_ptr_layout = {0, 0, 1, -1, sizeof(void*), {0}};
        st->layout = &opaque_ptr_layout;
        st->has_concrete_subtype = 0;
        return;
    }

    // Extract N - must be a constant Int
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

    // Check that element type is a type
    if (!jl_is_type(eltype))
        jl_error("SizedMemory element type must be a type");

    size_t elsz = 0, al = 1;
    int isunboxed = jl_islayout_inline(eltype, &elsz, &al);

    // Compute pointer map for GC marking
    uint32_t npointers = 0;
    int32_t first_ptr = -1;
    uint32_t *pointers = &first_ptr;  // Initialize to point to first_ptr
    int zi = 0;  // zero-init flag
    int haspadding = 0;
    int isbitsegal = 1;

    // Handle boxed (non-inlineable) element types
    if (!isunboxed) {
        elsz = sizeof(void*);  // Store pointers, not data
        al = sizeof(void*);
        zi = 1;  // Zero-init (NULL pointers indicate uninitialized)

        // Boxed elements: ALL positions are pointers
        npointers = n;

        // Allocate pointer map for boxed elements
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
    else {
        // Inline (unboxed) element types
        // Align element size
        elsz = LLT_ALIGN(elsz, al);

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
                // Pointers are specified as word offsets (in units of sizeof(void*))
                size_t words_per_elem = elsz / sizeof(void*);

                // SizedMemory has no header fields - data starts at object base
                size_t header_words = 0;

                for (size_t i = 0; i < (size_t)n; i++) {
                    for (uint32_t j = 0; j < el_layout->npointers; j++) {
                        uint32_t elem_ptr_offset = jl_ptr_offset(eldt, j);
                        // Compute absolute offset from object start in words
                        pointers[i * el_layout->npointers + j] =
                            (uint32_t)(header_words + i * words_per_elem + elem_ptr_offset);
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
    }

    // Compute total data size
    size_t total_data_size = 0;
    int overflow = __builtin_mul_overflow(n, elsz, &total_data_size);
    const uint64_t max_size = (((uint64_t)1) << 31) - 1;  // 2GB limit
    if (overflow || total_data_size >= max_size)
        jl_errorf("SizedMemory size too large: %zd * %zu bytes", n, elsz);

    // Check against GC_MAX_SZCLASS limit for inline allocation
    size_t total_obj_size = sizeof(jl_sizedmemory_t) + total_data_size;
    if (total_obj_size > GC_MAX_SZCLASS)
        jl_errorf("SizedMemory size %zu bytes exceeds maximum inline size %d bytes",
                  total_obj_size, GC_MAX_SZCLASS);

    // Create layout
    // nfields=0 because SizedMemory has no named fields, just inline data
    st->layout = jl_get_layout(
        total_data_size,  // size = total inline data
        0,                // nfields = 0 (no named fields)
        npointers,
        al < sizeof(void*) ? sizeof(void*) : al,  // alignment at least pointer-aligned for GC
        haspadding,
        isbitsegal,
        0,                // arrayelem flags (not applicable)
        NULL,             // no field descriptors (nfields=0)
        pointers
    );

    // Free allocated pointer map if we used malloc
    if (pointers != &first_ptr && (size_t)(npointers * sizeof(uint32_t)) >= jl_page_size)
        free(pointers);

    st->zeroinit = zi;
    st->has_concrete_subtype = 1;
    st->isbitstype = (npointers == 0 && !zi);
}
