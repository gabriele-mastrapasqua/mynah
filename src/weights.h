/* Loader safetensors: mmap del file + indice nome -> tensore.
 * I nomi dei tensori sono quelli HF verbatim (decisione da docs/prior-art.md). */
#ifndef MYNAH_WEIGHTS_H
#define MYNAH_WEIGHTS_H

#include <stddef.h>
#include <stdint.h>

typedef enum { MYNAH_DT_F32, MYNAH_DT_F64, MYNAH_DT_BF16, MYNAH_DT_F16, MYNAH_DT_I8, MYNAH_DT_U8 } mynah_dtype;

typedef struct {
    const char *name;      /* punta dentro l'header JSON (vive quanto il file) */
    mynah_dtype dtype;
    int n_dims;
    int64_t shape[8];
    const void *data;      /* mmap, read-only */
    size_t n_elems;
} mynah_tensor;

typedef struct mynah_safetensors mynah_safetensors;

mynah_safetensors *mynah_st_open(const char *path);
mynah_safetensors *mynah_st_open_quiet(const char *path); /* niente errore se assente */
void mynah_st_close(mynah_safetensors *st);

/* Lookup per nome esatto. NULL se assente. */
const mynah_tensor *mynah_st_get(const mynah_safetensors *st, const char *name);
size_t mynah_st_count(const mynah_safetensors *st);
const mynah_tensor *mynah_st_at(const mynah_safetensors *st, size_t i);

#endif
