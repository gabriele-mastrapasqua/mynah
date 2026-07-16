#include "stwrite.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char name[192], dtype[8];
    int64_t shape[4];
    int n_dims;
    const void *data;
    size_t bytes, offset;
} stw_entry;

struct mynah_stw {
    stw_entry *entries;
    size_t n, cap, total;
};

mynah_stw *mynah_stw_new(void) { return calloc(1, sizeof(mynah_stw)); }

int mynah_stw_add(mynah_stw *w, const char *name, const char *dtype,
                  const int64_t *shape, int n_dims, const void *data, size_t bytes) {
    if (w->n == w->cap) {
        w->cap = w->cap ? w->cap * 2 : 64;
        stw_entry *ne = realloc(w->entries, w->cap * sizeof(stw_entry));
        if (!ne) return -1;
        w->entries = ne;
    }
    stw_entry *e = &w->entries[w->n++];
    memset(e, 0, sizeof(*e));
    snprintf(e->name, sizeof(e->name), "%s", name);
    snprintf(e->dtype, sizeof(e->dtype), "%s", dtype);
    e->n_dims = n_dims > 4 ? 4 : n_dims;
    for (int i = 0; i < e->n_dims; i++) e->shape[i] = shape[i];
    e->data = data;
    e->bytes = bytes;
    e->offset = w->total;
    w->total += bytes;
    return 0;
}

int mynah_stw_write(mynah_stw *w, const char *path) {
    /* header JSON costruito a mano: nomi controllati (tensori HF), niente escaping
     * esotico da gestire */
    size_t hcap = 256 + w->n * 256;
    char *hdr = malloc(hcap);
    if (!hdr) return -1;
    size_t hl = 0;
    hdr[hl++] = '{';
    for (size_t i = 0; i < w->n; i++) {
        const stw_entry *e = &w->entries[i];
        hl += (size_t)snprintf(hdr + hl, hcap - hl,
                               "%s\"%s\":{\"dtype\":\"%s\",\"shape\":[", i ? "," : "",
                               e->name, e->dtype);
        for (int d = 0; d < e->n_dims; d++)
            hl += (size_t)snprintf(hdr + hl, hcap - hl, "%s%lld", d ? "," : "",
                                   (long long)e->shape[d]);
        hl += (size_t)snprintf(hdr + hl, hcap - hl, "],\"data_offsets\":[%zu,%zu]}",
                               e->offset, e->offset + e->bytes);
    }
    hdr[hl++] = '}';
    /* pad a multipli di 8 con spazi (convenzione safetensors) */
    while (hl % 8 != 0) hdr[hl++] = ' ';

    FILE *f = fopen(path, "wb");
    if (!f) { free(hdr); return -1; }
    uint64_t hlen = (uint64_t)hl;
    int ok = fwrite(&hlen, 8, 1, f) == 1 && fwrite(hdr, 1, hl, f) == hl;
    for (size_t i = 0; ok && i < w->n; i++)
        ok = fwrite(w->entries[i].data, 1, w->entries[i].bytes, f) == w->entries[i].bytes;
    free(hdr);
    return fclose(f) == 0 && ok ? 0 : -1;
}

void mynah_stw_free(mynah_stw *w) {
    if (!w) return;
    free(w->entries);
    free(w);
}
