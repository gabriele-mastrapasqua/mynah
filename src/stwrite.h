/* Writer safetensors minimale (header JSON + dati contigui).
 * Usato da `mynah quantize` per i checkpoint pre-quantizzati. */
#ifndef MYNAH_STWRITE_H
#define MYNAH_STWRITE_H

#include <stddef.h>
#include <stdint.h>

typedef struct mynah_stw mynah_stw;

mynah_stw *mynah_stw_new(void);
/* data deve restare valido fino a mynah_stw_write. dtype: "F32", "I8", "U8". */
int mynah_stw_add(mynah_stw *w, const char *name, const char *dtype,
                  const int64_t *shape, int n_dims, const void *data, size_t bytes);
int mynah_stw_write(mynah_stw *w, const char *path);
void mynah_stw_free(mynah_stw *w);

#endif
