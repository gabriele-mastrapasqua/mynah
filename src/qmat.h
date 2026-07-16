/* Matrice pesi con quantizzazione INT8 opzionale (per-riga simmetrica, al load).
 * Politica da prior-art (parakeet.cpp/qwen-tts): si quantizzano SOLO i grandi
 * linear consumati dalle GEMM; conv 2D, LSTM, norm, bias, embedding restano f32.
 *
 * Dispatch del prodotto:
 *  - f32: cblas_sgemm (invariato)
 *  - int8, T piccolo (streaming/decode): kernel dot diretto — legge 4x meno byte,
 *    vince dove il matvec è bandwidth-bound
 *  - int8, T grande (offline/batch): dequant in scratch + sgemm (compute-bound) */
#ifndef MYNAH_QMAT_H
#define MYNAH_QMAT_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    const float *f32;   /* pesi originali [n, k] (mmap) — usati se q == NULL */
    int8_t *q;          /* quantizzati [n, k], NULL se non quantizzata */
    float *scales;      /* [n] scale per riga */
    int n, k;
} mynah_qmat;

/* Inizializza da pesi f32 row-major [n, k]. quantize != 0 => costruisce l'int8. */
int mynah_qmat_init(mynah_qmat *m, const float *w, int n, int k, int quantize);
void mynah_qmat_free(mynah_qmat *m);

/* out[T, n] = x[T, k] @ W^T (layout linear PyTorch). */
void mynah_qmat_mul(const mynah_qmat *m, const float *x, float *out, int T);

/* Bytes residenti dei pesi di questa matrice (per report). */
size_t mynah_qmat_bytes(const mynah_qmat *m);

#endif
