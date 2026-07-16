#include "qmat.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#ifdef MYNAH_BLAS_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

int mynah_qmat_init(mynah_qmat *m, const float *w, int n, int k, int quantize) {
    m->f32 = w;
    m->q = NULL;
    m->scales = NULL;
    m->n = n;
    m->k = k;
    if (!quantize || !w) return 0;

    m->q = malloc((size_t)n * (size_t)k);
    m->scales = malloc((size_t)n * sizeof(float));
    if (!m->q || !m->scales) { mynah_qmat_free(m); return -1; }

    for (int i = 0; i < n; i++) {
        const float *row = w + (size_t)i * (size_t)k;
        float amax = 0.0f;
        for (int j = 0; j < k; j++) {
            const float a = fabsf(row[j]);
            if (a > amax) amax = a;
        }
        const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
        m->scales[i] = scale;
        const float inv = 1.0f / scale;
        int8_t *qrow = m->q + (size_t)i * (size_t)k;
        for (int j = 0; j < k; j++) {
            const float v = row[j] * inv;
            qrow[j] = (int8_t)(v >= 0.0f ? v + 0.5f : v - 0.5f);
        }
    }

    /* i f32 quantizzati non servono più: rilascia le pagine mmap interamente
     * possedute dal tensore (clean, file-backed: l'OS può rileggerle se serve).
     * macOS ignora MADV_DONTNEED sui file-backed: serve MADV_FREE_REUSABLE. */
    const long pg = sysconf(_SC_PAGESIZE);
    uintptr_t lo = ((uintptr_t)w + (uintptr_t)pg - 1) & ~((uintptr_t)pg - 1);
    uintptr_t hi = ((uintptr_t)w + (size_t)n * (size_t)k * 4u) & ~((uintptr_t)pg - 1);
    if (hi > lo) {
#ifdef MADV_FREE_REUSABLE
        if (madvise((void *)lo, hi - lo, MADV_FREE_REUSABLE) != 0)
#endif
            madvise((void *)lo, hi - lo, MADV_DONTNEED);
    }
    return 0;
}

void mynah_qmat_free(mynah_qmat *m) {
    free(m->q);
    free(m->scales);
    m->q = NULL;
    m->scales = NULL;
}

size_t mynah_qmat_bytes(const mynah_qmat *m) {
    return m->q ? (size_t)m->n * (size_t)m->k + (size_t)m->n * 4u
                : (size_t)m->n * (size_t)m->k * 4u;
}

/* soglia righe: sotto -> dot int8 diretto (bandwidth-bound), sopra -> dequant+GEMM */
#define QMAT_SMALL_T 16

void mynah_qmat_mul(const mynah_qmat *m, const float *x, float *out, int T) {
    if (!m->q) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, m->n, m->k,
                    1.0f, x, m->k, m->f32, m->k, 0.0f, out, m->n);
        return;
    }
    if (T <= QMAT_SMALL_T) {
        for (int t = 0; t < T; t++) {
            const float *xr = x + (size_t)t * (size_t)m->k;
            float *o = out + (size_t)t * (size_t)m->n;
            for (int i = 0; i < m->n; i++) {
                const int8_t *qrow = m->q + (size_t)i * (size_t)m->k;
                float acc = 0.0f;
                for (int j = 0; j < m->k; j++) acc += xr[j] * (float)qrow[j];
                o[i] = acc * m->scales[i];
            }
        }
        return;
    }
    /* dequant per-chiamata + GEMM: l'overhead (ms) si ammortizza sul T grande */
    float *wd = malloc((size_t)m->n * (size_t)m->k * sizeof(float));
    if (!wd) return;
    for (int i = 0; i < m->n; i++) {
        const int8_t *qrow = m->q + (size_t)i * (size_t)m->k;
        float *dr = wd + (size_t)i * (size_t)m->k;
        const float s = m->scales[i];
        for (int j = 0; j < m->k; j++) dr[j] = (float)qrow[j] * s;
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, m->n, m->k,
                1.0f, x, m->k, wd, m->k, 0.0f, out, m->n);
    free(wd);
}
