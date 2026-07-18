#include "backend.h"
#include "qmat.h"

#include <stdio.h>
#include <math.h>
#include <string.h>

#ifdef MYNAH_BLAS_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#ifdef MYNAH_METAL
int mynah_metal_available(void);
int mynah_metal_gemm_wt(const float *x, const float *w, float *out, int T, int n, int k);
int mynah_metal_ffn_wt(const float *x, const float *w1, int n1, const float *w2, int n2,
                       float *out, int T, int k);
int mynah_metal_gemm3_wt(const float *x, const float *wa, const float *wb,
                         const float *wc, float *oa, float *ob, float *oc,
                         int T, int n, int k);
#endif
#ifdef MYNAH_CUDA
int mynah_cuda_available(void);
int mynah_cuda_gemm_wt(const float *x, const float *w, float *out, int T, int n, int k);
#endif

static int g_backend = MYNAH_BACKEND_CPU;

/* sotto questa soglia di righe la GEMM resta su CPU: il round-trip GPU non paga */
#define METAL_MIN_T 24

int mynah_set_backend(const char *name) {
    if (name && strcmp(name, "cuda") == 0) {
#ifdef MYNAH_CUDA
        if (mynah_cuda_available()) {
            g_backend = MYNAH_BACKEND_CUDA;
            fprintf(stderr, "mynah: backend CUDA attivo (GEMM grandi su GPU, T>=%d)\n",
                    METAL_MIN_T);
            return g_backend;
        }
        fprintf(stderr, "mynah: CUDA richiesto ma device non disponibile -> CPU\n");
#else
        fprintf(stderr, "mynah: CUDA richiesto ma non compilato (make cuda) -> CPU\n");
#endif
        g_backend = MYNAH_BACKEND_CPU;
        return g_backend;
    }
    if (name && strcmp(name, "metal") == 0) {
#ifdef MYNAH_METAL
        if (mynah_metal_available()) {
            g_backend = MYNAH_BACKEND_METAL;
            fprintf(stderr, "mynah: backend Metal attivo (GEMM grandi su GPU, T>=%d)\n",
                    METAL_MIN_T);
            return g_backend;
        }
        fprintf(stderr, "mynah: Metal richiesto ma device non disponibile -> CPU\n");
#else
        fprintf(stderr, "mynah: Metal richiesto ma non compilato in questa build -> CPU\n");
#endif
    }
    g_backend = MYNAH_BACKEND_CPU;
    return g_backend;
}

int mynah_backend(void) { return g_backend; }

void mynah_gemm_wt(const float *x, const float *w, float *out, int T, int n, int k) {
#ifdef MYNAH_METAL
    if (g_backend == MYNAH_BACKEND_METAL &&
        mynah_metal_gemm_wt(x, w, out, T, n, k) == 0)
        return;
#endif
#ifdef MYNAH_CUDA
    if (g_backend == MYNAH_BACKEND_CUDA && T >= METAL_MIN_T &&
        mynah_cuda_gemm_wt(x, w, out, T, n, k) == 0)
        return;
#endif
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, n, k,
                1.0f, x, k, w, k, 0.0f, out, n);
}

void mynah_ffn_wt(const float *x, const float *w1, int n1, const float *w2, int n2,
                  float *out, int T, int k, float *scratch) {
#ifdef MYNAH_METAL
    if (g_backend == MYNAH_BACKEND_METAL &&
        mynah_metal_ffn_wt(x, w1, n1, w2, n2, out, T, k) == 0)
        return;
#endif
    mynah_gemm_wt(x, w1, scratch, T, n1, k);
    const size_t nmid = (size_t)T * (size_t)n1;
    for (size_t i = 0; i < nmid; i++)
        scratch[i] = scratch[i] * mynah_sigmoid(scratch[i]);
    mynah_gemm_wt(scratch, w2, out, T, n2, n1);
}

void mynah_gemm3_wt(const float *x, const float *wa, const float *wb, const float *wc,
                    float *oa, float *ob, float *oc, int T, int n, int k) {
#ifdef MYNAH_METAL
    if (g_backend == MYNAH_BACKEND_METAL &&
        mynah_metal_gemm3_wt(x, wa, wb, wc, oa, ob, oc, T, n, k) == 0)
        return;
#endif
    mynah_gemm_wt(x, wa, oa, T, n, k);
    mynah_gemm_wt(x, wb, ob, T, n, k);
    mynah_gemm_wt(x, wc, oc, T, n, k);
}
