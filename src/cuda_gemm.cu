/* Backend CUDA per le GEMM grandi — pattern da qwen-tts (qwen_tts_cuda.c):
 * WEIGHT CACHE RESIDENTE (ogni pointer di peso caricato sul device UNA volta,
 * cache per-chiave-pointer), buffer I/O device riusabili, handle cuBLAS proprio.
 *
 * ⚠️ CROSS-COMPILED, NON ANCORA VALIDATO SU HARDWARE (stesso approccio del VNNI
 * di qwen-tts): compilare con `make cuda` su Linux+CUDA e validare con
 * `make test` prima di fidarsi. Fallback CPU automatico su ogni errore.
 */
#ifdef MYNAH_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static cublasHandle_t g_blas;
static cudaStream_t g_stream;
static int g_ready = -1;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    const void *host_ptr;
    float *dev;
} wc_ent;
static wc_ent *g_wc;
static int g_wc_n, g_wc_cap;

static float *g_in, *g_out;
static size_t g_in_cap, g_out_cap;

extern "C" int mynah_cuda_available(void) {
    pthread_mutex_lock(&g_mu);
    if (g_ready < 0) {
        int n = 0;
        g_ready = cudaGetDeviceCount(&n) == cudaSuccess && n > 0 &&
                  cublasCreate(&g_blas) == CUBLAS_STATUS_SUCCESS &&
                  cudaStreamCreate(&g_stream) == cudaSuccess;
        if (g_ready) cublasSetStream(g_blas, g_stream);
    }
    pthread_mutex_unlock(&g_mu);
    return g_ready == 1;
}

static float *weight_dev(const void *w, size_t bytes) {
    for (int i = 0; i < g_wc_n; i++)
        if (g_wc[i].host_ptr == w) return g_wc[i].dev;
    float *d = NULL;
    if (cudaMalloc(&d, bytes) != cudaSuccess) return NULL;
    if (cudaMemcpy(d, w, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d);
        return NULL;
    }
    if (g_wc_n == g_wc_cap) {
        g_wc_cap = g_wc_cap ? g_wc_cap * 2 : 256;
        g_wc = (wc_ent *)realloc(g_wc, (size_t)g_wc_cap * sizeof(wc_ent));
    }
    g_wc[g_wc_n++] = (wc_ent){w, d};
    return d;
}

static float *io_dev(float **slot, size_t *cap, size_t bytes) {
    if (!*slot || *cap < bytes) {
        size_t want = *cap ? *cap : (1u << 20);
        while (want < bytes) want *= 2;
        if (*slot) cudaFree(*slot);
        if (cudaMalloc(slot, want) != cudaSuccess) { *slot = NULL; return NULL; }
        *cap = want;
    }
    return *slot;
}

/* out[T,n] = x[T,k] @ W[n,k]^T. 0 = ok, -1 = fallback CPU. */
extern "C" int mynah_cuda_gemm_wt(const float *x, const float *w, float *out,
                                  int T, int n, int k) {
    if (!mynah_cuda_available()) return -1;
    pthread_mutex_lock(&g_mu);
    int rc = -1;
    do {
        float *dw = weight_dev(w, (size_t)n * (size_t)k * 4u);
        float *dx = io_dev(&g_in, &g_in_cap, (size_t)T * (size_t)k * 4u);
        float *dy = io_dev(&g_out, &g_out_cap, (size_t)T * (size_t)n * 4u);
        if (!dw || !dx || !dy) break;
        if (cudaMemcpyAsync(dx, x, (size_t)T * (size_t)k * 4u,
                            cudaMemcpyHostToDevice, g_stream) != cudaSuccess) break;

        /* cuBLAS è column-major: out_rm[T,n] = x_rm[T,k] @ W_rm[n,k]^T equivale a
         * out_cm[n,T] = W_cm[k,n]^T @ x_cm[k,T] -> gemm(op=T, op=N, m=n, n=T, k=k) */
        const float alpha = 1.0f, beta = 0.0f;
        if (cublasSgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, n, T, k,
                        &alpha, dw, k, dx, k, &beta, dy, n) != CUBLAS_STATUS_SUCCESS)
            break;
        if (cudaMemcpyAsync(out, dy, (size_t)T * (size_t)n * 4u,
                            cudaMemcpyDeviceToHost, g_stream) != cudaSuccess) break;
        if (cudaStreamSynchronize(g_stream) != cudaSuccess) break;
        rc = 0;
    } while (0);
    pthread_mutex_unlock(&g_mu);
    return rc;
}

#endif /* MYNAH_CUDA */
