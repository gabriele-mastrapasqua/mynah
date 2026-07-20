/* Backend CUDA per le GEMM grandi — pattern da qwen-tts (qwen_tts_cuda.c):
 * WEIGHT CACHE RESIDENTE condivisa (ogni pointer di peso caricato sul device
 * UNA volta) + CONTESTO PER-THREAD (handle cuBLAS, stream, buffer I/O in TLS):
 * N richieste server concorrenti fanno GEMM su stream indipendenti senza
 * serializzarsi — prima c'era un mutex globale attorno all'intera chiamata e
 * il throughput multi-richiesta era piatto (misurato A100 2026-07-20).
 *
 * Validato su hardware 2026-07-20 (A100-SXM4-40GB, CUDA 12.8, Ubuntu 24.04):
 * trascrizioni identiche a CPU su tutti i 10 modelli supportati, e2e verdi,
 * RTF in docs/benchmarks.md. Fallback CPU automatico su ogni errore.
 * I contesti TLS vivono fino all'exit del thread (server: pool persistente);
 * niente cleanup esplicito, come i pool BLAS. */
#ifdef MYNAH_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_ready = -1;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER; /* init + weight cache */

typedef struct {
    const void *host_ptr;
    float *dev;
} wc_ent;
static wc_ent *g_wc;
static int g_wc_n, g_wc_cap;

/* contesto per-thread: handle+stream+buffer. Creato lazy alla prima GEMM. */
typedef struct {
    cublasHandle_t blas;
    cudaStream_t stream;
    float *in, *out;
    size_t in_cap, out_cap;
    int state; /* 0 = da inizializzare, 1 = ok, -1 = fallito (resta su CPU) */
} cu_tls;
static __thread cu_tls t_cu;

extern "C" int mynah_cuda_available(void) {
    pthread_mutex_lock(&g_mu);
    if (g_ready < 0) {
        int n = 0;
        g_ready = cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
    }
    pthread_mutex_unlock(&g_mu);
    return g_ready == 1;
}

static cu_tls *tls_ctx(void) {
    if (t_cu.state == 0) {
        t_cu.state = cublasCreate(&t_cu.blas) == CUBLAS_STATUS_SUCCESS &&
                     cudaStreamCreate(&t_cu.stream) == cudaSuccess ? 1 : -1;
        if (t_cu.state == 1) cublasSetStream(t_cu.blas, t_cu.stream);
    }
    return t_cu.state == 1 ? &t_cu : NULL;
}

/* upload once, condiviso tra i thread: lookup lock-free non serve (la cache è
 * append-only ma realloc sposta l'array) -> mutex, tenuto solo per la cache. */
static float *weight_dev(const void *w, size_t bytes) {
    pthread_mutex_lock(&g_mu);
    for (int i = 0; i < g_wc_n; i++)
        if (g_wc[i].host_ptr == w) {
            float *d = g_wc[i].dev;
            pthread_mutex_unlock(&g_mu);
            return d;
        }
    float *d = NULL;
    if (cudaMalloc(&d, bytes) != cudaSuccess ||
        cudaMemcpy(d, w, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        if (d) cudaFree(d);
        pthread_mutex_unlock(&g_mu);
        return NULL;
    }
    if (g_wc_n == g_wc_cap) {
        g_wc_cap = g_wc_cap ? g_wc_cap * 2 : 256;
        g_wc = (wc_ent *)realloc(g_wc, (size_t)g_wc_cap * sizeof(wc_ent));
    }
    g_wc[g_wc_n++] = (wc_ent){w, d};
    pthread_mutex_unlock(&g_mu);
    return d;
}

static float *io_dev(float **slot, size_t *cap, size_t bytes) {
    if (!*slot || *cap < bytes) {
        size_t want = *cap ? *cap : (1u << 20);
        while (want < bytes) want *= 2;
        if (*slot) cudaFree(*slot);
        if (cudaMalloc(slot, want) != cudaSuccess) { *slot = NULL; *cap = 0; return NULL; }
        *cap = want;
    }
    return *slot;
}

/* out[T,n] = x[T,k] @ W[n,k]^T. 0 = ok, -1 = fallback CPU. */
extern "C" int mynah_cuda_gemm_wt(const float *x, const float *w, float *out,
                                  int T, int n, int k) {
    if (!mynah_cuda_available()) return -1;
    cu_tls *c = tls_ctx();
    if (!c) return -1;

    float *dw = weight_dev(w, (size_t)n * (size_t)k * 4u);
    float *dx = io_dev(&c->in, &c->in_cap, (size_t)T * (size_t)k * 4u);
    float *dy = io_dev(&c->out, &c->out_cap, (size_t)T * (size_t)n * 4u);
    if (!dw || !dx || !dy) return -1;
    if (cudaMemcpyAsync(dx, x, (size_t)T * (size_t)k * 4u,
                        cudaMemcpyHostToDevice, c->stream) != cudaSuccess) return -1;

    /* cuBLAS è column-major: out_rm[T,n] = x_rm[T,k] @ W_rm[n,k]^T equivale a
     * out_cm[n,T] = W_cm[k,n]^T @ x_cm[k,T] -> gemm(op=T, op=N, m=n, n=T, k=k) */
    const float alpha = 1.0f, beta = 0.0f;
    if (cublasSgemm(c->blas, CUBLAS_OP_T, CUBLAS_OP_N, n, T, k,
                    &alpha, dw, k, dx, k, &beta, dy, n) != CUBLAS_STATUS_SUCCESS)
        return -1;
    if (cudaMemcpyAsync(out, dy, (size_t)T * (size_t)n * 4u,
                        cudaMemcpyDeviceToHost, c->stream) != cudaSuccess) return -1;
    if (cudaStreamSynchronize(c->stream) != cudaSuccess) return -1;
    return 0;
}

#endif /* MYNAH_CUDA */
