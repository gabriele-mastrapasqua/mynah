/* Backend di calcolo per le GEMM grandi: CPU (BLAS) o Metal/MPS su macOS.
 * Pattern qwen-tts: richiesta -> resolve() -> nota -> fallback graceful CPU.
 * Il backend è una scelta di processo (mynah_set_backend prima del load). */
#ifndef MYNAH_BACKEND_H
#define MYNAH_BACKEND_H

enum { MYNAH_BACKEND_CPU = 0, MYNAH_BACKEND_METAL = 1, MYNAH_BACKEND_CUDA = 2 };

/* "cpu" | "metal" | "cuda". Ritorna il backend EFFETTIVO dopo il resolve (fallback CPU
 * con nota su stderr se quello richiesto non è disponibile). */
int mynah_set_backend(const char *name);
int mynah_backend(void);

/* out[T,n] = x[T,k] @ W[n,k]^T — dispatch: Metal per T grandi se attivo, else BLAS.
 * W deve essere stabile per la vita del processo (i pesi mmap lo sono): su Metal
 * viene copiato UNA volta in un MTLBuffer residente (cache per-pointer). */
void mynah_gemm_wt(const float *x, const float *w, float *out, int T, int n, int k);

/* FFN fusa: out[T,n2] = SiLU(x @ W1^T) @ W2^T. scratch: >= T*n1 float (usato
 * solo nel fallback CPU). Su Metal l'intermedio resta in GPU (un solo sync). */
void mynah_ffn_wt(const float *x, const float *w1, int n1, const float *w2, int n2,
                  float *out, int T, int k, float *scratch);

/* Tre GEMM sullo stesso input (q/k/v): su Metal un solo sync. */
void mynah_gemm3_wt(const float *x, const float *wa, const float *wb, const float *wc,
                    float *oa, float *ob, float *oc, int T, int n, int k);

#endif
