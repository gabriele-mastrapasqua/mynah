#include "subsampling.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MYNAH_BLAS_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

int mynah_subsampling_init(mynah_subsampling *ss, const mynah_safetensors *st) {
    memset(ss, 0, sizeof(*ss));
    ss->conv_in_w = mynah_st_get(st, "encoder.subsampling.conv_in.weight");
    ss->conv_in_b = mynah_st_get(st, "encoder.subsampling.conv_in.bias");
    ss->lin_w = mynah_st_get(st, "encoder.subsampling.linear.weight");
    ss->lin_b = mynah_st_get(st, "encoder.subsampling.linear.bias");
    for (int i = 0; i < 2; i++) {
        char name[96];
        snprintf(name, sizeof(name), "encoder.subsampling.layers.%d.depthwise_conv.weight", i);
        ss->dw_w[i] = mynah_st_get(st, name);
        snprintf(name, sizeof(name), "encoder.subsampling.layers.%d.depthwise_conv.bias", i);
        ss->dw_b[i] = mynah_st_get(st, name);
        snprintf(name, sizeof(name), "encoder.subsampling.layers.%d.pointwise_conv.weight", i);
        ss->pw_w[i] = mynah_st_get(st, name);
        snprintf(name, sizeof(name), "encoder.subsampling.layers.%d.pointwise_conv.bias", i);
        ss->pw_b[i] = mynah_st_get(st, name);
        if (!ss->dw_w[i] || !ss->dw_b[i] || !ss->pw_w[i] || !ss->pw_b[i]) return -1;
    }
    if (!ss->conv_in_w || !ss->conv_in_b || !ss->lin_w || !ss->lin_b) return -1;
    ss->channels = (int)ss->conv_in_w->shape[0];
    ss->d_model = (int)ss->lin_w->shape[0];
    return 0;
}

/* pad causale (2,1) su tempo e freq, poi conv k3 s2. x [C_in, T, F] -> out [C_out, T', F'].
 * depthwise: C_in == C_out, weight [C,1,3,3]. full: weight [C_out, C_in, 3, 3]. */
static void conv2d_causal_s2(const float *x, int C_in, int T, int F,
                             const float *w, const float *b, int C_out, int depthwise,
                             float *out, int *To_, int *Fo_) {
    const int k = 3, s = 2, pl = 2, pr = 1;
    const int Tp = T + pl + pr, Fp = F + pl + pr;
    const int To = (Tp - k) / s + 1, Fo = (Fp - k) / s + 1;
    *To_ = To; *Fo_ = Fo;

    for (int co = 0; co < C_out; co++) {
        float *dst = out + (size_t)co * (size_t)To * (size_t)Fo;
        for (int i = 0; i < To * Fo; i++) dst[i] = b[co];
    }
    /* loop diretto: (kernel 3x3, stride 2) — chiaro e cache-friendly per queste dimensioni */
    for (int co = 0; co < C_out; co++) {
        const int ci_lo = depthwise ? co : 0;
        const int ci_hi = depthwise ? co + 1 : C_in;
        float *dst = out + (size_t)co * (size_t)To * (size_t)Fo;
        for (int ci = ci_lo; ci < ci_hi; ci++) {
            const float *src = x + (size_t)ci * (size_t)T * (size_t)F;
            const float *ker = depthwise ? w + (size_t)co * 9u
                                         : w + ((size_t)co * (size_t)C_in + (size_t)ci) * 9u;
            for (int to = 0; to < To; to++) {
                const int t0 = to * s - pl;
                for (int fo = 0; fo < Fo; fo++) {
                    const int f0 = fo * s - pl;
                    float acc = 0.0f;
                    for (int dt = 0; dt < k; dt++) {
                        const int t = t0 + dt;
                        if (t < 0 || t >= T) continue;
                        for (int df = 0; df < k; df++) {
                            const int fq = f0 + df;
                            if (fq < 0 || fq >= F) continue;
                            acc += ker[dt * 3 + df] * src[(size_t)t * (size_t)F + (size_t)fq];
                        }
                    }
                    dst[(size_t)to * (size_t)Fo + (size_t)fo] += acc;
                }
            }
        }
    }
}

static void relu_inplace(float *x, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (x[i] < 0.0f) x[i] = 0.0f;
}

float *mynah_subsampling_forward(const mynah_subsampling *ss, const float *feats,
                                 int T, int n_mels, int *t_out) {
    const int C = ss->channels;

    /* buffer di lavoro dimensionati sul primo stadio (il più grande) */
    int To = 0, Fo = 0;
    const int To_max = (T + 3 - 3) / 2 + 1, Fo_max = (n_mels + 3 - 3) / 2 + 1;
    float *a = malloc((size_t)C * (size_t)To_max * (size_t)Fo_max * sizeof(float));
    float *bbuf = malloc((size_t)C * (size_t)To_max * (size_t)Fo_max * sizeof(float));
    if (!a || !bbuf) { free(a); free(bbuf); return NULL; }

    /* stadio 0: conv piena 1 -> C su [1, T, n_mels] */
    conv2d_causal_s2(feats, 1, T, n_mels,
                     (const float *)ss->conv_in_w->data, (const float *)ss->conv_in_b->data,
                     C, 0, a, &To, &Fo);
    relu_inplace(a, (size_t)C * (size_t)To * (size_t)Fo);

    /* stadi 1..2: depthwise + pointwise + ReLU */
    for (int i = 0; i < 2; i++) {
        int To2, Fo2;
        conv2d_causal_s2(a, C, To, Fo,
                         (const float *)ss->dw_w[i]->data, (const float *)ss->dw_b[i]->data,
                         C, 1, bbuf, &To2, &Fo2);
        /* pointwise 1x1: out[co, t, f] = sum_ci w[co, ci] * b[ci, t, f]  (GEMM [C, C] x [C, S]) */
        const size_t S = (size_t)To2 * (size_t)Fo2;
        const float *pw = (const float *)ss->pw_w[i]->data; /* [C, C, 1, 1] -> [C, C] */
        const float *pb = (const float *)ss->pw_b[i]->data;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, C, (int)S, C,
                    1.0f, pw, C, bbuf, (int)S, 0.0f, a, (int)S);
        for (int co = 0; co < C; co++) {
            float *row = a + (size_t)co * S;
            for (size_t j = 0; j < S; j++) row[j] += pb[co];
        }
        relu_inplace(a, (size_t)C * S);
        To = To2; Fo = Fo2;
    }

    /* flatten channel-major [T', C*F'] + linear -> d_model */
    const int CF = C * Fo;
    float *flat = malloc((size_t)To * (size_t)CF * sizeof(float));
    float *out = malloc((size_t)To * (size_t)ss->d_model * sizeof(float));
    if (!flat || !out) { free(a); free(bbuf); free(flat); free(out); return NULL; }
    for (int t = 0; t < To; t++)
        for (int c = 0; c < C; c++)
            for (int fq = 0; fq < Fo; fq++)
                flat[(size_t)t * (size_t)CF + (size_t)c * (size_t)Fo + (size_t)fq] =
                    a[(size_t)c * (size_t)To * (size_t)Fo + (size_t)t * (size_t)Fo + (size_t)fq];

    /* out = flat @ W^T + b — W [d_model, CF] row-major => GEMM con B trasposta */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, To, ss->d_model, CF,
                1.0f, flat, CF, (const float *)ss->lin_w->data, CF, 0.0f, out, ss->d_model);
    const float *lb = (const float *)ss->lin_b->data;
    for (int t = 0; t < To; t++)
        for (int d = 0; d < ss->d_model; d++) out[(size_t)t * (size_t)ss->d_model + (size_t)d] += lb[d];

    free(a); free(bbuf); free(flat);
    *t_out = To;
    return out;
}
