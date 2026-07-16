#include "encoder.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MYNAH_BLAS_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

/* ------------------------------------------------------------------ helpers */
static const float *T_(const mynah_safetensors *st, const char *fmt, int li, const char *suffix) {
    char name[160];
    snprintf(name, sizeof(name), fmt, li, suffix);
    const mynah_tensor *t = mynah_st_get(st, name);
    return t ? (const float *)t->data : NULL;
}

static void layer_norm_f(const float *x, const float *w, const float *b, float *out, int T, int d) {
    for (int t = 0; t < T; t++) {
        const float *row = x + (size_t)t * (size_t)d;
        float *o = out + (size_t)t * (size_t)d;
        double mu = 0.0, var = 0.0;
        for (int i = 0; i < d; i++) mu += row[i];
        mu /= d;
        for (int i = 0; i < d; i++) { double c = row[i] - mu; var += c * c; }
        var /= d;
        const float inv = (float)(1.0 / sqrt(var + 1e-5));
        for (int i = 0; i < d; i++) o[i] = ((row[i] - (float)mu) * inv) * w[i] + b[i];
    }
}

static void silu_inplace(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) x[i] = x[i] / (1.0f + expf(-x[i]));
}

/* out[T,n] = x[T,k] @ W[n,k]^T (row-major, layout linear PyTorch) */
static void matmul_wt(const float *x, const float *w, float *out, int T, int n, int k) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, n, k, 1.0f, x, k, w, k, 0.0f, out, n);
}

/* ------------------------------------------------------------------- init */
int mynah_encoder_init(mynah_encoder *enc, const mynah_safetensors *st) {
    memset(enc, 0, sizeof(*enc));
    if (mynah_subsampling_init(&enc->ss, st) != 0) return -1;

    /* dimensioni dalle shape */
    const mynah_tensor *bu = mynah_st_get(st, "encoder.layers.0.self_attn.bias_u");
    const mynah_tensor *ff1 = mynah_st_get(st, "encoder.layers.0.feed_forward1.linear1.weight");
    const mynah_tensor *dw = mynah_st_get(st, "encoder.layers.0.conv.depthwise_conv.weight");
    const mynah_tensor *ep = mynah_st_get(st, "encoder_projector.weight");
    const mynah_tensor *p1 = mynah_st_get(st, "prompt_projector.linear_1.weight");
    if (!bu || !ff1 || !dw || !ep || !p1) return -1;

    enc->d_model = enc->ss.d_model;
    enc->n_heads = (int)bu->shape[0];
    enc->d_head = (int)bu->shape[1];
    enc->ffn_dim = (int)ff1->shape[0];
    enc->conv_k = (int)dw->shape[2];
    enc->d_out = (int)ep->shape[0];
    enc->prompt_inter = (int)p1->shape[0];
    enc->num_prompts = (int)p1->shape[1] - enc->d_model;

    /* conta i layer */
    int n = 0;
    while (T_(st, "encoder.layers.%d.%s", n, "norm_out.weight")) n++;
    enc->n_layers = n;
    enc->layers = calloc((size_t)n, sizeof(mynah_enc_layer));
    if (!enc->layers || n == 0) return -1;

    const char *F = "encoder.layers.%d.%s";
    for (int li = 0; li < n; li++) {
        mynah_enc_layer *L = &enc->layers[li];
        L->ln_ff1_w = T_(st, F, li, "norm_feed_forward1.weight");
        L->ln_ff1_b = T_(st, F, li, "norm_feed_forward1.bias");
        L->ff1_w1 = T_(st, F, li, "feed_forward1.linear1.weight");
        L->ff1_w2 = T_(st, F, li, "feed_forward1.linear2.weight");
        L->ln_att_w = T_(st, F, li, "norm_self_att.weight");
        L->ln_att_b = T_(st, F, li, "norm_self_att.bias");
        L->q_w = T_(st, F, li, "self_attn.q_proj.weight");
        L->k_w = T_(st, F, li, "self_attn.k_proj.weight");
        L->v_w = T_(st, F, li, "self_attn.v_proj.weight");
        L->o_w = T_(st, F, li, "self_attn.o_proj.weight");
        L->relk_w = T_(st, F, li, "self_attn.relative_k_proj.weight");
        L->bias_u = T_(st, F, li, "self_attn.bias_u");
        L->bias_v = T_(st, F, li, "self_attn.bias_v");
        L->ln_conv_w = T_(st, F, li, "norm_conv.weight");
        L->ln_conv_b = T_(st, F, li, "norm_conv.bias");
        L->pw1_w = T_(st, F, li, "conv.pointwise_conv1.weight");
        L->dw_w = T_(st, F, li, "conv.depthwise_conv.weight");
        L->cnorm_w = T_(st, F, li, "conv.norm.weight");
        L->cnorm_b = T_(st, F, li, "conv.norm.bias");
        L->pw2_w = T_(st, F, li, "conv.pointwise_conv2.weight");
        L->ln_ff2_w = T_(st, F, li, "norm_feed_forward2.weight");
        L->ln_ff2_b = T_(st, F, li, "norm_feed_forward2.bias");
        L->ff2_w1 = T_(st, F, li, "feed_forward2.linear1.weight");
        L->ff2_w2 = T_(st, F, li, "feed_forward2.linear2.weight");
        L->ln_out_w = T_(st, F, li, "norm_out.weight");
        L->ln_out_b = T_(st, F, li, "norm_out.bias");
        if (!L->ln_ff1_w || !L->ff1_w1 || !L->q_w || !L->relk_w || !L->pw1_w ||
            !L->cnorm_w || !L->ff2_w1 || !L->ln_out_w) {
            fprintf(stderr, "encoder: tensori mancanti al layer %d\n", li);
            return -1;
        }
    }

    enc->prompt_l1_w = (const float *)p1->data;
    enc->prompt_l1_b = (const float *)mynah_st_get(st, "prompt_projector.linear_1.bias")->data;
    enc->prompt_l2_w = (const float *)mynah_st_get(st, "prompt_projector.linear_2.weight")->data;
    enc->prompt_l2_b = (const float *)mynah_st_get(st, "prompt_projector.linear_2.bias")->data;
    enc->encproj_w = (const float *)ep->data;
    enc->encproj_b = (const float *)mynah_st_get(st, "encoder_projector.bias")->data;
    return 0;
}

void mynah_encoder_free(mynah_encoder *enc) { free(enc->layers); enc->layers = NULL; }

/* --------------------------------------------------------------- pos emb */
void mynah_pos_emb(const mynah_encoder *enc, int T, float *pe) {
    const int d = enc->d_model, P = 2 * T - 1;
    for (int p = 0; p < P; p++) {
        const double pos = (double)(T - 1 - p);
        for (int j = 0; j < d / 2; j++) {
            const double freq = pos * pow(10000.0, -2.0 * j / (double)d);
            pe[(size_t)p * (size_t)d + 2 * (size_t)j] = (float)sin(freq);
            pe[(size_t)p * (size_t)d + 2 * (size_t)j + 1] = (float)cos(freq);
        }
    }
}

/* ------------------------------------------------------------- attention */
static void attention(const mynah_encoder *enc, const mynah_enc_layer *L, const float *x,
                      float *out, int T, const float *pe, int left, int right) {
    const int d = enc->d_model, H = enc->n_heads, dk = enc->d_head, P = 2 * T - 1;
    const float scaling = 1.0f / sqrtf((float)dk);
    const int chunk = right + 1, lc = left / chunk;

    float *q = malloc(3 * (size_t)T * (size_t)d * sizeof(float));
    float *k = q + (size_t)T * (size_t)d;
    float *v = k + (size_t)T * (size_t)d;
    float *rk = malloc((size_t)P * (size_t)d * sizeof(float));
    float *scores = malloc((size_t)T * (size_t)T * sizeof(float));
    float *bd = malloc((size_t)T * (size_t)P * sizeof(float));
    float *qb = malloc((size_t)T * (size_t)d * sizeof(float));
    float *ctx = malloc((size_t)T * (size_t)d * sizeof(float));
    if (!q || !rk || !scores || !bd || !qb || !ctx) { free(q); free(rk); free(scores); free(bd); free(qb); free(ctx); return; }

    matmul_wt(x, L->q_w, q, T, d, d);
    matmul_wt(x, L->k_w, k, T, d, d);
    matmul_wt(x, L->v_w, v, T, d, d);
    matmul_wt(pe, L->relk_w, rk, P, d, d);

    for (int h = 0; h < H; h++) {
        const size_t ho = (size_t)h * (size_t)dk;
        /* q + bias_v (per matrix_bd): gemm su viste strided per head */
        for (int t = 0; t < T; t++)
            for (int i = 0; i < dk; i++)
                qb[(size_t)t * (size_t)dk + (size_t)i] = q[(size_t)t * (size_t)d + ho + (size_t)i] + L->bias_v[ho + (size_t)i];
        /* bd_full[t, p] = qv[t] . rk[p]  — rk per head è strided: gemm con lda=d */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, P, dk,
                    1.0f, qb, dk, rk + ho, d, 0.0f, bd, P);

        /* q + bias_u (per matrix_ac) */
        for (int t = 0; t < T; t++)
            for (int i = 0; i < dk; i++)
                qb[(size_t)t * (size_t)dk + (size_t)i] = q[(size_t)t * (size_t)d + ho + (size_t)i] + L->bias_u[ho + (size_t)i];
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, T, dk,
                    1.0f, qb, dk, k + ho, d, 0.0f, scores, T);

        /* scores = (ac + rel_shift(bd)) * scaling + softmax, SOLO sulla finestra
         * chunked_limited — che per riga è contigua: jc in [tc-lc, tc]
         * => j in [ (tc-lc)*chunk, (tc+1)*chunk ). Niente -inf (UB con -ffast-math),
         * niente lavoro sui masked (prior-art §A: finestra nei limiti del loop). */
        for (int t = 0; t < T; t++) {
            float *srow = scores + (size_t)t * (size_t)T;
            const float *brow = bd + (size_t)t * (size_t)P;
            const int tc = t / chunk;
            int j0 = (tc - lc) * chunk;
            if (j0 < 0) j0 = 0;
            int j1 = (tc + 1) * chunk;
            if (j1 > T) j1 = T;

            float maxv = -3.0e38f;
            for (int j = j0; j < j1; j++) {
                /* rel_shift in forma chiusa: bd_shifted[t,j] = bd[t, T-1 + j - t] */
                srow[j] = (srow[j] + brow[T - 1 + j - t]) * scaling;
                if (srow[j] > maxv) maxv = srow[j];
            }
            float sum = 0.0f;
            for (int j = j0; j < j1; j++) {
                srow[j] = expf(srow[j] - maxv);
                sum += srow[j];
            }
            const float inv = 1.0f / sum;
            for (int j = 0; j < j0; j++) srow[j] = 0.0f;
            for (int j = j0; j < j1; j++) srow[j] *= inv;
            for (int j = j1; j < T; j++) srow[j] = 0.0f;
        }

        /* ctx_h = scores @ v_h (v strided per head) */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, T, dk, T,
                    1.0f, scores, T, v + ho, d, 0.0f, qb, dk);
        for (int t = 0; t < T; t++)
            memcpy(ctx + (size_t)t * (size_t)d + ho, qb + (size_t)t * (size_t)dk, (size_t)dk * sizeof(float));
    }

    matmul_wt(ctx, L->o_w, out, T, d, d);
    free(q); free(rk); free(scores); free(bd); free(qb); free(ctx);
}

/* ------------------------------------------------------------ conv module */
static void conv_module(const mynah_encoder *enc, const mynah_enc_layer *L, const float *x,
                        float *out, int T) {
    const int d = enc->d_model, k = enc->conv_k;
    float *h2 = malloc((size_t)T * 2u * (size_t)d * sizeof(float));
    float *g = malloc((size_t)T * (size_t)d * sizeof(float));
    float *c = malloc((size_t)T * (size_t)d * sizeof(float));
    if (!h2 || !g || !c) { free(h2); free(g); free(c); return; }

    /* pointwise_conv1 [2d, d, 1] come linear, poi GLU sui canali */
    matmul_wt(x, L->pw1_w, h2, T, 2 * d, d);
    for (int t = 0; t < T; t++) {
        const float *a = h2 + (size_t)t * 2u * (size_t)d;
        const float *b = a + d;
        float *o = g + (size_t)t * (size_t)d;
        for (int i = 0; i < d; i++) o[i] = a[i] / (1.0f + expf(-b[i]));
    }

    /* depthwise causale k: pad left k-1 (zeri) — dw_w [d, 1, k] */
    for (int t = 0; t < T; t++) {
        float *o = c + (size_t)t * (size_t)d;
        memset(o, 0, (size_t)d * sizeof(float));
        const int j0 = (t - (k - 1) < 0) ? (k - 1 - t) : 0;
        for (int j = j0; j < k; j++) {
            const float *src = g + (size_t)(t - (k - 1) + j) * (size_t)d;
            for (int i = 0; i < d; i++) o[i] += L->dw_w[(size_t)i * (size_t)k + (size_t)j] * src[i];
        }
    }

    layer_norm_f(c, L->cnorm_w, L->cnorm_b, g, T, d);
    silu_inplace(g, (size_t)T * (size_t)d);
    matmul_wt(g, L->pw2_w, out, T, d, d);
    free(h2); free(g); free(c);
}

/* ------------------------------------------------------------------ layer */
int mynah_encoder_layer(const mynah_encoder *enc, int li, float *x, int T,
                        const float *pe, int left_ctx, int right_ctx) {
    const mynah_enc_layer *L = &enc->layers[li];
    const int d = enc->d_model;
    const size_t n = (size_t)T * (size_t)d;
    float *tmp = malloc(n * sizeof(float));
    float *tmp2 = malloc((size_t)T * (size_t)enc->ffn_dim * sizeof(float));
    if (!tmp || !tmp2) { free(tmp); free(tmp2); return -1; }

    /* ½ FFN1 */
    layer_norm_f(x, L->ln_ff1_w, L->ln_ff1_b, tmp, T, d);
    matmul_wt(tmp, L->ff1_w1, tmp2, T, enc->ffn_dim, d);
    silu_inplace(tmp2, (size_t)T * (size_t)enc->ffn_dim);
    matmul_wt(tmp2, L->ff1_w2, tmp, T, d, enc->ffn_dim);
    for (size_t i = 0; i < n; i++) x[i] += 0.5f * tmp[i];

    /* MHSA */
    float *xn = malloc(n * sizeof(float));
    if (!xn) { free(tmp); free(tmp2); return -1; }
    layer_norm_f(x, L->ln_att_w, L->ln_att_b, xn, T, d);
    attention(enc, L, xn, tmp, T, pe, left_ctx, right_ctx);
    for (size_t i = 0; i < n; i++) x[i] += tmp[i];

    /* Conv */
    layer_norm_f(x, L->ln_conv_w, L->ln_conv_b, xn, T, d);
    conv_module(enc, L, xn, tmp, T);
    for (size_t i = 0; i < n; i++) x[i] += tmp[i];

    /* ½ FFN2 */
    layer_norm_f(x, L->ln_ff2_w, L->ln_ff2_b, tmp, T, d);
    matmul_wt(tmp, L->ff2_w1, tmp2, T, enc->ffn_dim, d);
    silu_inplace(tmp2, (size_t)T * (size_t)enc->ffn_dim);
    matmul_wt(tmp2, L->ff2_w2, tmp, T, d, enc->ffn_dim);
    for (size_t i = 0; i < n; i++) x[i] += 0.5f * tmp[i];

    /* LN out */
    layer_norm_f(x, L->ln_out_w, L->ln_out_b, xn, T, d);
    memcpy(x, xn, n * sizeof(float));
    free(tmp); free(tmp2); free(xn);
    return 0;
}

/* -------------------------------------------------- prompt + projector */
void mynah_encoder_post(const mynah_encoder *enc, const float *x, int T, int prompt_id,
                        float *out) {
    const int d = enc->d_model, np = enc->num_prompts, di = enc->prompt_inter;
    const int dcat = d + np;
    float *cat = calloc((size_t)T * (size_t)dcat, sizeof(float));
    float *mid = malloc((size_t)T * (size_t)di * sizeof(float));
    float *fused = malloc((size_t)T * (size_t)d * sizeof(float));
    if (!cat || !mid || !fused) { free(cat); free(mid); free(fused); return; }

    for (int t = 0; t < T; t++) {
        memcpy(cat + (size_t)t * (size_t)dcat, x + (size_t)t * (size_t)d, (size_t)d * sizeof(float));
        cat[(size_t)t * (size_t)dcat + (size_t)d + (size_t)prompt_id] = 1.0f;
    }
    matmul_wt(cat, enc->prompt_l1_w, mid, T, di, dcat);
    for (int t = 0; t < T; t++)
        for (int i = 0; i < di; i++) {
            float *v = &mid[(size_t)t * (size_t)di + (size_t)i];
            *v += enc->prompt_l1_b[i];
            if (*v < 0.0f) *v = 0.0f;
        }
    matmul_wt(mid, enc->prompt_l2_w, fused, T, d, di);
    for (int t = 0; t < T; t++)
        for (int i = 0; i < d; i++) fused[(size_t)t * (size_t)d + (size_t)i] += enc->prompt_l2_b[i];

    matmul_wt(fused, enc->encproj_w, out, T, enc->d_out, d);
    for (int t = 0; t < T; t++)
        for (int i = 0; i < enc->d_out; i++) out[(size_t)t * (size_t)enc->d_out + (size_t)i] += enc->encproj_b[i];
    free(cat); free(mid); free(fused);
}

/* ---------------------------------------------------------------- forward */
float *mynah_encoder_forward(const mynah_encoder *enc, const float *feats, int t_mel,
                             int n_mels, int prompt_id, int left_ctx, int right_ctx,
                             int *t_out) {
    int T;
    float *x = mynah_subsampling_forward(&enc->ss, feats, t_mel, n_mels, &T);
    if (!x) return NULL;

    float *pe = malloc((size_t)(2 * T - 1) * (size_t)enc->d_model * sizeof(float));
    if (!pe) { free(x); return NULL; }
    mynah_pos_emb(enc, T, pe);

    for (int li = 0; li < enc->n_layers; li++)
        if (mynah_encoder_layer(enc, li, x, T, pe, left_ctx, right_ctx) != 0) {
            free(x); free(pe);
            return NULL;
        }
    free(pe);

    float *out = malloc((size_t)T * (size_t)enc->d_out * sizeof(float));
    if (!out) { free(x); return NULL; }
    mynah_encoder_post(enc, x, T, prompt_id, out);
    free(x);
    *t_out = T;
    return out;
}
