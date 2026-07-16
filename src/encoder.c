#include "encoder.h"

#include "backend.h"

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
    mynah_gemm_wt(x, w, out, T, n, k);
}

/* ------------------------------------------------------------------- init */
int mynah_encoder_init(mynah_encoder *enc, const mynah_safetensors *st, int quantize) {
    memset(enc, 0, sizeof(*enc));
    if (mynah_subsampling_init(&enc->ss, st) != 0) return -1;

    /* dimensioni dalle shape (ffn_dim dopo l'init del primo qmat: nel file
     * pre-quantizzato il f32 di linear1 non esiste) */
    const mynah_tensor *bu = mynah_st_get(st, "encoder.layers.0.self_attn.bias_u");
    const mynah_tensor *dw = mynah_st_get(st, "encoder.layers.0.conv.depthwise_conv.weight");
    const mynah_tensor *ep = mynah_st_get(st, "encoder_projector.weight");
    const mynah_tensor *p1 = mynah_st_get(st, "prompt_projector.linear_1.weight");
    if (!bu || !dw || !ep || !p1) return -1;

    enc->d_model = enc->ss.d_model;
    enc->n_heads = (int)bu->shape[0];
    enc->d_head = (int)bu->shape[1];
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
        L->ln_att_w = T_(st, F, li, "norm_self_att.weight");
        L->ln_att_b = T_(st, F, li, "norm_self_att.bias");
        L->relk_w = T_(st, F, li, "self_attn.relative_k_proj.weight");
        L->bias_u = T_(st, F, li, "self_attn.bias_u");
        L->bias_v = T_(st, F, li, "self_attn.bias_v");
        L->ln_conv_w = T_(st, F, li, "norm_conv.weight");
        L->ln_conv_b = T_(st, F, li, "norm_conv.bias");
        L->dw_w = T_(st, F, li, "conv.depthwise_conv.weight");
        L->cnorm_w = T_(st, F, li, "conv.norm.weight");
        L->cnorm_b = T_(st, F, li, "conv.norm.bias");
        L->ln_ff2_w = T_(st, F, li, "norm_feed_forward2.weight");
        L->ln_ff2_b = T_(st, F, li, "norm_feed_forward2.bias");
        L->ln_out_w = T_(st, F, li, "norm_out.weight");
        L->ln_out_b = T_(st, F, li, "norm_out.bias");
        if (!L->ln_ff1_w || !L->relk_w || !L->cnorm_w || !L->ln_out_w) {
            fprintf(stderr, "encoder: tensori mancanti al layer %d\n", li);
            return -1;
        }
        /* grandi linear: qmat — cerca prima la forma pre-quantizzata (.q8/.q4) */
        int rc = 0;
        char qn[160];
        #define QM(field, suffix) \
            (snprintf(qn, sizeof(qn), "encoder.layers.%d." suffix, li), \
             mynah_qmat_init_st(&L->field, st, qn, quantize))
        rc |= QM(ff1_w1, "feed_forward1.linear1.weight");
        rc |= QM(ff1_w2, "feed_forward1.linear2.weight");
        rc |= QM(ff2_w1, "feed_forward2.linear1.weight");
        rc |= QM(ff2_w2, "feed_forward2.linear2.weight");
        rc |= QM(q_w, "self_attn.q_proj.weight");
        rc |= QM(k_w, "self_attn.k_proj.weight");
        rc |= QM(v_w, "self_attn.v_proj.weight");
        rc |= QM(o_w, "self_attn.o_proj.weight");
        rc |= QM(pw1_w, "conv.pointwise_conv1.weight");
        rc |= QM(pw2_w, "conv.pointwise_conv2.weight");
        #undef QM
        if (rc != 0) {
            fprintf(stderr, "encoder: init qmat fallita al layer %d\n", li);
            return -1;
        }
    }

    enc->ffn_dim = enc->layers[0].ff1_w1.n;

    enc->prompt_l1_w = (const float *)p1->data;
    enc->prompt_l1_b = (const float *)mynah_st_get(st, "prompt_projector.linear_1.bias")->data;
    enc->prompt_l2_w = (const float *)mynah_st_get(st, "prompt_projector.linear_2.weight")->data;
    enc->prompt_l2_b = (const float *)mynah_st_get(st, "prompt_projector.linear_2.bias")->data;
    enc->encproj_w = (const float *)ep->data;
    enc->encproj_b = (const float *)mynah_st_get(st, "encoder_projector.bias")->data;
    return 0;
}

void mynah_encoder_free(mynah_encoder *enc) {
    for (int li = 0; li < enc->n_layers && enc->layers; li++) {
        mynah_enc_layer *L = &enc->layers[li];
        mynah_qmat_free(&L->ff1_w1); mynah_qmat_free(&L->ff1_w2);
        mynah_qmat_free(&L->ff2_w1); mynah_qmat_free(&L->ff2_w2);
        mynah_qmat_free(&L->q_w); mynah_qmat_free(&L->k_w);
        mynah_qmat_free(&L->v_w); mynah_qmat_free(&L->o_w);
        mynah_qmat_free(&L->pw1_w); mynah_qmat_free(&L->pw2_w);
    }
    free(enc->layers);
    enc->layers = NULL;
}

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

    mynah_qmat_qkv(&L->q_w, &L->k_w, &L->v_w, x, q, k, v, T);
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

    mynah_qmat_mul(&L->o_w, ctx, out, T);
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
    mynah_qmat_mul(&L->pw1_w, x, h2, T);
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
    mynah_qmat_mul(&L->pw2_w, g, out, T);
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

    /* ½ FFN1 (path fuso: su Metal un solo sync) */
    layer_norm_f(x, L->ln_ff1_w, L->ln_ff1_b, tmp, T, d);
    mynah_qmat_ffn(&L->ff1_w1, &L->ff1_w2, tmp, tmp, T, tmp2);
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

    /* ½ FFN2 (fuso) */
    layer_norm_f(x, L->ln_ff2_w, L->ln_ff2_b, tmp, T, d);
    mynah_qmat_ffn(&L->ff2_w1, &L->ff2_w2, tmp, tmp, T, tmp2);
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

/* --------------------------------------------------------------- streaming */

int mynah_enc_stream_init(mynah_enc_stream *es, const mynah_encoder *enc,
                          int left_ctx, int right_ctx, int n_mels) {
    memset(es, 0, sizeof(*es));
    es->enc = enc;
    es->left = left_ctx;
    es->right = right_ctx;
    es->q = right_ctx + 1;
    if (mynah_ss_stream_init(&es->ss, &enc->ss, n_mels) != 0) return -1;
    const size_t kv = (size_t)enc->n_layers * (size_t)left_ctx * (size_t)enc->d_model;
    const size_t cv = (size_t)enc->n_layers * (size_t)(enc->conv_k - 1) * (size_t)enc->d_model;
    es->k_cache = calloc(kv, sizeof(float));
    es->v_cache = calloc(kv, sizeof(float));
    es->conv_cache = calloc(cv, sizeof(float));
    return (es->k_cache && es->v_cache && es->conv_cache) ? 0 : -1;
}

void mynah_enc_stream_free(mynah_enc_stream *es) {
    mynah_ss_stream_free(&es->ss);
    free(es->k_cache); free(es->v_cache); free(es->conv_cache);
    es->k_cache = es->v_cache = es->conv_cache = NULL;
}

int mynah_enc_stream_need(const mynah_enc_stream *es) {
    const int sub = 8; /* subsampling_factor: 3 stadi stride-2 */
    return es->cache_valid == 0 && es->ss.first ? 1 + sub * es->right
                                                : sub * (es->right + 1);
}

/* Attention streaming: Q righe query, K/V = [cache valida ; chunk], senza mask
 * (la cache contiene esattamente il left context ammesso dalla griglia chunked). */
static void stream_attention(const mynah_encoder *enc, const mynah_enc_layer *L,
                             const float *x, const float *kn, const float *vn,
                             float *out, int Q,
                             const float *k_cache, const float *v_cache, int valid) {
    const int d = enc->d_model, H = enc->n_heads, dk = enc->d_head;
    const int K = valid + Q, P = 2 * K - 1;
    const float scaling = 1.0f / sqrtf((float)dk);

    float *qkv = malloc((size_t)Q * (size_t)d * sizeof(float));
    float *keys = malloc(2 * (size_t)K * (size_t)d * sizeof(float));
    float *pe = malloc((size_t)P * (size_t)enc->d_model * sizeof(float));
    float *rk = malloc((size_t)P * (size_t)d * sizeof(float));
    float *scores = malloc((size_t)Q * (size_t)K * sizeof(float));
    float *bd = malloc((size_t)Q * (size_t)P * sizeof(float));
    float *qb = malloc((size_t)Q * (size_t)d * sizeof(float));
    float *ctx = malloc((size_t)Q * (size_t)d * sizeof(float));
    if (!qkv || !keys || !pe || !rk || !scores || !bd || !qb || !ctx) goto done;

    {
        float *q = qkv;
        float *kk = keys, *vv = keys + (size_t)K * (size_t)d;
        mynah_qmat_mul(&L->q_w, x, q, Q);

        /* keys = cache valida ++ nuove (l'update della cache lo fa il chiamante) */
        memcpy(kk, k_cache, (size_t)valid * (size_t)d * sizeof(float));
        memcpy(kk + (size_t)valid * (size_t)d, kn, (size_t)Q * (size_t)d * sizeof(float));
        memcpy(vv, v_cache, (size_t)valid * (size_t)d * sizeof(float));
        memcpy(vv + (size_t)valid * (size_t)d, vn, (size_t)Q * (size_t)d * sizeof(float));

        mynah_pos_emb(enc, K, pe); /* [2K-1, d], posizioni K-1..-(K-1) */
        matmul_wt(pe, L->relk_w, rk, P, d, d);

        for (int h = 0; h < H; h++) {
            const size_t ho = (size_t)h * (size_t)dk;
            for (int t = 0; t < Q; t++)
                for (int i = 0; i < dk; i++)
                    qb[(size_t)t * (size_t)dk + (size_t)i] =
                        q[(size_t)t * (size_t)d + ho + (size_t)i] + L->bias_v[ho + (size_t)i];
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Q, P, dk,
                        1.0f, qb, dk, rk + ho, d, 0.0f, bd, P);

            for (int t = 0; t < Q; t++)
                for (int i = 0; i < dk; i++)
                    qb[(size_t)t * (size_t)dk + (size_t)i] =
                        q[(size_t)t * (size_t)d + ho + (size_t)i] + L->bias_u[ho + (size_t)i];
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Q, K, dk,
                        1.0f, qb, dk, kk + ho, d, 0.0f, scores, K);

            for (int t = 0; t < Q; t++) {
                float *srow = scores + (size_t)t * (size_t)K;
                const float *brow = bd + (size_t)t * (size_t)P;
                /* rel_shift: p = (K-1) - (valid + t) + j, j in [0, K) */
                const int base = K - 1 - valid - t;
                float maxv = -3.0e38f;
                for (int j = 0; j < K; j++) {
                    srow[j] = (srow[j] + brow[base + j]) * scaling;
                    if (srow[j] > maxv) maxv = srow[j];
                }
                float sum = 0.0f;
                for (int j = 0; j < K; j++) { srow[j] = expf(srow[j] - maxv); sum += srow[j]; }
                const float inv = 1.0f / sum;
                for (int j = 0; j < K; j++) srow[j] *= inv;
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Q, dk, K,
                        1.0f, scores, K, vv + ho, d, 0.0f, qb, dk);
            for (int t = 0; t < Q; t++)
                memcpy(ctx + (size_t)t * (size_t)d + ho, qb + (size_t)t * (size_t)dk,
                       (size_t)dk * sizeof(float));
        }
        mynah_qmat_mul(&L->o_w, ctx, out, Q);
    }
done:
    free(qkv); free(keys); free(pe); free(rk); free(scores); free(bd); free(qb); free(ctx);
}

/* aggiorna la cache K/V di un layer con le nuove righe k/v del chunk */
static void update_kv_cache(float *cache, const float *fresh, int valid, int Q,
                            int left, int d) {
    const int total = valid + Q;
    const int keep = total < left ? total : left;
    const int from_old = keep - Q > 0 ? keep - Q : 0;      /* righe vecchie da tenere */
    const int drop_old = valid - from_old;                  /* righe vecchie da scartare */
    if (from_old > 0 && drop_old > 0)
        memmove(cache, cache + (size_t)drop_old * (size_t)d,
                (size_t)from_old * (size_t)d * sizeof(float));
    const int fresh_keep = keep - from_old;                 /* righe nuove da tenere (<= Q) */
    memcpy(cache + (size_t)from_old * (size_t)d,
           fresh + (size_t)(Q - fresh_keep) * (size_t)d,
           (size_t)fresh_keep * (size_t)d * sizeof(float));
}

/* Conv module streaming: cache [k-1, d] anteposta, aggiornata con le ultime k-1 righe. */
static void stream_conv_module(const mynah_encoder *enc, const mynah_enc_layer *L,
                               const float *x, float *out, int Q, float *cache) {
    const int d = enc->d_model, k = enc->conv_k;
    float *h2 = malloc((size_t)Q * 2u * (size_t)d * sizeof(float));
    float *gp = malloc((size_t)(Q + k - 1) * (size_t)d * sizeof(float));
    float *c = malloc((size_t)Q * (size_t)d * sizeof(float));
    if (!h2 || !gp || !c) { free(h2); free(gp); free(c); return; }

    mynah_qmat_mul(&L->pw1_w, x, h2, Q);
    memcpy(gp, cache, (size_t)(k - 1) * (size_t)d * sizeof(float));
    for (int t = 0; t < Q; t++) {
        const float *a = h2 + (size_t)t * 2u * (size_t)d;
        const float *b = a + d;
        float *o = gp + (size_t)(k - 1 + t) * (size_t)d;
        for (int i = 0; i < d; i++) o[i] = a[i] / (1.0f + expf(-b[i]));
    }
    /* aggiorna cache = ultime k-1 righe di gp */
    memcpy(cache, gp + (size_t)Q * (size_t)d, (size_t)(k - 1) * (size_t)d * sizeof(float));

    for (int t = 0; t < Q; t++) {
        float *o = c + (size_t)t * (size_t)d;
        memset(o, 0, (size_t)d * sizeof(float));
        for (int j = 0; j < k; j++) {
            const float *src = gp + (size_t)(t + j) * (size_t)d;
            for (int i = 0; i < d; i++) o[i] += L->dw_w[(size_t)i * (size_t)k + (size_t)j] * src[i];
        }
    }
    float *tmp = malloc((size_t)Q * (size_t)d * sizeof(float));
    if (tmp) {
        layer_norm_f(c, L->cnorm_w, L->cnorm_b, tmp, Q, d);
        silu_inplace(tmp, (size_t)Q * (size_t)d);
        mynah_qmat_mul(&L->pw2_w, tmp, out, Q);
        free(tmp);
    }
    free(h2); free(gp); free(c);
}

int mynah_enc_stream_step(mynah_enc_stream *es, const float *mel, int n_mel,
                          int n_mels, int prompt_id, int is_last, float *out) {
    const mynah_encoder *enc = es->enc;
    const int d = enc->d_model;

    float *x = malloc((size_t)(es->q + 2) * (size_t)d * sizeof(float));
    if (!x) return -1;
    const int Q = mynah_ss_stream_step(&enc->ss, &es->ss, mel, n_mel, n_mels, is_last, x);
    if (Q <= 0) { free(x); return -1; }

    const size_t nd = (size_t)Q * (size_t)d;
    float *tmp = malloc(nd * sizeof(float));
    float *tmp2 = malloc((size_t)Q * (size_t)enc->ffn_dim * sizeof(float));
    float *xn = malloc(nd * sizeof(float));
    float *kn = malloc(2 * nd * sizeof(float));
    if (!tmp || !tmp2 || !xn || !kn) { free(x); free(tmp); free(tmp2); free(xn); free(kn); return -1; }

    for (int li = 0; li < enc->n_layers; li++) {
        const mynah_enc_layer *L = &enc->layers[li];
        float *kc = es->k_cache + (size_t)li * (size_t)es->left * (size_t)d;
        float *vc = es->v_cache + (size_t)li * (size_t)es->left * (size_t)d;
        float *cc = es->conv_cache + (size_t)li * (size_t)(enc->conv_k - 1) * (size_t)d;

        /* ½ FFN1 */
        layer_norm_f(x, L->ln_ff1_w, L->ln_ff1_b, tmp, Q, d);
        mynah_qmat_mul(&L->ff1_w1, tmp, tmp2, Q);
        silu_inplace(tmp2, (size_t)Q * (size_t)enc->ffn_dim);
        mynah_qmat_mul(&L->ff1_w2, tmp2, tmp, Q);
        for (size_t i = 0; i < nd; i++) x[i] += 0.5f * tmp[i];

        /* MHSA con cache: servono k/v del chunk per aggiornare la cache DOPO */
        layer_norm_f(x, L->ln_att_w, L->ln_att_b, xn, Q, d);
        mynah_qmat_mul(&L->k_w, xn, kn, Q);
        mynah_qmat_mul(&L->v_w, xn, kn + nd, Q);
        stream_attention(enc, L, xn, kn, kn + nd, tmp, Q, kc, vc, es->cache_valid);
        update_kv_cache(kc, kn, es->cache_valid, Q, es->left, d);
        update_kv_cache(vc, kn + nd, es->cache_valid, Q, es->left, d);
        for (size_t i = 0; i < nd; i++) x[i] += tmp[i];

        /* Conv con cache */
        layer_norm_f(x, L->ln_conv_w, L->ln_conv_b, xn, Q, d);
        stream_conv_module(enc, L, xn, tmp, Q, cc);
        for (size_t i = 0; i < nd; i++) x[i] += tmp[i];

        /* ½ FFN2 + LN out */
        layer_norm_f(x, L->ln_ff2_w, L->ln_ff2_b, tmp, Q, d);
        mynah_qmat_mul(&L->ff2_w1, tmp, tmp2, Q);
        silu_inplace(tmp2, (size_t)Q * (size_t)enc->ffn_dim);
        mynah_qmat_mul(&L->ff2_w2, tmp2, tmp, Q);
        for (size_t i = 0; i < nd; i++) x[i] += 0.5f * tmp[i];
        layer_norm_f(x, L->ln_out_w, L->ln_out_b, xn, Q, d);
        memcpy(x, xn, nd * sizeof(float));
    }

    es->cache_valid = (es->cache_valid + Q < es->left) ? es->cache_valid + Q : es->left;

    mynah_encoder_post(enc, x, Q, prompt_id, out);
    free(x); free(tmp); free(tmp2); free(xn); free(kn);
    return Q;
}

/* --------------------------------------------------------- forward batched
 * Packing senza padding: x = concat dei frame di tutte le sequenze [ΣT, d].
 * FFN/LN girano sull'intero packed (weight-stationary); attention e conv,
 * che dipendono dalla causalità per-sequenza, iterano sui segmenti. */
static int encoder_layer_batch(const mynah_encoder *enc, int li, float *x,
                               const int *t_enc, int batch, float *const *pes,
                               int left, int right) {
    const mynah_enc_layer *L = &enc->layers[li];
    const int d = enc->d_model;
    int T_total = 0;
    for (int b = 0; b < batch; b++) T_total += t_enc[b];
    const size_t n = (size_t)T_total * (size_t)d;

    float *tmp = malloc(n * sizeof(float));
    float *tmp2 = malloc((size_t)T_total * (size_t)enc->ffn_dim * sizeof(float));
    float *xn = malloc(n * sizeof(float));
    if (!tmp || !tmp2 || !xn) { free(tmp); free(tmp2); free(xn); return -1; }

    /* ½ FFN1 — packed, fuso */
    layer_norm_f(x, L->ln_ff1_w, L->ln_ff1_b, tmp, T_total, d);
    mynah_qmat_ffn(&L->ff1_w1, &L->ff1_w2, tmp, tmp, T_total, tmp2);
    for (size_t i = 0; i < n; i++) x[i] += 0.5f * tmp[i];

    /* MHSA — per segmento */
    layer_norm_f(x, L->ln_att_w, L->ln_att_b, xn, T_total, d);
    for (int b = 0, off = 0; b < batch; off += t_enc[b], b++)
        attention(enc, L, xn + (size_t)off * (size_t)d, tmp + (size_t)off * (size_t)d,
                  t_enc[b], pes[b], left, right);
    for (size_t i = 0; i < n; i++) x[i] += tmp[i];

    /* Conv — per segmento (causale) */
    layer_norm_f(x, L->ln_conv_w, L->ln_conv_b, xn, T_total, d);
    for (int b = 0, off = 0; b < batch; off += t_enc[b], b++)
        conv_module(enc, L, xn + (size_t)off * (size_t)d, tmp + (size_t)off * (size_t)d,
                    t_enc[b]);
    for (size_t i = 0; i < n; i++) x[i] += tmp[i];

    /* ½ FFN2 + LN out — packed, fuso */
    layer_norm_f(x, L->ln_ff2_w, L->ln_ff2_b, tmp, T_total, d);
    mynah_qmat_ffn(&L->ff2_w1, &L->ff2_w2, tmp, tmp, T_total, tmp2);
    for (size_t i = 0; i < n; i++) x[i] += 0.5f * tmp[i];
    layer_norm_f(x, L->ln_out_w, L->ln_out_b, xn, T_total, d);
    memcpy(x, xn, n * sizeof(float));

    free(tmp); free(tmp2); free(xn);
    return 0;
}

int mynah_encoder_forward_batch(const mynah_encoder *enc, const float *const *feats,
                                const int *t_mel, int batch, int n_mels,
                                const int *prompt_ids, int left_ctx, int right_ctx,
                                float **outs, int *t_outs) {
    const int d = enc->d_model;
    float **seq = calloc((size_t)batch, sizeof(float *));
    float **pes = calloc((size_t)batch, sizeof(float *));
    if (!seq || !pes) { free(seq); free(pes); return -1; }

    int T_total = 0, rc = -1;
    for (int b = 0; b < batch; b++) {
        seq[b] = mynah_subsampling_forward(&enc->ss, feats[b], t_mel[b], n_mels, &t_outs[b]);
        if (!seq[b]) goto done;
        pes[b] = malloc((size_t)(2 * t_outs[b] - 1) * (size_t)d * sizeof(float));
        if (!pes[b]) goto done;
        mynah_pos_emb(enc, t_outs[b], pes[b]);
        T_total += t_outs[b];
    }

    float *x = malloc((size_t)T_total * (size_t)d * sizeof(float));
    if (!x) goto done;
    for (int b = 0, off = 0; b < batch; off += t_outs[b], b++) {
        memcpy(x + (size_t)off * (size_t)d, seq[b],
               (size_t)t_outs[b] * (size_t)d * sizeof(float));
        free(seq[b]);
        seq[b] = NULL;
    }

    for (int li = 0; li < enc->n_layers; li++)
        if (encoder_layer_batch(enc, li, x, t_outs, batch, pes, left_ctx, right_ctx) != 0) {
            free(x);
            goto done;
        }

    rc = 0;
    for (int b = 0, off = 0; b < batch; off += t_outs[b], b++) {
        outs[b] = malloc((size_t)t_outs[b] * (size_t)enc->d_out * sizeof(float));
        if (!outs[b]) { rc = -1; continue; }
        mynah_encoder_post(enc, x + (size_t)off * (size_t)d, t_outs[b], prompt_ids[b], outs[b]);
    }
    free(x);

done:
    for (int b = 0; b < batch; b++) { free(seq[b]); free(pes[b]); }
    free(seq); free(pes);
    return rc;
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
