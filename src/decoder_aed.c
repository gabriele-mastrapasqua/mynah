#include "decoder_aed.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MYNAH_BLAS_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

/* y = W x + b, W [n, k] row-major */
static void mv(const float *w, const float *b, const float *x, float *y, int n, int k) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, n, k, 1.0f, w, k, x, 1, 0.0f, y, 1);
    if (b)
        for (int i = 0; i < n; i++) y[i] += b[i];
}

static void ln(const float *x, const float *w, const float *b, float *y, int d) {
    double mu = 0.0, var = 0.0;
    for (int i = 0; i < d; i++) mu += x[i];
    mu /= d;
    for (int i = 0; i < d; i++) { const double c = x[i] - mu; var += c * c; }
    const float inv = (float)(1.0 / sqrt(var / d + 1e-5));
    for (int i = 0; i < d; i++) y[i] = (float)((x[i] - mu) * inv) * w[i] + b[i];
}

static void softmax_inplace(float *s, int n) {
    float m = s[0];
    for (int i = 1; i < n; i++) if (s[i] > m) m = s[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { s[i] = expf(s[i] - m); sum += s[i]; }
    const float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) s[i] *= inv;
}

#define T_(st, name) ((const float *)(mynah_st_get((st), (name)) ? mynah_st_get((st), (name))->data : NULL))

int mynah_aed_init(mynah_aed *a, const mynah_safetensors *st,
                   int n_layers, int n_heads, int max_seq, int max_gen_delta) {
    memset(a, 0, sizeof(*a));
    const mynah_tensor *emb = mynah_st_get(st, "aed.embedding.weight");
    if (!emb) return -1;
    a->emb = (const float *)emb->data;
    a->vocab = (int)emb->shape[0];
    a->d = (int)emb->shape[1];
    a->n_layers = n_layers;
    a->n_heads = n_heads;
    a->max_seq = max_seq;
    a->max_gen_delta = max_gen_delta;
    a->pos = T_(st, "aed.pos_enc");
    a->embln_w = T_(st, "aed.emb_norm.weight");
    a->embln_b = T_(st, "aed.emb_norm.bias");
    a->fin_w = T_(st, "aed.final_norm.weight");
    a->fin_b = T_(st, "aed.final_norm.bias");
    a->head_w = T_(st, "aed.head.weight");
    a->head_b = T_(st, "aed.head.bias");
    const mynah_tensor *proj = mynah_st_get(st, "enc_dec_proj.weight");
    if (proj) {                       /* assente quando enc hidden == dec hidden */
        a->proj_w = (const float *)proj->data;
        a->proj_b = T_(st, "enc_dec_proj.bias");
        a->d_enc = (int)proj->shape[1];
    } else {
        a->d_enc = a->d;
    }
    const mynah_tensor *ff1 = mynah_st_get(st, "aed.layers.0.ffn.linear1.weight");
    if (!a->pos || !a->embln_w || !a->fin_w || !a->head_w || !ff1) return -1;
    a->ffn = (int)ff1->shape[0];

    a->layers = calloc((size_t)n_layers, sizeof(*a->layers));
    if (!a->layers) return -1;
    for (int li = 0; li < n_layers; li++) {
        mynah_aed_layer *L = &a->layers[li];
        char n[96];
#define G(field, suffix) \
        snprintf(n, sizeof(n), "aed.layers.%d." suffix, li); \
        L->field = T_(st, n); \
        if (!L->field) { fprintf(stderr, "mynah: aed manca %s\n", n); return -1; }
        G(ln_self_w, "ln_self.weight")   G(ln_self_b, "ln_self.bias")
        G(sq_w, "self_attn.q_proj.weight") G(sq_b, "self_attn.q_proj.bias")
        G(sk_w, "self_attn.k_proj.weight") G(sk_b, "self_attn.k_proj.bias")
        G(sv_w, "self_attn.v_proj.weight") G(sv_b, "self_attn.v_proj.bias")
        G(so_w, "self_attn.o_proj.weight") G(so_b, "self_attn.o_proj.bias")
        G(ln_cross_w, "ln_cross.weight") G(ln_cross_b, "ln_cross.bias")
        G(cq_w, "cross_attn.q_proj.weight") G(cq_b, "cross_attn.q_proj.bias")
        G(ck_w, "cross_attn.k_proj.weight") G(ck_b, "cross_attn.k_proj.bias")
        G(cv_w, "cross_attn.v_proj.weight") G(cv_b, "cross_attn.v_proj.bias")
        G(co_w, "cross_attn.o_proj.weight") G(co_b, "cross_attn.o_proj.bias")
        G(ln_ffn_w, "ln_ffn.weight")     G(ln_ffn_b, "ln_ffn.bias")
        G(ff1_w, "ffn.linear1.weight")   G(ff1_b, "ffn.linear1.bias")
        G(ff2_w, "ffn.linear2.weight")   G(ff2_b, "ffn.linear2.bias")
#undef G
    }
    return 0;
}

void mynah_aed_free(mynah_aed *a) {
    free(a->layers);
    a->layers = NULL;
}

/* attention di UNA query su n_kv chiavi/valori [n_kv, d] (righe contigue),
 * per-head, score scalati 1/sqrt(dk). out [d]. scores: scratch [n_kv]. */
static void attend(const float *q, const float *K, const float *V, int n_kv,
                   int H, int dk, float *scores, float *out) {
    const int d = H * dk;
    const float scale = 1.0f / sqrtf((float)dk);
    for (int h = 0; h < H; h++) {
        const float *qh = q + h * dk;
        cblas_sgemv(CblasRowMajor, CblasNoTrans, n_kv, dk, scale, K + h * dk, d,
                    qh, 1, 0.0f, scores, 1);
        softmax_inplace(scores, n_kv);
        cblas_sgemv(CblasRowMajor, CblasTrans, n_kv, dk, 1.0f, V + h * dk, d,
                    scores, 1, 0.0f, out + h * dk, 1);
    }
}

int mynah_aed_decode(const mynah_aed *a, const float *enc, int T,
                     const int *prompt, int n_prompt, int eos,
                     int *tokens, int cap) {
    const int d = a->d, H = a->n_heads, dk = d / H, nl = a->n_layers;
    int max_len = n_prompt + T + a->max_gen_delta;
    if (max_len > a->max_seq) max_len = a->max_seq;

    /* proiezione encoder -> spazio decoder */
    float *encp;
    if (a->proj_w) {
        encp = malloc((size_t)T * (size_t)d * sizeof(float));
        if (!encp) return -1;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, d, a->d_enc,
                    1.0f, enc, a->d_enc, a->proj_w, a->d_enc, 0.0f, encp, d);
        if (a->proj_b)
            for (int t = 0; t < T; t++)
                for (int i = 0; i < d; i++) encp[(size_t)t * d + i] += a->proj_b[i];
    } else {
        encp = (float *)enc;
    }

    /* cross K/V per layer (una volta) + cache self K/V + scratch */
    const size_t td = (size_t)T * (size_t)d, md = (size_t)max_len * (size_t)d;
    float *ckv = malloc(2u * (size_t)nl * td * sizeof(float));
    float *skv = malloc(2u * (size_t)nl * md * sizeof(float));
    float *scr = malloc((size_t)(6 * d + a->ffn + (T > max_len ? T : max_len)) * sizeof(float));
    float *logits = malloc((size_t)a->vocab * sizeof(float));
    int rc = -1, n_out = 0;
    if (!ckv || !skv || !scr || !logits) goto done;
    float *x = scr, *xn = scr + d, *q = scr + 2 * d, *att = scr + 3 * d,
          *tmp = scr + 4 * d, *kv = scr + 5 * d, *ff = scr + 6 * d,
          *scores = scr + 6 * d + a->ffn;

    for (int li = 0; li < nl; li++) {
        const mynah_aed_layer *L = &a->layers[li];
        float *CK = ckv + (size_t)(2 * li) * td, *CV = CK + td;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, d, d,
                    1.0f, encp, d, L->ck_w, d, 0.0f, CK, d);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T, d, d,
                    1.0f, encp, d, L->cv_w, d, 0.0f, CV, d);
        for (int t = 0; t < T; t++)
            for (int i = 0; i < d; i++) {
                CK[(size_t)t * d + i] += L->ck_b[i];
                CV[(size_t)t * d + i] += L->cv_b[i];
            }
    }

    int cur = prompt[0];
    for (int p = 0; p < max_len; p++) {
        /* embedding + pos enc (buffer dal ckpt) + LN */
        const float *e = a->emb + (size_t)cur * d, *pe = a->pos + (size_t)p * d;
        for (int i = 0; i < d; i++) x[i] = e[i] + pe[i];
        ln(x, a->embln_w, a->embln_b, x, d);

        for (int li = 0; li < nl; li++) {
            const mynah_aed_layer *L = &a->layers[li];
            float *SK = skv + (size_t)(2 * li) * md, *SV = SK + md;
            /* self-attention causale: nuova k/v in cache, attention su [0..p] */
            ln(x, L->ln_self_w, L->ln_self_b, xn, d);
            mv(L->sq_w, L->sq_b, xn, q, d, d);
            mv(L->sk_w, L->sk_b, xn, SK + (size_t)p * d, d, d);
            mv(L->sv_w, L->sv_b, xn, SV + (size_t)p * d, d, d);
            attend(q, SK, SV, p + 1, H, dk, scores, att);
            mv(L->so_w, L->so_b, att, tmp, d, d);
            for (int i = 0; i < d; i++) x[i] += tmp[i];
            /* cross-attention sull'encoder proiettato */
            ln(x, L->ln_cross_w, L->ln_cross_b, xn, d);
            mv(L->cq_w, L->cq_b, xn, q, d, d);
            attend(q, ckv + (size_t)(2 * li) * td, ckv + (size_t)(2 * li) * td + td,
                   T, H, dk, scores, att);
            mv(L->co_w, L->co_b, att, tmp, d, d);
            for (int i = 0; i < d; i++) x[i] += tmp[i];
            /* FFN ReLU */
            ln(x, L->ln_ffn_w, L->ln_ffn_b, xn, d);
            mv(L->ff1_w, L->ff1_b, xn, ff, a->ffn, d);
            for (int i = 0; i < a->ffn; i++) if (ff[i] < 0.0f) ff[i] = 0.0f;
            mv(L->ff2_w, L->ff2_b, ff, tmp, d, a->ffn);
            for (int i = 0; i < d; i++) x[i] += tmp[i];
        }

        if (p + 1 < n_prompt) {          /* ancora nel prompt: niente sampling */
            cur = prompt[p + 1];
            continue;
        }
        ln(x, a->fin_w, a->fin_b, kv, d);
        mv(a->head_w, a->head_b, kv, logits, a->vocab, d);
        int best = 0;
        for (int i = 1; i < a->vocab; i++) if (logits[i] > logits[best]) best = i;
        if (best == eos) break;
        if (n_out >= cap) break;
        tokens[n_out++] = best;
        cur = best;
    }
    rc = n_out;

done:
    if (encp != enc) free(encp);
    free(ckv); free(skv); free(scr); free(logits);
    return rc;
}
