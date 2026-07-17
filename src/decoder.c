#include "decoder.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MYNAH_BLAS_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

int mynah_decoder_init(mynah_decoder *dec, const mynah_safetensors *st,
                       int blank, int max_symbols, int quantize) {
    memset(dec, 0, sizeof(*dec));
    const mynah_tensor *emb = mynah_st_get(st, "decoder.embedding.weight");
    const mynah_tensor *proj_w = mynah_st_get(st, "decoder.decoder_projector.weight");
    const mynah_tensor *proj_b = mynah_st_get(st, "decoder.decoder_projector.bias");
    const mynah_tensor *head_b = mynah_st_get(st, "joint.head.bias");
    if (!emb || !proj_w || !proj_b || !head_b) return -1;

    dec->embedding = (const float *)emb->data;
    dec->proj_w = (const float *)proj_w->data;
    dec->proj_b = (const float *)proj_b->data;
    dec->head_b = (const float *)head_b->data;
    dec->vocab = (int)emb->shape[0];
    dec->hidden = (int)emb->shape[1];
    if (mynah_qmat_init_st(&dec->head, st, "joint.head.weight", quantize) != 0) return -1;
    dec->blank = blank;
    dec->max_symbols = max_symbols;

    int n = 0;
    for (; n < MYNAH_MAX_PRED_LAYERS; n++) {
        char name[64];
        snprintf(name, sizeof(name), "decoder.lstm.weight_ih_l%d", n);
        const mynah_tensor *w = mynah_st_get(st, name);
        if (!w) break;
        dec->w_ih[n] = (const float *)w->data;
        snprintf(name, sizeof(name), "decoder.lstm.weight_hh_l%d", n);
        dec->w_hh[n] = (const float *)mynah_st_get(st, name)->data;
        snprintf(name, sizeof(name), "decoder.lstm.bias_ih_l%d", n);
        dec->b_ih[n] = (const float *)mynah_st_get(st, name)->data;
        snprintf(name, sizeof(name), "decoder.lstm.bias_hh_l%d", n);
        dec->b_hh[n] = (const float *)mynah_st_get(st, name)->data;
    }
    dec->n_layers = n;
    return (n > 0 && dec->hidden <= 1024) ? 0 : -1;
}

void mynah_dec_state_reset(const mynah_decoder *dec, mynah_dec_state *s) {
    memset(s, 0, sizeof(*s));
    (void)dec;
    s->last_token = -1;
}

static inline float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

/* Un passo LSTM stacked + projector: input = embedding[token]. Aggiorna h/c e s->g. */
static void pred_step(const mynah_decoder *dec, mynah_dec_state *s, int token) {
    const int H = dec->hidden;
    const float *x = dec->embedding + (size_t)token * (size_t)H;
    float z[4 * 1024];

    for (int l = 0; l < dec->n_layers; l++) {
        for (int i = 0; i < 4 * H; i++) z[i] = dec->b_ih[l][i] + dec->b_hh[l][i];
        cblas_sgemv(CblasRowMajor, CblasNoTrans, 4 * H, H, 1.0f, dec->w_ih[l], H,
                    x, 1, 1.0f, z, 1);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, 4 * H, H, 1.0f, dec->w_hh[l], H,
                    s->h[l], 1, 1.0f, z, 1);
        for (int i = 0; i < H; i++) {
            const float ig = sigmoid_f(z[i]);
            const float fg = sigmoid_f(z[H + i]);
            const float gg = tanhf(z[2 * H + i]);
            const float og = sigmoid_f(z[3 * H + i]);
            s->c[l][i] = fg * s->c[l][i] + ig * gg;
            s->h[l][i] = og * tanhf(s->c[l][i]);
        }
        x = s->h[l];
    }
    /* decoder_projector */
    memcpy(s->g, dec->proj_b, (size_t)H * sizeof(float));
    cblas_sgemv(CblasRowMajor, CblasNoTrans, H, H, 1.0f, dec->proj_w, H,
                s->h[dec->n_layers - 1], 1, 1.0f, s->g, 1);
    s->last_token = token;
}

/* Greedy a BLOCCHI: g cambia solo su emissione non-blank, quindi i frame di una
 * run di blank condividono la stessa g e il joint head (il costo dominante:
 * matvec V x H per frame) si batcha in UNA GEMM [B, V] — stessa matrice pesi
 * letta una volta per blocco invece che per frame. Il blocco è adattivo
 * (raddoppia sulle run di blank, riparte corto dopo un'emissione) per non
 * sprecare righe oltre il primo frame che emette. Semantica identica al loop
 * per-frame (l'inner loop su un frame che emette resta scalare). */
#define DEC_BMAX 32

static int argmax_bias(const float *lg, const float *bias, float *tmp, int V) {
    for (int k = 0; k < V; k++) tmp[k] = lg[k] + bias[k];
    int best = 0;
    for (int k = 1; k < V; k++)
        if (tmp[k] > tmp[best]) best = k;
    return best;
}

int mynah_greedy_decode(const mynah_decoder *dec, mynah_dec_state *s,
                        const float *enc, int T, int *tokens, int cap) {
    const int H = dec->hidden, V = dec->vocab;
    float joint[1024];
    float *jin = malloc((size_t)DEC_BMAX * (size_t)H * sizeof(float));
    float *logits = malloc((size_t)DEC_BMAX * (size_t)V * sizeof(float));
    float *lb = malloc((size_t)V * sizeof(float));
    if (!jin || !logits || !lb) { free(jin); free(logits); free(lb); return 0; }
    int n_out = 0;

    /* Head come matrice f32 per la GEMM BLAS diretta (CPU: deterministica tra
     * backend — il decode non passa MAI dalla GPU). Se quantizzata e T grande
     * (offline) si dequantizza UNA volta per chiamata: 33 MB letti/scritti una
     * volta contro la matrice int8 riletta per ogni frame. Per T piccolo
     * (chunk streaming) resta il dot quantizzato per-frame di qmat. */
    const float *W = dec->head.qtype == MYNAH_Q_F32 ? dec->head.f32 : NULL;
    float *wd = NULL;
    if (!W && T > 16) {
        wd = malloc((size_t)V * (size_t)H * sizeof(float));
        if (wd) { mynah_qmat_dequant(&dec->head, wd); W = wd; }
    }

    if (s->last_token < 0) pred_step(dec, s, dec->blank); /* SOS = blank, stato zero */

    int t = 0, B = 4;
    while (t < T) {
        const int Bc = (T - t) < B ? (T - t) : B;
        for (int b = 0; b < Bc; b++) {
            const float *e = enc + (size_t)(t + b) * (size_t)H;
            float *ji = jin + (size_t)b * (size_t)H;
            for (int i = 0; i < H; i++) {
                const float v = e[i] + s->g[i];
                ji[i] = v > 0.0f ? v : 0.0f;               /* ReLU */
            }
        }
        if (W)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Bc, V, H,
                        1.0f, jin, H, W, H, 0.0f, logits, V);
        else
            mynah_qmat_mul(&dec->head, jin, logits, Bc);

        int first = -1, best = -1;
        for (int b = 0; b < Bc; b++) {
            const int am = argmax_bias(logits + (size_t)b * (size_t)V, dec->head_b, lb, V);
            if (am != dec->blank) { first = b; best = am; break; }
        }
        if (first < 0) {                                   /* run di soli blank */
            t += Bc;
            if (B < DEC_BMAX) B *= 2;
            continue;
        }

        /* frame t+first emette: inner loop scalare (la prima emissione è già nota) */
        t += first;
        const float *e = enc + (size_t)t * (size_t)H;
        for (int emitted = 0; emitted < dec->max_symbols; emitted++) {
            if (emitted > 0) {                             /* iterazione 0: dal batch */
                for (int i = 0; i < H; i++) {
                    const float v = e[i] + s->g[i];
                    joint[i] = v > 0.0f ? v : 0.0f;
                }
                if (W)
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, V, H,
                                1.0f, joint, H, W, H, 0.0f, logits, V);
                else
                    mynah_qmat_mul(&dec->head, joint, logits, 1);
                best = argmax_bias(logits, dec->head_b, lb, V);
                if (best == dec->blank) break;
            }
            if (n_out < cap) tokens[n_out++] = best;
            pred_step(dec, s, best);                       /* stato avanza solo su emit */
        }
        t++;
        B = 4;                                             /* riparte corto dopo un emit */
    }
    free(jin); free(logits); free(lb); free(wd);
    return n_out;
}
