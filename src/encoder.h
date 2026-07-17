/* Encoder FastConformer cache-aware (path offline-chunked; streaming in M1.3).
 * Blocco pre-norm macaron: ½FFN -> MHSA rel-pos -> Conv -> ½FFN -> LN out.
 * Riferimento numerico: tools/oracle/model.py + docs/nemotron-arch.md.
 * Tutte le dimensioni derivate dalle shape dei tensori (config-driven). */
#ifndef MYNAH_ENCODER_H
#define MYNAH_ENCODER_H

#include "qmat.h"
#include "subsampling.h"
#include "weights.h"

typedef struct {
    const float *ln_ff1_w, *ln_ff1_b;
    mynah_qmat ff1_w1, ff1_w2;
    const float *ln_att_w, *ln_att_b;
    mynah_qmat q_w, k_w, v_w, o_w;
    const float *relk_w;   /* f32 sempre: usato con T=2L-1 grande a ogni chunk */
    const float *bias_u, *bias_v;
    const float *ln_conv_w, *ln_conv_b;
    mynah_qmat pw1_w, pw2_w;
    const float *dw_w, *cnorm_w, *cnorm_b;
    const float *ln_ff2_w, *ln_ff2_b;
    mynah_qmat ff2_w1, ff2_w2;
    const float *ln_out_w, *ln_out_b;
} mynah_enc_layer;

typedef struct {
    mynah_subsampling ss;
    mynah_enc_layer *layers;
    int n_layers, d_model, n_heads, d_head, ffn_dim, conv_k;
    /* prompt (post-encoder) + projector verso il joint space */
    const float *prompt_l1_w, *prompt_l1_b, *prompt_l2_w, *prompt_l2_b;
    const float *encproj_w, *encproj_b;
    int num_prompts, prompt_inter, d_out;
} mynah_encoder;

/* quantize != 0: INT8 per-riga sui grandi linear (FFN, attn q/k/v/o, pointwise
 * conv). Costruita al load dal f32; ~2.4x meno memoria residente. */
int mynah_encoder_init(mynah_encoder *enc, const mynah_safetensors *st, int quantize);
void mynah_encoder_free(mynah_encoder *enc);

/* Positional embedding rel [2T-1, d_model] (interleaved sin/cos, pos T-1..-(T-1)).
 * Buffer allocato dal caller: (2T-1)*d_model float. */
void mynah_pos_emb(const mynah_encoder *enc, int T, float *pe);

/* Un blocco conformer in-place su x [T, d_model]. left/right = att context. */
int mynah_encoder_layer(const mynah_encoder *enc, int li, float *x, int T,
                        const float *pe, int left_ctx, int right_ctx);

/* Prompt one-hot + prompt_projector + encoder_projector: x [T,d_model] -> out [T,d_out]. */
void mynah_encoder_post(const mynah_encoder *enc, const float *x, int T, int prompt_id,
                        float *out);

/* Forward completo offline: feats [T_mel, n_mels] validi -> out [T_enc, d_out] (malloc). */
float *mynah_encoder_forward(const mynah_encoder *enc, const float *feats, int t_mel,
                             int n_mels, int prompt_id, int left_ctx, int right_ctx,
                             int *t_out);

/* Forward batched weight-stationary (lunghezze variabili, packing senza padding):
 * le GEMM per-frame (FFN, proiezioni — >95% dei FLOP) girano su [ΣT, d] leggendo i
 * pesi UNA volta; attention e conv (per-sequenza) iterano sui segmenti.
 * outs[b] riceve un buffer malloc [t_outs[b], d_out] (caller free). 0 = ok. */
int mynah_encoder_forward_batch(const mynah_encoder *enc, const float *const *feats,
                                const int *t_mel, int batch, int n_mels,
                                const int *prompt_ids, int left_ctx, int right_ctx,
                                float **outs, int *t_outs);

/* --------------------------------------------------------- streaming cache-aware
 * Ogni chunk mel (primo: 1+8r frame, poi 8(r+1)) produce q = r+1 frame encoder,
 * che coincidono con UN chunk della griglia chunked_limited: il left context in
 * cache (56 frame, sempre divisibile per r+1) è ESATTAMENTE il contesto ammesso
 * => attention piena su [cache valida + chunk], niente mask. Vedi prior-art §A. */
typedef struct {
    const mynah_encoder *enc;
    mynah_ss_stream ss;
    float *k_cache, *v_cache;   /* [n_layers, left, d_model] */
    float *conv_cache;          /* [n_layers, conv_k-1, d_model] */
    int left, right, q;         /* q = right+1 frame encoder per chunk */
    int cache_valid;            /* frame validi nella cache K/V (0..left) */
    /* scratch del percorso caldo, UN malloc all'init (zero malloc per chunk):
     * puntatori ritagliati da scr. Dimensionati su Qmax = q+2, Kmax = left+Qmax. */
    float *scr;
    float *sx, *stmp, *stmp2, *sxn, *skn;                    /* step */
    float *sa_pe, *sa_q, *sa_keys, *sa_rk, *sa_sc, *sa_bd,   /* attention */
          *sa_qb, *sa_ctx;
    float *sc_h2, *sc_gp, *sc_c, *sc_t;                      /* conv module */
} mynah_enc_stream;

int mynah_enc_stream_init(mynah_enc_stream *es, const mynah_encoder *enc,
                          int left_ctx, int right_ctx, int n_mels);
void mynah_enc_stream_free(mynah_enc_stream *es);

/* Frame mel richiesti dal prossimo chunk (primo: 1+8r, poi 8(r+1)). */
int mynah_enc_stream_need(const mynah_enc_stream *es);

/* mel chunk [n_mel, n_mels] esatto -> out [q, d_out] (buffer caller >= q*d_out).
 * Ritorna il numero di frame encoder prodotti, -1 su errore. */
int mynah_enc_stream_step(mynah_enc_stream *es, const float *mel, int n_mel,
                          int n_mels, int prompt_id, int is_last, float *out);

#endif
