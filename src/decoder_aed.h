/* Decoder AED (Canary): Transformer pre-LN con self-attention causale (KV cache)
 * e cross-attention sull'encoder out proiettato. Greedy autoregressivo col prompt
 * canary2; semantica in docs/canary-arch.md, riferimento oracle/model.py. */
#ifndef MYNAH_DECODER_AED_H
#define MYNAH_DECODER_AED_H

#include "weights.h"

typedef struct {
    const float *ln_self_w, *ln_self_b;
    const float *sq_w, *sq_b, *sk_w, *sk_b, *sv_w, *sv_b, *so_w, *so_b;
    const float *ln_cross_w, *ln_cross_b;
    const float *cq_w, *cq_b, *ck_w, *ck_b, *cv_w, *cv_b, *co_w, *co_b;
    const float *ln_ffn_w, *ln_ffn_b;
    const float *ff1_w, *ff1_b, *ff2_w, *ff2_b;
} mynah_aed_layer;

typedef struct {
    int n_layers, d, n_heads, ffn, vocab, max_seq, max_gen_delta;
    int d_enc;                       /* input di enc_dec_proj (= d se identity) */
    const float *proj_w, *proj_b;    /* enc_dec_proj (NULL = identity)          */
    const float *emb;                /* [vocab, d]                              */
    const float *pos;                /* [max_seq, d] buffer dal ckpt (già /√d)  */
    const float *embln_w, *embln_b;
    const float *fin_w, *fin_b;      /* final_layer_norm                        */
    const float *head_w, *head_b;    /* [vocab, d]                              */
    mynah_aed_layer *layers;
} mynah_aed;

/* 0 = ok, -1 = pesi AED assenti (modello non-AED). */
int mynah_aed_init(mynah_aed *a, const mynah_safetensors *st,
                   int n_layers, int n_heads, int max_seq, int max_gen_delta);
void mynah_aed_free(mynah_aed *a);

/* Greedy: enc [T, d_enc] -> token generati (senza prompt né EOS) in tokens[cap].
 * Ritorna il numero di token, -1 su errore. */
int mynah_aed_decode(const mynah_aed *a, const float *enc, int T,
                     const int *prompt, int n_prompt, int eos,
                     int *tokens, int cap);

#endif
