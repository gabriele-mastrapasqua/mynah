/* Encoder FastConformer cache-aware (path offline-chunked; streaming in M1.3).
 * Blocco pre-norm macaron: ½FFN -> MHSA rel-pos -> Conv -> ½FFN -> LN out.
 * Riferimento numerico: tools/oracle/model.py + docs/nemotron-arch.md.
 * Tutte le dimensioni derivate dalle shape dei tensori (config-driven). */
#ifndef MYNAH_ENCODER_H
#define MYNAH_ENCODER_H

#include "subsampling.h"
#include "weights.h"

typedef struct {
    const float *ln_ff1_w, *ln_ff1_b, *ff1_w1, *ff1_w2;
    const float *ln_att_w, *ln_att_b;
    const float *q_w, *k_w, *v_w, *o_w, *relk_w, *bias_u, *bias_v;
    const float *ln_conv_w, *ln_conv_b, *pw1_w, *dw_w, *cnorm_w, *cnorm_b, *pw2_w;
    const float *ln_ff2_w, *ln_ff2_b, *ff2_w1, *ff2_w2;
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

int mynah_encoder_init(mynah_encoder *enc, const mynah_safetensors *st);
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

#endif
