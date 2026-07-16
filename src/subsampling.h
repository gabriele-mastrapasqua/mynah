/* Subsampling FastConformer dw_striding causale 8x (3 stadi stride-2):
 * conv_in piena + ReLU, poi 2x (depthwise + pointwise + ReLU), flatten
 * channel-major, linear -> d_model. Vedi docs/nemotron-arch.md. */
#ifndef MYNAH_SUBSAMPLING_H
#define MYNAH_SUBSAMPLING_H

#include "weights.h"

typedef struct {
    /* puntatori risolti UNA volta al load (decisione prior-art: niente lookup nel forward) */
    const mynah_tensor *conv_in_w, *conv_in_b;         /* [C,1,3,3], [C] */
    const mynah_tensor *dw_w[2], *dw_b[2];             /* [C,1,3,3], [C] */
    const mynah_tensor *pw_w[2], *pw_b[2];             /* [C,C,1,1], [C] */
    const mynah_tensor *lin_w, *lin_b;                 /* [d_model, C*F'], [d_model] */
    int channels;                                      /* 256 */
    int d_model;                                       /* 1024 */
} mynah_subsampling;

/* Risolve i tensori dal safetensors (prefisso HF "encoder.subsampling."). 0 = ok. */
int mynah_subsampling_init(mynah_subsampling *ss, const mynah_safetensors *st);

/* feats [T, n_mels] float32 (solo frame validi) -> out [T_out, d_model] (malloc).
 * Scrive *t_out. NULL su errore. Offline: pad causale tempo (2,1) per stadio. */
float *mynah_subsampling_forward(const mynah_subsampling *ss, const float *feats,
                                 int T, int n_mels, int *t_out);

/* ------------------------------------------------------- streaming (cache-aware)
 * Cache per stadio: l'ultimo time-frame dell'input (left_pad = k - stride = 1);
 * al primo chunk si antepone 1 zero extra (init_pad) => left effettivo 2, come
 * l'offline. Vedi docs/nemotron-arch.md (reference HF CausalConv2dCacheLayer). */
typedef struct {
    float *cache[3];        /* [C_in, F] ultimo frame input dello stadio */
    int cin[3], fdim[3];
    int first;
} mynah_ss_stream;

int mynah_ss_stream_init(mynah_ss_stream *sst, const mynah_subsampling *ss, int n_mels);
void mynah_ss_stream_free(mynah_ss_stream *sst);

/* mel chunk [n_mel, n_mels] (size ESATTA: primo = 1+8r, poi 8(r+1)) ->
 * out [q, d_model] con q = r+1 (buffer del caller). Ritorna q, -1 su errore. */
int mynah_ss_stream_step(const mynah_subsampling *ss, mynah_ss_stream *sst,
                         const float *mel, int n_mel, int n_mels, int is_last, float *out);

#endif
