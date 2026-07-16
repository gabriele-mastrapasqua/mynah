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

#endif
