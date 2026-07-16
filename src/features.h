/* Log-mel config-driven — replica del feature extractor Nemotron
 * (vedi docs/nemotron-arch.md e tools/oracle/features.py):
 * preemph -> center pad costante -> Hann simmetrica paddata a n_fft -> |rfft|^2
 * -> filterbank mel -> log(x + guard). Nessuna normalizzazione (normalize "NA"). */
#ifndef MYNAH_FEATURES_H
#define MYNAH_FEATURES_H

#include <stddef.h>

typedef struct {
    int sample_rate;       /* 16000 */
    int n_mels;            /* 128 */
    int n_fft;             /* 512 */
    int win_length;        /* 400 */
    int hop_length;        /* 160 */
    double preemphasis;    /* 0.97 */
    double log_zero_guard; /* 2^-24 */
    const float *mel_fb;   /* [n_fft/2+1, n_mels] da mel_filters.safetensors */
    const float *window;   /* [win_length] */
} mynah_feat_cfg;

/* Calcola il log-mel offline. Ritorna feats [T, n_mels] float32 (malloc, caller
 * free), scrive *n_frames (= 1 + S/hop) e *valid_frames (= S/hop; i frame oltre
 * il valid sono azzerati, come nel feature extractor HF). NULL su errore. */
float *mynah_log_mel(const mynah_feat_cfg *cfg, const float *audio, size_t n_samples,
                     int *n_frames, int *valid_frames);

#endif
