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

/* ------------------------------------------------------------- mel streaming
 * Incrementale, identico bit-a-bit all'offline sui frame validi (possibile solo
 * perché Nemotron non normalizza — vedi docs/prior-art.md §A.7).
 * Il frame t copre i campioni [t*hop-256, t*hop+256): è pronto quando il segnale
 * arriva a t*hop+256; a finish() si emettono i frame residui (< S/hop) leggendo
 * gli zeri del pad destro. */
typedef struct {
    const mynah_feat_cfg *cfg;
    double *buf;            /* finestra scorrevole di segnale preemfatizzato   */
    size_t buf_len, buf_cap;
    size_t base;            /* indice assoluto del campione buf[0]             */
    size_t total;           /* campioni totali visti                           */
    double *win;            /* finestra Hann center-paddata a n_fft (precomp.) */
    float last_raw;         /* carry per la preemphasis tra feed               */
    long next_frame;        /* prossimo frame mel da emettere                  */
    int finished;
} mynah_mel_stream;

int mynah_mel_stream_init(mynah_mel_stream *ms, const mynah_feat_cfg *cfg);
void mynah_mel_stream_free(mynah_mel_stream *ms);

/* Aggiunge campioni; scrive in *out (capienza cap_frames righe da n_mels) i frame
 * mel diventati pronti. Ritorna il numero di frame scritti. */
int mynah_mel_stream_feed(mynah_mel_stream *ms, const float *audio, size_t n,
                          float *out, int cap_frames);

/* Fine stream: emette i frame residui fino a S/hop (esclusi). */
int mynah_mel_stream_finish(mynah_mel_stream *ms, float *out, int cap_frames);

#endif
