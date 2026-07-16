#include "features.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* FFT complessa radix-2 iterativa in double (n_fft potenza di 2).
 * Dimensioni piccole (512): la chiarezza vince, e il double compra la parità
 * con l'oracolo (vedi docs/prior-art.md — stessa scelta di parakeet.cpp). */
static void fft_radix2(double *re, double *im, int n) {
    /* bit reversal */
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            double tr = re[i]; re[i] = re[j]; re[j] = tr;
            double ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = -2.0 * M_PI / (double)len;
        double wr = cos(ang), wi = sin(ang);
        for (int i = 0; i < n; i += len) {
            double cr = 1.0, ci = 0.0;
            for (int k = 0; k < len / 2; k++) {
                int a = i + k, b = i + k + len / 2;
                double xr = re[b] * cr - im[b] * ci;
                double xi = re[b] * ci + im[b] * cr;
                re[b] = re[a] - xr; im[b] = im[a] - xi;
                re[a] += xr;        im[a] += xi;
                double ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
        }
    }
}

float *mynah_log_mel(const mynah_feat_cfg *cfg, const float *audio, size_t n_samples,
                     int *n_frames, int *valid_frames) {
    const int n_fft = cfg->n_fft, hop = cfg->hop_length, n_mels = cfg->n_mels;
    const int n_bins = n_fft / 2 + 1;
    const int pad = n_fft / 2;
    const size_t S = n_samples;

    const int T = 1 + (int)(S / (size_t)hop);
    const int valid = (int)(S / (size_t)hop);

    /* preemphasis + center pad in un buffer double */
    double *y = calloc(S + 2 * (size_t)pad, sizeof(double));
    if (!y) return NULL;
    if (S > 0) y[pad] = audio[0];
    for (size_t i = 1; i < S; i++)
        y[pad + i] = (double)audio[i] - cfg->preemphasis * (double)audio[i - 1];

    /* finestra center-paddata a n_fft */
    double *win = calloc((size_t)n_fft, sizeof(double));
    int off = (n_fft - cfg->win_length) / 2;
    for (int i = 0; i < cfg->win_length; i++) win[off + i] = (double)cfg->window[i];

    float *feats = calloc((size_t)T * (size_t)n_mels, sizeof(float));
    double *re = malloc((size_t)n_fft * sizeof(double));
    double *im = malloc((size_t)n_fft * sizeof(double));
    double *power = malloc((size_t)n_bins * sizeof(double));
    if (!feats || !re || !im || !power) {
        free(y); free(win); free(feats); free(re); free(im); free(power);
        return NULL;
    }

    for (int t = 0; t < valid; t++) { /* i frame >= valid restano a zero */
        const double *frame = y + (size_t)t * (size_t)hop;
        for (int i = 0; i < n_fft; i++) { re[i] = frame[i] * win[i]; im[i] = 0.0; }
        fft_radix2(re, im, n_fft);
        for (int b = 0; b < n_bins; b++) power[b] = re[b] * re[b] + im[b] * im[b];

        float *row = feats + (size_t)t * (size_t)n_mels;
        for (int m = 0; m < n_mels; m++) {
            double acc = 0.0;
            for (int b = 0; b < n_bins; b++)
                acc += power[b] * (double)cfg->mel_fb[(size_t)b * (size_t)n_mels + (size_t)m];
            row[m] = (float)log(acc + cfg->log_zero_guard);
        }
    }

    free(y); free(win); free(re); free(im); free(power);
    *n_frames = T;
    *valid_frames = valid;
    return feats;
}

/* ------------------------------------------------------------- mel streaming */

/* Calcola un singolo frame mel dal segnale preemfatizzato (coordinate campione,
 * pad sinistro/destro letti come zero). Condiviso concettualmente con l'offline:
 * frame t = finestra [t*hop - n_fft/2, t*hop + n_fft/2). */
static void mel_one_frame(const mynah_feat_cfg *cfg, const double *sig, size_t n_sig,
                          int t, float *row) {
    const int n_fft = cfg->n_fft, n_bins = n_fft / 2 + 1, n_mels = cfg->n_mels;
    const long start = (long)t * cfg->hop_length - n_fft / 2;

    double re[4096], im[4096], power[2049];
    double win[4096];
    const int off = (n_fft - cfg->win_length) / 2;
    memset(win, 0, (size_t)n_fft * sizeof(double));
    for (int i = 0; i < cfg->win_length; i++) win[off + i] = (double)cfg->window[i];

    for (int i = 0; i < n_fft; i++) {
        const long s = start + i;
        const double v = (s >= 0 && (size_t)s < n_sig) ? sig[s] : 0.0;
        re[i] = v * win[i];
        im[i] = 0.0;
    }
    fft_radix2(re, im, n_fft);
    for (int b = 0; b < n_bins; b++) power[b] = re[b] * re[b] + im[b] * im[b];
    for (int m = 0; m < n_mels; m++) {
        double acc = 0.0;
        for (int b = 0; b < n_bins; b++)
            acc += power[b] * (double)cfg->mel_fb[(size_t)b * (size_t)n_mels + (size_t)m];
        row[m] = (float)log(acc + cfg->log_zero_guard);
    }
}

int mynah_mel_stream_init(mynah_mel_stream *ms, const mynah_feat_cfg *cfg) {
    memset(ms, 0, sizeof(*ms));
    if (cfg->n_fft > 4096) return -1;
    ms->cfg = cfg;
    ms->cap = 65536;
    ms->samples = malloc(ms->cap * sizeof(double));
    return ms->samples ? 0 : -1;
}

void mynah_mel_stream_free(mynah_mel_stream *ms) { free(ms->samples); ms->samples = NULL; }

int mynah_mel_stream_feed(mynah_mel_stream *ms, const float *audio, size_t n,
                          float *out, int cap_frames) {
    const mynah_feat_cfg *cfg = ms->cfg;
    if (ms->n_samples + n > ms->cap) {
        while (ms->n_samples + n > ms->cap) ms->cap *= 2;
        double *nb = realloc(ms->samples, ms->cap * sizeof(double));
        if (!nb) return 0;
        ms->samples = nb;
    }
    for (size_t i = 0; i < n; i++) {
        const float prev = (ms->n_samples + i == 0) ? 0.0f : ms->last_raw;
        const double pre = (ms->n_samples + i == 0)
                               ? (double)audio[i]
                               : (double)audio[i] - cfg->preemphasis * (double)prev;
        ms->samples[ms->n_samples + i] = pre;
        ms->last_raw = audio[i];
    }
    ms->n_samples += n;

    /* frame t pronto quando il segnale copre t*hop + n_fft/2 */
    int emitted = 0;
    while (emitted < cap_frames) {
        const size_t need = (size_t)ms->next_frame * (size_t)cfg->hop_length + (size_t)cfg->n_fft / 2;
        if (need > ms->n_samples) break;
        mel_one_frame(cfg, ms->samples, ms->n_samples, ms->next_frame,
                      out + (size_t)emitted * (size_t)cfg->n_mels);
        ms->next_frame++;
        emitted++;
    }
    return emitted;
}

int mynah_mel_stream_finish(mynah_mel_stream *ms, float *out, int cap_frames) {
    const mynah_feat_cfg *cfg = ms->cfg;
    const int valid = (int)(ms->n_samples / (size_t)cfg->hop_length);
    int emitted = 0;
    while (ms->next_frame < valid && emitted < cap_frames) {
        mel_one_frame(cfg, ms->samples, ms->n_samples, ms->next_frame,
                      out + (size_t)emitted * (size_t)cfg->n_mels);
        ms->next_frame++;
        emitted++;
    }
    ms->finished = 1;
    return emitted;
}
