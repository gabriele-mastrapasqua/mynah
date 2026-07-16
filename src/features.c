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
