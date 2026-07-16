#include "audio.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static uint32_t rd_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint16_t rd_u16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }

float *mynah_wav_load(const char *path, size_t *n_samples, int *sample_rate) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "audio: impossibile aprire %s\n", path); return NULL; }

    uint8_t hdr[12];
    if (fread(hdr, 1, 12, f) != 12 || memcmp(hdr, "RIFF", 4) != 0 || memcmp(hdr + 8, "WAVE", 4) != 0) {
        fprintf(stderr, "audio: %s non è un WAV RIFF\n", path);
        fclose(f);
        return NULL;
    }

    int channels = 0, bits = 0, sr = 0;
    uint8_t *data = NULL;
    uint32_t data_len = 0;

    uint8_t chdr[8];
    while (fread(chdr, 1, 8, f) == 8) {
        uint32_t sz = rd_u32(chdr + 4);
        if (memcmp(chdr, "fmt ", 4) == 0) {
            uint8_t fmt[16];
            if (sz < 16 || fread(fmt, 1, 16, f) != 16) break;
            if (sz > 16) fseek(f, sz - 16, SEEK_CUR);
            uint16_t audio_format = rd_u16(fmt);
            channels = rd_u16(fmt + 2);
            sr = (int)rd_u32(fmt + 4);
            bits = rd_u16(fmt + 14);
            if (audio_format != 1 /* PCM */) {
                fprintf(stderr, "audio: formato WAV %u non supportato (solo PCM)\n", audio_format);
                fclose(f);
                return NULL;
            }
        } else if (memcmp(chdr, "data", 4) == 0) {
            data = malloc(sz);
            if (!data || fread(data, 1, sz, f) != sz) {
                fprintf(stderr, "audio: chunk data troncato in %s\n", path);
                free(data);
                fclose(f);
                return NULL;
            }
            data_len = sz;
        } else {
            fseek(f, sz + (sz & 1), SEEK_CUR); /* chunk dispari: pad byte */
        }
    }
    fclose(f);

    if (!data || channels <= 0 || bits != 16) {
        fprintf(stderr, "audio: %s — atteso PCM16 con chunk fmt+data (ch=%d bits=%d)\n", path, channels, bits);
        free(data);
        return NULL;
    }

    size_t frames = data_len / (size_t)(channels * 2);
    float *out = malloc(frames * sizeof(float));
    if (!out) { free(data); return NULL; }

    const int16_t *s = (const int16_t *)data;
    for (size_t i = 0; i < frames; i++) {
        int32_t acc = 0;
        for (int c = 0; c < channels; c++) acc += s[i * (size_t)channels + (size_t)c];
        out[i] = (float)acc / (float)channels / 32768.0f;
    }
    free(data);

    *n_samples = frames;
    *sample_rate = sr;
    return out;
}

float *mynah_resample(const float *in, size_t n_in, int sr_in, int sr_out, size_t *n_out) {
    if (sr_in == sr_out) {
        float *copy = malloc(n_in * sizeof(float));
        if (!copy) return NULL;
        memcpy(copy, in, n_in * sizeof(float));
        *n_out = n_in;
        return copy;
    }
    const double ratio = (double)sr_out / (double)sr_in;
    /* in downsampling il sinc va tagliato alla nuova Nyquist */
    const double fc = ratio < 1.0 ? ratio : 1.0;
    const int taps = 32;
    const size_t N = (size_t)((double)n_in * ratio);
    float *out = malloc(N * sizeof(float));
    if (!out) return NULL;

    for (size_t i = 0; i < N; i++) {
        const double center = (double)i / ratio;   /* posizione nel segnale sorgente */
        const long i0 = (long)floor(center) - taps + 1;
        double acc = 0.0, wsum = 0.0;
        for (long j = i0; j < i0 + 2 * taps; j++) {
            const double x = ((double)j - center) * fc;
            const double sinc = x == 0.0 ? 1.0 : sin(M_PI * x) / (M_PI * x);
            const double hw = 0.5 + 0.5 * cos(M_PI * ((double)j - center) / (double)taps);
            const double w = sinc * hw;
            wsum += w;
            if (j >= 0 && (size_t)j < n_in) acc += w * (double)in[j];
        }
        out[i] = (float)(acc / (wsum > 1e-9 ? wsum : 1.0));
    }
    *n_out = N;
    return out;
}
