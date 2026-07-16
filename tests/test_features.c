/* Parità log-mel: C vs oracolo numpy.
 * Uso: test_features <model_dir> <file.wav> <golden_dir>
 * Exit: 0 ok, 1 mismatch, 77 skip (modello o golden assenti). */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/audio.h"
#include "../src/features.h"
#include "../src/weights.h"

/* Loader .npy minimale: v1.0, C-order, dtype <f4 o <f8. */
static double *npy_load(const char *path, size_t *n_elems) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    unsigned char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "\x93NUMPY", 6) != 0) { fclose(f); return NULL; }
    unsigned short hlen;
    if (fread(&hlen, 2, 1, f) != 1) { fclose(f); return NULL; }
    char *hdr = malloc((size_t)hlen + 1);
    if (fread(hdr, 1, hlen, f) != hlen) { free(hdr); fclose(f); return NULL; }
    hdr[hlen] = '\0';

    int is_f8 = strstr(hdr, "<f8") != NULL;
    if (!is_f8 && !strstr(hdr, "<f4")) { fprintf(stderr, "npy: dtype non supportato: %s\n", hdr); free(hdr); fclose(f); return NULL; }
    size_t total = 1;
    const char *sh = strstr(hdr, "'shape': (");
    if (!sh) { free(hdr); fclose(f); return NULL; }
    sh += strlen("'shape': (");
    while (*sh && *sh != ')') {
        if (*sh >= '0' && *sh <= '9') { total *= strtoull(sh, (char **)&sh, 10); }
        else sh++;
    }
    free(hdr);

    double *out = malloc(total * sizeof(double));
    if (is_f8) {
        if (fread(out, 8, total, f) != total) { free(out); fclose(f); return NULL; }
    } else {
        float *tmp = malloc(total * sizeof(float));
        if (fread(tmp, 4, total, f) != total) { free(tmp); free(out); fclose(f); return NULL; }
        for (size_t i = 0; i < total; i++) out[i] = (double)tmp[i];
        free(tmp);
    }
    fclose(f);
    *n_elems = total;
    return out;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "uso: %s <model_dir> <wav> <golden_dir>\n", argv[0]); return 2; }

    char path[1024];
    snprintf(path, sizeof(path), "%s/mel_filters.safetensors", argv[1]);
    mynah_safetensors *mf = mynah_st_open(path);
    if (!mf) return 77;

    snprintf(path, sizeof(path), "%s/mel.npy", argv[3]);
    size_t n_golden;
    double *golden = npy_load(path, &n_golden);
    if (!golden) { mynah_st_close(mf); return 77; }

    size_t n_samples; int sr;
    float *audio = mynah_wav_load(argv[2], &n_samples, &sr);
    if (!audio || sr != 16000) { fprintf(stderr, "wav non valido o non 16 kHz\n"); return 2; }

    const mynah_tensor *fb = mynah_st_get(mf, "mel_fb");
    const mynah_tensor *win = mynah_st_get(mf, "window");
    if (!fb || !win) { fprintf(stderr, "mel_filters.safetensors incompleto\n"); return 2; }

    mynah_feat_cfg cfg = {
        .sample_rate = 16000, .n_mels = (int)fb->shape[1], .n_fft = (int)(fb->shape[0] - 1) * 2,
        .win_length = (int)win->shape[0], .hop_length = 160,
        .preemphasis = 0.97, .log_zero_guard = pow(2.0, -24.0),
        .mel_fb = (const float *)fb->data, .window = (const float *)win->data,
    };

    int T, valid;
    float *feats = mynah_log_mel(&cfg, audio, n_samples, &T, &valid);
    if (!feats) return 2;

    size_t n_c = (size_t)T * (size_t)cfg.n_mels;
    if (n_c != n_golden) {
        fprintf(stderr, "FAIL: shape C %d x %d = %zu vs golden %zu\n", T, cfg.n_mels, n_c, n_golden);
        return 1;
    }

    double max_diff = 0.0, sum_diff = 0.0;
    for (size_t i = 0; i < n_c; i++) {
        double d = fabs((double)feats[i] - golden[i]);
        if (d > max_diff) max_diff = d;
        sum_diff += d;
    }
    double tol = 5e-4; /* tolleranza feature vs riferimento (prior-art §B.4) */
    printf("mel parity: T=%d valid=%d n_mels=%d | max|d|=%.3e mean|d|=%.3e (tol %.0e)\n",
           T, valid, cfg.n_mels, max_diff, sum_diff / (double)n_c, tol);

    free(audio); free(feats); free(golden); mynah_st_close(mf);
    if (max_diff > tol) { fprintf(stderr, "FAIL\n"); return 1; }
    printf("OK\n");
    return 0;
}
