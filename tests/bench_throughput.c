/* Throughput batch: replica lo stesso wav B volte (B = 1,2,4,...,max-batch) e
 * misura l'aggregato audio-s/s della trascrizione batch weight-stationary
 * (mynah_transcribe_batch). Pensato per stimare quante richieste parallele
 * regge un backend GPU (make cuda / metal), ma gira anche su cpu.
 * Verifica anche che ogni item del batch produca il testo del run B=1.
 * Uso: bench_throughput <model_dir> <wav> [--lang l] [--backend cpu|metal|cuda]
 *                       [--max-batch N] [--runs R]
 * Exit: 0 ok, 1 mismatch testi, 2 uso errato, 77 modello/wav assenti. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../src/audio.h"
#include "../src/backend.h"
#include "../src/mynah.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    const char *model_dir = NULL, *wav = NULL, *lang = "auto";
    int max_batch = 32, runs = 2;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--lang") == 0 && i + 1 < argc) lang = argv[++i];
        else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) mynah_set_backend(argv[++i]);
        else if (strcmp(argv[i], "--max-batch") == 0 && i + 1 < argc) max_batch = atoi(argv[++i]);
        else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) runs = atoi(argv[++i]);
        else if (!model_dir) model_dir = argv[i];
        else if (!wav) wav = argv[i];
        else { fprintf(stderr, "argomento ignoto: %s\n", argv[i]); return 2; }
    }
    if (!model_dir || !wav) {
        fprintf(stderr, "uso: %s <model_dir> <wav> [--lang l] [--backend b] "
                        "[--max-batch N] [--runs R]\n", argv[0]);
        return 2;
    }
    if (max_batch < 1) max_batch = 1;
    if (runs < 1) runs = 1;

    mynah_model *m = mynah_load(model_dir);
    if (!m) return 77;

    size_t ns;
    int sr;
    float *samples = mynah_wav_load(wav, &ns, &sr);
    if (!samples) { mynah_free(m); return 77; }
    if (sr != 16000) {
        size_t n_rs;
        float *rs = mynah_resample(samples, ns, sr, 16000, &n_rs);
        free(samples);
        if (!rs) { mynah_free(m); return 77; }
        samples = rs; ns = n_rs;
    }
    const double dur = (double)ns / 16000.0;

    /* riferimento B=1 (e warm-up: pesi su device, buffer allocati) */
    char *ref = mynah_transcribe(m, samples, ns, lang, -1, NULL);
    if (!ref) { free(samples); mynah_free(m); return 1; }

    const float **sv = malloc((size_t)max_batch * sizeof *sv);
    size_t *nv = malloc((size_t)max_batch * sizeof *nv);
    const char **lv = malloc((size_t)max_batch * sizeof *lv);
    char **out = malloc((size_t)max_batch * sizeof *out);
    if (!sv || !nv || !lv || !out) return 1;
    for (int b = 0; b < max_batch; b++) { sv[b] = samples; nv[b] = ns; lv[b] = lang; }

    printf("# %s | %s %.1fs | lang=%s | runs=%d (best)\n", model_dir, wav, dur, lang, runs);
    printf("%6s %10s %12s %14s\n", "batch", "wall-s", "s-item", "xRT-aggr");
    int rc = 0;
    for (int B = 1; B <= max_batch && rc == 0; B *= 2) {
        double best = 1e30;
        for (int r = 0; r < runs; r++) {
            for (int b = 0; b < B; b++) out[b] = NULL;
            double t0 = now_sec();
            if (mynah_transcribe_batch(m, sv, nv, B, lv, -1, out, NULL) != 0) {
                fprintf(stderr, "FAIL: transcribe_batch B=%d\n", B);
                rc = 1;
                break;
            }
            double dt = now_sec() - t0;
            if (dt < best) best = dt;
            for (int b = 0; b < B; b++) {
                if (out[b] && strcmp(out[b], ref) != 0) {
                    fprintf(stderr, "MISMATCH B=%d item %d:\n  ref:   %s\n  item:  %s\n",
                            B, b, ref, out[b]);
                    rc = 1;
                }
                free(out[b]);
            }
        }
        if (rc == 0)
            printf("%6d %10.3f %12.3f %14.1f\n",
                   B, best, best / B, (double)B * dur / best);
        fflush(stdout);
    }

    free(ref); free(samples);
    free(sv); free(nv); free(lv); free(out);
    mynah_free(m);
    return rc;
}
