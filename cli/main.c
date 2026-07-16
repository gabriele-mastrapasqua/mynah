/* mynah — CLI. Sottocomandi: transcribe (offline). stream arriva con M1.3. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mynah.h"
#include "audio.h"

static void usage(void) {
    printf("mynah %s — native ASR runtime for NeMo speech models\n\n", mynah_version());
    printf("Uso: mynah <comando> [opzioni]\n\n");
    printf("Comandi:\n");
    printf("  transcribe -m <model_dir> -i <file.wav> [--lang auto] [--lookahead N]\n");
    printf("             trascrizione offline (WAV PCM16 16 kHz)\n");
    printf("  stream     -m <model_dir>                 streaming da stdin     [M1.3]\n");
    printf("  --version                                 stampa la versione\n");
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int cmd_transcribe(int argc, char **argv) {
    const char *model_dir = NULL, *wav = NULL, *lang = "auto";
    int lookahead = -1;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_dir = argv[++i];
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) wav = argv[++i];
        else if (strcmp(argv[i], "--lang") == 0 && i + 1 < argc) lang = argv[++i];
        else if (strcmp(argv[i], "--lookahead") == 0 && i + 1 < argc) lookahead = atoi(argv[++i]);
        else { fprintf(stderr, "opzione ignota: %s\n", argv[i]); return 2; }
    }
    if (!model_dir || !wav) { usage(); return 2; }

    double t0 = now_sec();
    mynah_model *m = mynah_load(model_dir);
    if (!m) return 1;
    double t_load = now_sec() - t0;

    size_t n_samples;
    int sr;
    float *samples = mynah_wav_load(wav, &n_samples, &sr);
    if (!samples) { mynah_free(m); return 1; }
    if (sr != 16000) {
        fprintf(stderr, "mynah: servono 16 kHz, il file è a %d Hz (resampler in arrivo)\n", sr);
        free(samples); mynah_free(m);
        return 1;
    }

    char lang_out[16];
    t0 = now_sec();
    char *text = mynah_transcribe(m, samples, n_samples, lang, lookahead, lang_out);
    double t_run = now_sec() - t0;
    if (!text) { free(samples); mynah_free(m); return 1; }

    const double dur = (double)n_samples / 16000.0;
    fprintf(stderr, "[%.1fs audio | load %.2fs | inferenza %.2fs | RTF %.3f | lang=%s]\n",
            dur, t_load, t_run, t_run / dur, lang_out[0] ? lang_out : lang);
    printf("%s\n", text);

    free(text); free(samples); mynah_free(m);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "--version") == 0) { printf("%s\n", mynah_version()); return 0; }
    if (argc >= 2 && strcmp(argv[1], "transcribe") == 0) return cmd_transcribe(argc - 2, argv + 2);
    usage();
    return argc < 2 ? 0 : 1;
}
