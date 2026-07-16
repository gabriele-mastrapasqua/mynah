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
        fprintf(stderr, "[resample %d Hz -> 16000 Hz]\n", sr);
        size_t n_rs;
        float *rs = mynah_resample(samples, n_samples, sr, 16000, &n_rs);
        free(samples);
        if (!rs) { mynah_free(m); return 1; }
        samples = rs;
        n_samples = n_rs;
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

static void print_partial(const mynah_result *res, void *ud) {
    (void)ud;
    fputs(res->text, stdout);
    fflush(stdout);
}

static int cmd_stream(int argc, char **argv) {
    const char *model_dir = NULL, *lang = "auto";
    int lookahead = -1;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_dir = argv[++i];
        else if (strcmp(argv[i], "--lang") == 0 && i + 1 < argc) lang = argv[++i];
        else if (strcmp(argv[i], "--lookahead") == 0 && i + 1 < argc) lookahead = atoi(argv[++i]);
        else { fprintf(stderr, "opzione ignota: %s\n", argv[i]); return 2; }
    }
    if (!model_dir) { usage(); return 2; }

    mynah_model *m = mynah_load(model_dir);
    if (!m) return 1;
    mynah_stream *s = mynah_stream_open(m, lang, lookahead);
    if (!s) { mynah_free(m); return 1; }
    fprintf(stderr, "[stream: raw s16le 16 kHz mono da stdin, lang=%s]\n", lang);

    short pcm[1600]; /* 100 ms per read */
    float buf[1600];
    size_t got;
    while ((got = fread(pcm, sizeof(short), 1600, stdin)) > 0) {
        for (size_t i = 0; i < got; i++) buf[i] = (float)pcm[i] / 32768.0f;
        if (mynah_stream_feed(s, buf, got, print_partial, NULL) != 0) break;
    }
    mynah_stream_finish(s, print_partial, NULL);
    printf("\n");
    if (mynah_stream_lang(s)[0]) fprintf(stderr, "[lang=%s]\n", mynah_stream_lang(s));

    mynah_stream_close(s);
    mynah_free(m);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "--version") == 0) { printf("%s\n", mynah_version()); return 0; }
    if (argc >= 2 && strcmp(argv[1], "transcribe") == 0) return cmd_transcribe(argc - 2, argv + 2);
    if (argc >= 2 && strcmp(argv[1], "stream") == 0) return cmd_stream(argc - 2, argv + 2);
    usage();
    return argc < 2 ? 0 : 1;
}
