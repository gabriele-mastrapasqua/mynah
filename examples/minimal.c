/* Esempio minimo di libmynah: trascrive un WAV e stampa testo + parole con tempi.
 *
 * Build (dalla root del repo):
 *   make lib
 *   cc -O2 -Isrc examples/minimal.c libmynah.a -o minimal \
 *      -framework Accelerate                    # macOS
 *   cc -O2 -Isrc examples/minimal.c libmynah.a -o minimal \
 *      -lopenblas -lm -lpthread                 # Linux
 *
 * Uso: ./minimal <model_dir> <file.wav>
 */
#include <stdio.h>
#include <stdlib.h>

#include "mynah.h"
#include "audio.h"   /* mynah_wav_load / mynah_resample (helper WAV di libmynah) */

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "uso: %s <model_dir> <file.wav>\n", argv[0]);
        return 2;
    }

    mynah_model *m = mynah_load(argv[1]);
    if (!m) return 1;

    size_t n;
    int sr;
    float *samples = mynah_wav_load(argv[2], &n, &sr);
    if (!samples) { mynah_free(m); return 1; }
    if (sr != 16000) {
        size_t n2;
        float *rs = mynah_resample(samples, n, sr, 16000, &n2);
        free(samples);
        if (!rs) { mynah_free(m); return 1; }
        samples = rs;
        n = n2;
    }

    char lang[16];
    mynah_word *words = NULL;
    int n_words = 0;
    char *text = mynah_transcribe_ts(m, samples, n, "auto", /*lookahead=*/-1, lang,
                                     &words, &n_words);
    if (text) {
        printf("[%s] %s\n", lang[0] ? lang : "auto", text);
        for (int i = 0; i < n_words; i++)
            printf("  %6.2f %6.2f  %s\n", words[i].t0, words[i].t1, words[i].word);
    }

    mynah_words_free(words, n_words);
    free(text);
    free(samples);
    mynah_free(m);
    return text ? 0 : 1;
}
