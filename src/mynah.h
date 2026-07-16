/* Mynah — a lightweight native C runtime for streaming and offline ASR.
 *
 * API pubblica di libmynah. Fase attuale: trascrizione offline-chunked (M1.2).
 * Streaming API (mynah_stream_*) in arrivo con M1.3.
 */
#ifndef MYNAH_H
#define MYNAH_H

#include <stddef.h>
#include <stdbool.h>

#define MYNAH_VERSION_MAJOR 0
#define MYNAH_VERSION_MINOR 0
#define MYNAH_VERSION_PATCH 1
#define MYNAH_VERSION "0.0.1-dev"

#ifdef __cplusplus
extern "C" {
#endif

/* Risultato incrementale di trascrizione (streaming e offline).
 * text è UTF-8, valido solo per la durata della callback. */
typedef struct {
    const char *text;      /* testo del segmento/parziale                   */
    double      t0, t1;    /* finestra temporale in secondi (se disponibile) */
    bool        is_final;  /* false = partial (può cambiare), true = commit  */
    const char *lang;      /* tag lingua rilevata, NULL se non disponibile   */
} mynah_result;

typedef void (*mynah_result_cb)(const mynah_result *res, void *userdata);

const char *mynah_version(void);

/* ----------------------------------------------------------------- modello */
typedef struct mynah_model mynah_model;

/* Carica un modello convertito (directory con mynah.json + model.safetensors
 * + tokens.json + mel_filters.safetensors). NULL su errore. */
mynah_model *mynah_load(const char *model_dir);
void mynah_free(mynah_model *m);

/* Risolve un tag lingua ("it-IT", "auto", ...) nel prompt id. -1 se ignoto. */
int mynah_lang_id(const mynah_model *m, const char *lang);

/* Lookahead (right context) validi per il modello, es. {3,0,6,13}. */
int mynah_lookaheads(const mynah_model *m, int out[8]);

/* Trascrizione offline: samples float32 [-1,1] 16 kHz mono.
 * lang: tag ("auto" per detection). lookahead: -1 = default del modello.
 * Ritorna testo UTF-8 (malloc, caller free); se lang_out != NULL (>= 16 byte)
 * vi scrive la lingua rilevata. NULL su errore. */
char *mynah_transcribe(mynah_model *m, const float *samples, size_t n_samples,
                       const char *lang, int lookahead, char *lang_out);

/* --------------------------------------------------------------- streaming
 * Cache-aware, latenza = (lookahead+1) * 80 ms. Il testo emesso via callback è
 * SEMPRE definitivo (greedy monotono, mai ritrattato): is_final = true. */
typedef struct mynah_stream mynah_stream;

mynah_stream *mynah_stream_open(mynah_model *m, const char *lang, int lookahead);

/* Alimenta campioni float32 16 kHz mono; la callback riceve i delta di testo. */
int mynah_stream_feed(mynah_stream *s, const float *samples, size_t n,
                      mynah_result_cb cb, void *userdata);

/* Fine stream: processa la coda (ultimo chunk paddato) ed emette il resto. */
int mynah_stream_finish(mynah_stream *s, mynah_result_cb cb, void *userdata);

/* Lingua rilevata finora ("" se non ancora emessa). */
const char *mynah_stream_lang(const mynah_stream *s);

void mynah_stream_close(mynah_stream *s);

#ifdef __cplusplus
}
#endif

#endif /* MYNAH_H */
