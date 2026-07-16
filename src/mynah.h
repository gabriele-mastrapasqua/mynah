/* Mynah — a lightweight native C runtime for streaming and offline ASR.
 *
 * API pubblica di libmynah. In questa fase (M0) contiene solo lo scheletro:
 * versione e i typedef dei callback. Le funzioni verranno aggiunte man mano
 * che gli stadi della pipeline superano la validazione contro l'oracolo
 * (vedi TODO.md — ogni cosa dichiarata qui DEVE esistere ed essere testata).
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

#ifdef __cplusplus
}
#endif

#endif /* MYNAH_H */
