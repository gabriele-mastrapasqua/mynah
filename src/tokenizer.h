/* Detokenizer SentencePiece-BPE: id -> pezzi -> testo UTF-8.
 * Carica tokens.json (array di pieces, generato dal convertitore).
 * I token speciali <xx-XX> sono tag lingua: strippati e riportati come lang. */
#ifndef MYNAH_TOKENIZER_H
#define MYNAH_TOKENIZER_H

typedef struct {
    char **pieces;
    int n_pieces;
} mynah_tokenizer;

int mynah_tokenizer_load(mynah_tokenizer *tk, const char *tokens_json_path);
void mynah_tokenizer_free(mynah_tokenizer *tk);

/* id della PRIMA occorrenza del piece (i tokenizer aggregati ripetono <unk> nei
 * sub-vocab: gli speciali stanno nel primo). -1 se assente. */
int mynah_tok_find(const mynah_tokenizer *tk, const char *piece);

/* Decodifica tokens -> testo (malloc, caller free). Se un tag lingua è presente
 * scrive fino a 15 char in lang_out (se non NULL). ▁ -> spazio, spazio iniziale via. */
char *mynah_detokenize(const mynah_tokenizer *tk, const int *tokens, int n,
                       char *lang_out);

#include "mynah.h"  /* mynah_word */

/* Raggruppa i token in parole (nuova parola a ogni piece che inizia con ▁) con
 * timestamp dai frame di emissione: t0 = frame del primo piece * frame_sec,
 * t1 = (frame dell'ultimo piece + 1) * frame_sec. I token speciali <...> sono
 * saltati. Scrive in *out un array malloc (caller: mynah_words_free). 0 = ok. */
int mynah_detokenize_words(const mynah_tokenizer *tk, const int *tokens,
                           const int *frames, int n, double frame_sec,
                           mynah_word **out, int *n_out);

#endif
