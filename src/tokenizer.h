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

/* Decodifica tokens -> testo (malloc, caller free). Se un tag lingua è presente
 * scrive fino a 15 char in lang_out (se non NULL). ▁ -> spazio, spazio iniziale via. */
char *mynah_detokenize(const mynah_tokenizer *tk, const int *tokens, int n,
                       char *lang_out);

#endif
