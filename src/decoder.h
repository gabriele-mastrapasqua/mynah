/* Decoder RNNT: prediction network LSTM + joint, greedy decode.
 * Semantica NeMo/HF (docs/nemotron-arch.md): SOS = blank, lo stato LSTM avanza
 * SOLO su emissione non-blank, max_symbols_per_step limita l'inner loop.
 * Stato liftato e chunk-invariante: decodifica incrementale ≡ intera (prior-art). */
#ifndef MYNAH_DECODER_H
#define MYNAH_DECODER_H

#include "qmat.h"
#include "weights.h"

#define MYNAH_MAX_PRED_LAYERS 4

typedef struct {
    const float *embedding;                 /* [vocab, H] */
    const float *w_ih[MYNAH_MAX_PRED_LAYERS], *w_hh[MYNAH_MAX_PRED_LAYERS];
    const float *b_ih[MYNAH_MAX_PRED_LAYERS], *b_hh[MYNAH_MAX_PRED_LAYERS];
    const float *proj_w, *proj_b;           /* decoder_projector [H, H] */
    mynah_qmat head;                        /* joint.head [vocab, H] (int8 opz.) */
    const float *head_b;
    int vocab, hidden, n_layers, blank, max_symbols;
} mynah_decoder;

typedef struct {
    float h[MYNAH_MAX_PRED_LAYERS][1024];   /* dimensione max; usa dec->hidden */
    float c[MYNAH_MAX_PRED_LAYERS][1024];
    float g[1024];                          /* output pred-net corrente (cache) */
    int last_token;                         /* -1 = non inizializzato */
} mynah_dec_state;

int mynah_decoder_init(mynah_decoder *dec, const mynah_safetensors *st,
                       int blank, int max_symbols, int quantize);

void mynah_dec_state_reset(const mynah_decoder *dec, mynah_dec_state *s);

/* Greedy RNNT su enc [T, H]. Appende i token emessi a tokens[] (capienza cap),
 * ritorna il numero di token emessi. Lo stato è persistente tra chiamate
 * (streaming-ready): passare lo stesso stato per i chunk successivi. */
int mynah_greedy_decode(const mynah_decoder *dec, mynah_dec_state *s,
                        const float *enc, int T, int *tokens, int cap);

#endif
