/* Lettura audio: WAV PCM16 (mono/stereo -> mono). Per ora richiede 16 kHz;
 * il resampler arriva con un task dedicato (TODO 0.5). */
#ifndef MYNAH_AUDIO_H
#define MYNAH_AUDIO_H

#include <stddef.h>

/* Carica un WAV PCM16. Ritorna campioni float32 in [-1,1] (malloc, caller free)
 * e scrive *n_samples e *sample_rate. NULL su errore (messaggio su stderr). */
float *mynah_wav_load(const char *path, size_t *n_samples, int *sample_rate);

#endif
