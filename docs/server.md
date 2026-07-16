# mynah-server — API HTTP + WebSocket

```sh
make && ./mynah-server -m models/nemotron-3.5-asr-streaming-0.6b -p 8090 --threads 4
```

## Endpoint

### POST /v1/audio/transcriptions — OpenAI-compatible

Multipart form-data (come l'API OpenAI/Whisper) o body raw `audio/wav`:

```sh
curl -F file=@audio.wav -F language=auto http://localhost:8090/v1/audio/transcriptions
# {"text": "..."}

curl -F file=@audio.wav -F language=it-IT -F response_format=verbose_json ...
# {"text": "...", "task": "transcribe", "language": "it-IT", "duration": 5.23}

curl -X POST --data-binary @audio.wav -H 'Content-Type: audio/wav' ...
```

Campi: `file` (WAV PCM16, qualsiasi sample rate — resampling automatico),
`language` (tag locale o `auto`, default auto), `response_format`
(`json` | `text` | `verbose_json`), `lookahead` (0|1|3|6|13, default del modello).

### GET /v1/audio/stream — WebSocket streaming

Query: `?lang=auto&lookahead=3`. Protocollo:
- client → server: frame **binari** con PCM s16le 16 kHz mono (qualsiasi taglia);
- server → client: frame testo JSON `{"text": "<delta definitivo>", "language": ...,
  "audio_seconds": ...}` man mano che il testo si consolida;
- alla close del client il server processa la coda, invia `{"done": true,
  "language": "..."}` e chiude.

Client di riferimento (stdlib Python): `tools/eval/ws_client.py`.

### GET /v1/models · GET /v1/health · OPTIONS (CORS)

## Concorrenza

Il modello è **read-only** (pesi mmap) e condiviso fra i worker: ogni richiesta ha solo
il proprio stato di decode (~12 MB per stream). `--threads N` = richieste servite in
parallelo; le eccedenze si accodano (503 oltre 128 in coda). Nessun clone del modello,
nessun lock sul percorso caldo.

**Batching cross-richiesta** (`--batch N`, default 8): le trascrizioni REST pendenti
vengono aggregate (finestra 25 ms) e processate **weight-stationary**: packing senza
padding dei frame di tutte le richieste, GEMM per-frame (FFN/proiezioni, >95% dei FLOP)
su `[ΣT, d]` con i pesi letti una volta per layer; attention/conv restano per-sequenza.
Output identico al percorso B=1 (verificato).

Numeri onesti su Apple Silicon (Accelerate multithreaded): batch ≈ pool di thread per
il throughput (la GEMM singola satura già i core) — il batch vale ~1.4× sul sequenziale
a cache calda e riduce contesa/footprint. Il guadagno grosso è atteso su x86/OpenBLAS
molti-core e sui futuri backend GPU (M5), dove i pesi letti una volta contano davvero.
`--batch 1` disabilita (torna al per-richiesta nei worker).

## Test

`make test-server` — REST (multipart, raw, verbose, errori), 4 richieste concorrenti,
WebSocket streaming end-to-end. Skip automatico se il modello non è scaricato.

## Note operative

- Un solo modello per processo (`/v1/models` ne elenca uno).
- Timeout/limiti: body ≤ 200 MB, header ≤ 64 KB, coda ≤ 128 connessioni.
- TLS/auth fuori scope: mettere dietro un reverse proxy (nginx/caddy) in produzione.
