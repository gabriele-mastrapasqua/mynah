# samples/ — audio reale per i test di qualità

Clip da **[FLEURS](https://huggingface.co/datasets/google/fleurs)** (Google,
[Conneau et al. 2022](https://arxiv.org/abs/2205.12446)), licenza
**CC-BY 4.0** — ridistribuiti qui con attribuzione, invariati (16 kHz mono PCM16).

La proprietà che li rende preziosi: le frasi sono **parallele tra le lingue**
(stesso `fleurs_id` = stessa frase, letta da madrelingua): la versione inglese
fa da riferimento per valutare la **speech translation** di Canary, non solo
l'ASR. Trascrizioni e riferimenti in [`manifest.json`](manifest.json).

| clip | contenuto | lingue |
|---|---|---|
| fleurs_1521 | il satellite che ritrasmette il segnale | it, en, de, es, fr, pt, nl, pl, ru, uk, ja |
| fleurs_1534 | Timbuctù come metafora di terra lontana | it, en, de, es, fr, pt, nl, pl, ru, uk, ja |
| fleurs_long | ~95 s di frasi EN concatenate con pause | en |
| long/en_long.mp3 | ~5 min (long transcribe: CER + RTF) | en |
| long/de_long.mp3 | ~2 min di frasi parallele (long translate de→en) | de |

I clip `long/*.mp3` (32 kbps) richiedono **ffmpeg** nel PATH per i test (decode
al volo; il runtime resta WAV-only — per file mp3/m4a/ecc. l'utente converte con
`ffmpeg -i file.mp3 -ar 16000 -ac 1 out.wav`).

Le 11 lingue esercitano i 40 locale di Nemotron e le 25 lingue EU di v3
(alfabeti: latino, cirillico, giapponese). Il clip lungo ha riferimento esatto:
serve ai check di **segmentazione su silenzio**, **timestamp** su ~100 parole e
**streaming cache-aware** (audio vero in stdin).

Uso: `make test-samples` (CER ASR + word-overlap traduzione, backend cpu+metal,
`--backend cuda` pronto per Linux; vedi `tools/eval/test_samples.py`).
Rigenerazione: `cd tools && uv run python fetch_fleurs_samples.py`.
