# samples/ — audio reale per i test di qualità

Clip da **[FLEURS](https://huggingface.co/datasets/google/fleurs)** (Google,
[Conneau et al. 2022](https://arxiv.org/abs/2205.12446)), licenza
**CC-BY 4.0** — ridistribuiti qui con attribuzione, invariati (16 kHz mono PCM16).

La proprietà che li rende preziosi: le frasi sono **parallele tra le lingue**
(stesso `fleurs_id` = stessa frase, letta da madrelingua): la versione inglese
fa da riferimento per valutare la **speech translation** di Canary, non solo
l'ASR. Trascrizioni e riferimenti in [`manifest.json`](manifest.json).

| id | contenuto | lingue |
|---|---|---|
| 1521 | il satellite che ritrasmette il segnale | it, en, de, es, fr |
| 1534 | Timbuctù come metafora di terra lontana | it, en, de, es, fr |

Uso: `make test-samples` (CER ASR + word-overlap traduzione, backend cpu+metal;
vedi `tools/eval/test_samples.py`). Rigenerazione: `cd tools && uv run python
fetch_fleurs_samples.py`.
