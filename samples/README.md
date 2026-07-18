# samples/ — real audio for quality tests

Clips from **[FLEURS](https://huggingface.co/datasets/google/fleurs)** (Google,
[Conneau et al. 2022](https://arxiv.org/abs/2205.12446)), license
**CC-BY 4.0** — redistributed here with attribution, unmodified (16 kHz mono PCM16).

The property that makes them valuable: the sentences are **parallel across languages**
(same `fleurs_id` = same sentence, read by native speakers): the English version
serves as the reference for evaluating Canary's **speech translation**, not just
the ASR. Transcriptions and references in [`manifest.json`](manifest.json).

| clip | content | languages |
|---|---|---|
| fleurs_1521 | the satellite relaying the signal | it, en, de, es, fr, pt, nl, pl, ru, uk, ja |
| fleurs_1534 | Timbuktu as a metaphor for a faraway land | it, en, de, es, fr, pt, nl, pl, ru, uk, ja |
| fleurs_long | ~95 s of concatenated EN sentences with pauses | en |
| long/en_long.wav | ~5 min (long transcribe: CER + RTF) | en |
| long/de_long.wav | ~2 min of parallel sentences (long translate de→en) | de |

Everything is WAV PCM16 16 kHz: the tests have no decoding dependencies. The runtime
reads ONLY WAV: for mp3/m4a/etc. convert with ffmpeg
(`ffmpeg -i file.mp3 -ar 16000 -ac 1 out.wav`).

The 11 languages exercise Nemotron's 40 locales and v3's 25 EU languages
(scripts: Latin, Cyrillic, Japanese). The long clip has an exact reference:
it serves the checks for **silence-based segmentation**, **timestamps** over ~100 words and
**cache-aware streaming** (real audio on stdin).

Usage: `make test-samples` (ASR CER + translation word-overlap, cpu+metal backends,
`--backend cuda` ready for Linux; see `tools/eval/test_samples.py`).
Regeneration: `cd tools && uv run python fetch_fleurs_samples.py`.
