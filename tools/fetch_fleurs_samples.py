#!/usr/bin/env python3
"""Costruisce samples/ dal dataset FLEURS (google/fleurs, CC-BY 4.0).

Frasi PARALLELE tra le lingue (stesso id FLEURS = stessa frase tradotta):
l'inglese fa da riferimento per valutare la speech translation di Canary.
I wav (16 kHz mono PCM16) vengono COMMITTATI nel repo: pochi e leggeri.

Uso: uv run python fetch_fleurs_samples.py            # dalla dir tools/
Riscarica i tsv, estrae in streaming SOLO i wav scelti dai dev.tar.gz
(--fast-read), scrive samples/<lang>/fleurs_<id>.wav + manifest.json.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
CACHE = Path("/tmp/fleurs")

# lingua mynah -> config FLEURS
LANGS = {"it": "it_it", "en": "en_us", "de": "de_de", "es": "es_419", "fr": "fr_fr"}
# id FLEURS delle frasi scelte (parallele in tutte le lingue, 7-13 s,
# contenuti distintivi: 1521 satellite, 1534 Timbuctù)
IDS = ["1521", "1534"]


def tsv_rows(cfg: str) -> dict[str, dict]:
    path = CACHE / f"{cfg}.tsv"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sL", f"{BASE}/{cfg}/dev.tsv", "-o", str(path)], check=True)
    rows: dict[str, dict] = {}
    with open(path) as f:
        for r in csv.reader(f, delimiter="\t"):
            if len(r) >= 7 and r[0] not in rows:
                rows[r[0]] = {"file": r[1], "raw": r[2], "dur": int(r[5]) / 16000.0}
    return rows


def extract(cfg: str, files: list[str]) -> None:
    outdir = CACHE / cfg
    if all((outdir / "dev" / f).exists() for f in files):
        return
    outdir.mkdir(parents=True, exist_ok=True)
    pats = [f"*/{f}" for f in files]
    curl = subprocess.Popen(["curl", "-sL", f"{BASE}/{cfg}/audio/dev.tar.gz"],
                            stdout=subprocess.PIPE)
    subprocess.run(["tar", "-xzf", "-", "-C", str(outdir), "--fast-read", *pats],
                   stdin=curl.stdout, check=True)
    curl.stdout.close()
    curl.wait()


def main() -> None:
    samples_dir = ROOT / "samples"
    manifest: dict = {
        "source": "FLEURS (google/fleurs) — Conneau et al., CC-BY 4.0",
        "url": "https://huggingface.co/datasets/google/fleurs",
        "note": "frasi parallele tra lingue: stesso id = stessa frase; l'inglese "
                "è il riferimento per la speech translation",
        "samples": [],
    }
    texts = {lang: tsv_rows(cfg) for lang, cfg in LANGS.items()}
    for lang, cfg in LANGS.items():
        rows = texts[lang]
        missing = [i for i in IDS if i not in rows]
        if missing:
            sys.exit(f"{cfg}: id mancanti nel dev set: {missing}")
        extract(cfg, [rows[i]["file"] for i in IDS])
        (samples_dir / lang).mkdir(parents=True, exist_ok=True)
        for i in IDS:
            src = CACHE / cfg / "dev" / rows[i]["file"]
            dst = samples_dir / lang / f"fleurs_{i}.wav"
            # FLEURS distribuisce float32: riscrittura in PCM16 (metà peso,
            # supporto universale — il runtime legge WAV PCM16)
            import soundfile as sf
            audio, sr = sf.read(src, dtype="float32")
            sf.write(dst, audio, sr, subtype="PCM_16")
            manifest["samples"].append({
                "file": f"{lang}/fleurs_{i}.wav",
                "lang": lang,
                "fleurs_id": int(i),
                "duration_sec": round(rows[i]["dur"], 1),
                "text": rows[i]["raw"],
                "en_ref": texts["en"][i]["raw"],
            })
    (samples_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    total = sum(s["duration_sec"] for s in manifest["samples"])
    print(f"OK samples/: {len(manifest['samples'])} wav, {total:.0f}s totali")


if __name__ == "__main__":
    main()
