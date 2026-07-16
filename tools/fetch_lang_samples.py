#!/usr/bin/env python3
"""Scarica sample audio reali per lingua da Tatoeba (frasi brevi con audio CC).

Per ogni lingua supportata da Nemotron 3.5 prende fino a N frasi con audio,
le converte a WAV 16 kHz mono (ffmpeg) e scrive un manifest con testo di
riferimento e attribuzione. I sample NON vanno committati (solo fixture locali,
licenze audio Tatoeba variabili): tests/audio/langs/ è in .gitignore.

Uso: uv run python fetch_lang_samples.py [n_per_lingua=3]
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "audio" / "langs"

# locale Nemotron -> codice ISO 639-3 Tatoeba (le varianti regionali condividono
# la lingua: en-GB testata via eng, ecc.)
LOCALES = {
    "en-US": "eng", "es-ES": "spa", "fr-FR": "fra", "it-IT": "ita", "pt-BR": "por",
    "nl-NL": "nld", "de-DE": "deu", "tr-TR": "tur", "ru-RU": "rus", "ar-AR": "ara",
    "hi-IN": "hin", "ja-JP": "jpn", "ko-KR": "kor", "vi-VN": "vie", "uk-UA": "ukr",
    "pl-PL": "pol", "sv-SE": "swe", "cs-CZ": "ces", "nb-NO": "nob", "da-DK": "dan",
    "bg-BG": "bul", "fi-FI": "fin", "hr-HR": "hrv", "sk-SK": "slk", "zh-CN": "cmn",
    "hu-HU": "hun", "ro-RO": "ron", "et-EE": "est", "el-GR": "ell", "lt-LT": "lit",
    "lv-LV": "lav", "mt-MT": "mlt", "sl-SI": "slv", "he-IL": "heb", "th-TH": "tha",
    "nn-NO": "nno",
}

API = "https://api.tatoeba.org/unstable/sentences?lang={code}&has_audio=yes&sort=created&limit=30"
AUDIO = "https://audio.tatoeba.org/sentences/{code}/{sid}.mp3"


def fetch_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "mynah-asr-tests"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def main() -> None:
    n_per_lang = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, list[dict]] = {}
    report = []

    for locale, code in LOCALES.items():
        lang_dir = OUT_DIR / locale
        lang_dir.mkdir(exist_ok=True)
        got = 0
        try:
            data = fetch_json(API.format(code=code)).get("data", [])
        except Exception as e:
            report.append(f"{locale}: API errore ({e})")
            continue

        entries = []
        for s in data:
            if got >= n_per_lang:
                break
            text = s.get("text", "")
            if len(text) < 12:          # frasi troppo corte non testano nulla
                continue
            sid = s["id"]
            wav = lang_dir / f"{sid}.wav"
            if not wav.exists():
                mp3 = lang_dir / f"{sid}.mp3"
                try:
                    urllib.request.urlretrieve(AUDIO.format(code=code, sid=sid), mp3)
                    subprocess.run(
                        ["ffmpeg", "-v", "quiet", "-y", "-i", str(mp3), "-ar", "16000",
                         "-ac", "1", str(wav)],
                        check=True,
                    )
                except Exception:
                    mp3.unlink(missing_ok=True)
                    continue
                finally:
                    mp3.unlink(missing_ok=True)
            entries.append({
                "wav": f"{locale}/{sid}.wav",
                "text": text,
                "tatoeba_id": sid,
                "sentence_license": s.get("license"),
                "owner": s.get("owner"),
            })
            got += 1
            time.sleep(0.2)

        if entries:
            manifest[locale] = entries
        report.append(f"{locale}: {got} sample")

    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1))
    print("\n".join(report))
    total = sum(len(v) for v in manifest.values())
    print(f"\nTotale: {total} sample in {len(manifest)} lingue -> {OUT_DIR}")


if __name__ == "__main__":
    main()
