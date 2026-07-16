#!/usr/bin/env python3
"""Suite multilingua: trascrive i sample per-lingua (tests/audio/langs/) con
`mynah transcribe --lang auto` e verifica (a) language detection, (b) CER vs testo
di riferimento (normalizzato: lowercase, senza punteggiatura).

Uso: uv run python -m eval.test_langs [--cer-max 0.3] [--mynah ../mynah] [--model DIR]
Exit: 0 tutte le lingue ok, 1 fallimenti, 77 skip (sample o modello assenti).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# Tier "adaptation-ready" della model card: qualità debole BY DESIGN (serve
# fine-tuning). Segnalate come WEAK, non contano come fallimento della suite.
ADAPTATION_TIER = {"el-GR", "lt-LT", "lv-LV", "mt-MT", "sl-SI", "he-IL", "th-TH", "nn-NO"}


def normalize(s: str) -> str:
    s = re.sub(r"<[^<>]{1,12}>", " ", s)   # tag lingua spelled-out dal modello
    s = unicodedata.normalize("NFKC", s).lower()
    s = "".join(c for c in s if not unicodedata.category(c).startswith("P"))
    return re.sub(r"\s+", " ", s).strip()


def cer(ref: str, hyp: str) -> float:
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cer-max", type=float, default=0.3)
    ap.add_argument("--mynah", default=str(ROOT / "mynah"))
    ap.add_argument("--model", default=str(ROOT / "models/nemotron-3.5-asr-streaming-0.6b"))
    args = ap.parse_args()

    manifest_path = ROOT / "tests/audio/langs/manifest.json"
    if not manifest_path.exists() or not Path(args.model, "mynah.json").exists():
        print("SKIP: sample (tools/fetch_lang_samples.py) o modello assenti")
        sys.exit(77)
    manifest = json.loads(manifest_path.read_text())

    def transcribe(wav: Path, lang: str) -> tuple[str, str]:
        proc = subprocess.run(
            [args.mynah, "transcribe", "-m", args.model, "-i", str(wav), "--lang", lang],
            capture_output=True, text=True, timeout=300)
        m = re.search(r"lang=([\w-]+)", proc.stderr)
        return proc.stdout.strip(), (m.group(1) if m else "?")

    # Criterio primario: CER con lingua ESPLICITA (l'ASR funziona per la lingua?).
    # Secondario/informativo: language detection con "auto" (sui clip corti il
    # modello può non rilevare la lingua e non emettere nulla: comportamento noto).
    n_lang_ok = n_lang_fail = 0
    failures = []
    print(f"{'locale':8} {'sample':>6} {'CER ok':>7} {'CER medio':>10} {'auto-lang':>10}  esito")
    for locale, entries in sorted(manifest.items()):
        cers, auto_hits = [], 0
        for e in entries:
            wav = ROOT / "tests/audio/langs" / e["wav"]
            hyp, _ = transcribe(wav, locale)
            c = cer(e["text"], hyp)
            cers.append(c)
            _, detected = transcribe(wav, "auto")
            if detected.split("-")[0] == locale.split("-")[0]:
                auto_hits += 1
            if c > args.cer_max:
                sample_id = e.get("tatoeba_audio_id") or e.get("fleurs_id") or "?"
                failures.append(f"  {locale} [{sample_id}] cer={c:.2f}\n"
                                f"    ref: {e['text']}\n    hyp: {hyp}")
        n_ok = sum(1 for c in cers if c <= args.cer_max)
        avg = sum(cers) / len(cers)
        ok = n_ok * 2 > len(cers)   # maggioranza dei sample sotto soglia
        if ok:
            esito = "OK"
            n_lang_ok += 1
        elif locale in ADAPTATION_TIER:
            esito = "WEAK (tier adattamento: atteso)"
        else:
            esito = "FAIL"
            n_lang_fail += 1
        print(f"{locale:8} {len(entries):>6} {n_ok}/{len(cers):>5} {avg:>10.3f} "
              f"{auto_hits}/{len(entries):>7}  {esito}")

    print(f"\n{n_lang_ok} lingue OK, {n_lang_fail} FAIL (soglia CER {args.cer_max}, criterio: maggioranza)")
    if failures:
        print("\nDettaglio sample sopra soglia:")
        print("\n".join(failures))
    sys.exit(0 if n_lang_fail == 0 else 1)


if __name__ == "__main__":
    main()
