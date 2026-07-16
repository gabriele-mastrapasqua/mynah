#!/usr/bin/env bash
# Scarica un modello dal repo HuggingFace nel layout locale models/<nome>/.
# Non interattivo (lezione qwen-tts). Default: il target v1.
#
# Uso: scripts/download_model.sh [repo_id] [dest_dir]
set -euo pipefail

REPO_ID="${1:-nvidia/nemotron-3.5-asr-streaming-0.6b}"
NAME="$(basename "$REPO_ID")"
DEST="${2:-models/$NAME}"
BASE="https://huggingface.co/$REPO_ID/resolve/main"

# File del porting HF-native (per Nemotron/Parakeet recenti). Il .nemo (2x più grande,
# stesso contenuto) NON serve al runtime: config e tokenizer sono già estratti in reference/.
FILES=(config.json generation_config.json processor_config.json tokenizer_config.json tokenizer.json model.safetensors)

mkdir -p "$DEST"
for f in "${FILES[@]}"; do
  if [[ -f "$DEST/$f" ]]; then
    echo "skip  $f (già presente)"
    continue
  fi
  echo "fetch $f"
  curl -fL --retry 3 --progress-bar "$BASE/$f" -o "$DEST/$f.part"
  mv "$DEST/$f.part" "$DEST/$f"
done
echo "OK → $DEST"
