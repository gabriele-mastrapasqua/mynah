#!/bin/sh
# End-to-end: mynah transcribe sui WAV fixture, confronto col testo atteso.
# Exit: 0 ok, 1 mismatch, 77 skip (modello assente).
MODEL_DIR="${1:-models/nemotron-3.5-asr-streaming-0.6b}"
[ -f "$MODEL_DIR/mynah.json" ] || exit 77

fail=0
check() { # wav, lang, expected substring
    out=$(./mynah transcribe -m "$MODEL_DIR" -i "$1" --lang "$2" 2>/dev/null)
    case "$out" in
        *"$3"*) printf 'e2e %-16s OK: %s\n' "$(basename "$1")/$2" "$out" ;;
        *) printf 'e2e %-16s FAIL:\n  atteso  ~ %s\n  ottenuto: %s\n' "$(basename "$1")/$2" "$3" "$out"; fail=1 ;;
    esac
}

check tests/audio/test_it.wav auto  "Ciao, questo è un test di riconoscimento vocale in italiano."
check tests/audio/test_it.wav it-IT "Il gatto dorme sul divano."
check tests/audio/test_en.wav auto  "Hello, this is a speech recognition test."
check tests/audio/test_de.wav auto  "die Besprechung beginnt um neun Uhr"
check tests/audio/test_fr.wav auto  "la réunion commence à neuf heures"
check tests/audio/test_es.wav auto  "la reunión empieza"

# checkpoint pre-quantizzati (se generati con: mynah quantize)
checkq() { # quant, expected substring
    out=$(./mynah transcribe -m "$MODEL_DIR" -i tests/audio/test_it.wav --lang it-IT --quant "$1" 2>/dev/null)
    case "$out" in
        *"$2"*) printf 'e2e quant-%-6s OK: %s\n' "$1" "$out" ;;
        *) printf 'e2e quant-%-6s FAIL: %s\n' "$1" "$out"; fail=1 ;;
    esac
}
[ -f "$MODEL_DIR/model.int8.safetensors" ] && checkq int8 "riconoscimento vocale in italiano"
[ -f "$MODEL_DIR/model.int4.safetensors" ] && checkq int4 "riconoscimento vocale in italiano"

# backend Metal (solo macOS; fallback CPU altrove rende il check comunque valido)
out=$(./mynah transcribe -m "$MODEL_DIR" -i tests/audio/test_it.wav --lang it-IT --backend metal 2>/dev/null)
case "$out" in
    *"riconoscimento vocale in italiano"*) echo "e2e backend-metal OK: $out" ;;
    *) echo "e2e backend-metal FAIL: $out"; fail=1 ;;
esac

exit $fail
