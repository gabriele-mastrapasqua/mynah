#!/bin/sh
# End-to-end: mynah transcribe sui WAV fixture, confronto col testo atteso.
# Testi attesi per engine (mynah.json): nemotron-streaming o parakeet-tdt
# (Parakeet: niente --lang, ITN attiva -> numeri in cifre).
# Exit: 0 ok, 1 mismatch, 77 skip (modello assente).
MODEL_DIR="${1:-models/nemotron-3.5-asr-streaming-0.6b}"
[ -f "$MODEL_DIR/mynah.json" ] || exit 77
ENGINE=$(sed -n 's/.*"engine": "\([^"]*\)".*/\1/p' "$MODEL_DIR/mynah.json")

fail=0
check() { # wav, lang, expected substring
    out=$(./mynah transcribe -m "$MODEL_DIR" -i "$1" --lang "$2" 2>/dev/null)
    case "$out" in
        *"$3"*) printf 'e2e %-16s OK: %s\n' "$(basename "$1")/$2" "$out" ;;
        *) printf 'e2e %-16s FAIL:\n  atteso  ~ %s\n  ottenuto: %s\n' "$(basename "$1")/$2" "$3" "$out"; fail=1 ;;
    esac
}

if [ "$ENGINE" = "parakeet-tdt" ]; then
    IT_SUB="riconoscimento vocale in italiano"
    check tests/audio/test_it.wav auto "Ciao, questo è un test di riconoscimento vocale in italiano: il gatto dorme sul divano."
    check tests/audio/test_en.wav auto "Hello, this is a speech recognition test, the weather is nice today."
    check tests/audio/test_de.wav auto "die Besprechung beginnt um 9 Uhr im großen Saal"
    check tests/audio/test_fr.wav auto "la réunion commence à 9h dans la grande salle"
    check tests/audio/test_es.wav auto "la reunión empieza a las 9 en la sala grande"
else
    IT_SUB="riconoscimento vocale in italiano"
    check tests/audio/test_it.wav auto  "Ciao, questo è un test di riconoscimento vocale in italiano."
    check tests/audio/test_it.wav it-IT "Il gatto dorme sul divano."
    check tests/audio/test_en.wav auto  "Hello, this is a speech recognition test."
    check tests/audio/test_de.wav auto  "die Besprechung beginnt um neun Uhr"
    check tests/audio/test_fr.wav auto  "la réunion commence à neuf heures"
    check tests/audio/test_es.wav auto  "la reunión empieza"
fi

# checkpoint pre-quantizzati (se generati con: mynah quantize)
checkq() { # quant, expected substring
    out=$(./mynah transcribe -m "$MODEL_DIR" -i tests/audio/test_it.wav --quant "$1" 2>/dev/null)
    case "$out" in
        *"$2"*) printf 'e2e quant-%-6s OK: %s\n' "$1" "$out" ;;
        *) printf 'e2e quant-%-6s FAIL: %s\n' "$1" "$out"; fail=1 ;;
    esac
}
[ -f "$MODEL_DIR/model.int8.safetensors" ] && checkq int8 "$IT_SUB"
[ -f "$MODEL_DIR/model.int4.safetensors" ] && checkq int4 "$IT_SUB"

# timestamp per parola: righe "t0 t1 parola", t0 monotono non-decrescente,
# t1 entro la durata dell'audio (5.2s + margine di un frame)
ts=$(./mynah transcribe -m "$MODEL_DIR" -i tests/audio/test_it.wav --timestamps 2>/dev/null)
ts_ok=$(printf '%s\n' "$ts" | awk 'NF<3 {bad=1} $1+0>$2+0 {bad=1} $1+0<prev {bad=1}
    {prev=$1+0; n++} END {print (bad || n<5 || prev>5.3) ? "FAIL" : "OK"}')
if [ "$ts_ok" = "OK" ]; then
    echo "e2e timestamps OK: $(printf '%s\n' "$ts" | wc -l | tr -d ' ') parole"
else
    echo "e2e timestamps FAIL:"; printf '%s\n' "$ts"; fail=1
fi

# backend Metal (solo macOS; per i modelli non-causali il kernel Metal non si
# applica e il gate ricade su CPU: il check verifica comunque il testo)
out=$(./mynah transcribe -m "$MODEL_DIR" -i tests/audio/test_it.wav --backend metal 2>/dev/null)
case "$out" in
    *"$IT_SUB"*) echo "e2e backend-metal OK: $out" ;;
    *) echo "e2e backend-metal FAIL: $out"; fail=1 ;;
esac

exit $fail
