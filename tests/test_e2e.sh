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

exit $fail
