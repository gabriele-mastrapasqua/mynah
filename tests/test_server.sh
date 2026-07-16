#!/bin/sh
# Test end-to-end di mynah-server: REST + concorrenza + WebSocket streaming.
# Exit: 0 ok, 1 fail, 77 skip (modello assente).
MODEL_DIR="${1:-models/nemotron-3.5-asr-streaming-0.6b}"
PORT="${2:-8199}"
[ -f "$MODEL_DIR/mynah.json" ] || exit 77

./mynah-server -m "$MODEL_DIR" -p "$PORT" --threads 4 --batch 4 2>/dev/null &
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null' EXIT

# attesa readiness
for i in $(seq 1 50); do
    curl -sf "http://localhost:$PORT/v1/health" >/dev/null 2>&1 && break
    sleep 0.2
done

fail=0
check() { # descrizione, comando, substring attesa
    out=$(eval "$2" 2>/dev/null)
    case "$out" in
        *"$3"*) echo "server $1 OK" ;;
        *) echo "server $1 FAIL: $out"; fail=1 ;;
    esac
}

check health "curl -s http://localhost:$PORT/v1/health" '"status":"ok"'
check models "curl -s http://localhost:$PORT/v1/models" 'nemotron'
check transcribe-multipart \
    "curl -s -F file=@tests/audio/test_it.wav -F language=it-IT http://localhost:$PORT/v1/audio/transcriptions" \
    "riconoscimento vocale in italiano"
check transcribe-verbose \
    "curl -s -F file=@tests/audio/test_en.wav -F language=auto -F response_format=verbose_json http://localhost:$PORT/v1/audio/transcriptions" \
    '"language":"en-US"'
check transcribe-raw-body \
    "curl -s -X POST --data-binary @tests/audio/test_it.wav -H 'Content-Type: audio/wav' http://localhost:$PORT/v1/audio/transcriptions" \
    "riconoscimento vocale"
check error-400 \
    "curl -s -X POST --data-binary 'garbage' http://localhost:$PORT/v1/audio/transcriptions" \
    '"error"'

# 4 richieste concorrenti (wait SOLO sulle curl: wait nudo aspetterebbe anche il server)
CURL_PIDS=""
for i in 1 2 3 4; do
    curl -s -F file=@tests/audio/test_it.wav -F language=it-IT \
        "http://localhost:$PORT/v1/audio/transcriptions" -o "/tmp/mynah_conc_$i.json" &
    CURL_PIDS="$CURL_PIDS $!"
done
wait $CURL_PIDS
conc_ok=1
for i in 1 2 3 4; do
    grep -q "riconoscimento vocale" "/tmp/mynah_conc_$i.json" || conc_ok=0
done
[ $conc_ok -eq 1 ] && echo "server concurrent-4 OK" || { echo "server concurrent-4 FAIL"; fail=1; }

# WebSocket streaming (richiede tools/ con uv)
if command -v uv >/dev/null 2>&1; then
    ws_out=$(cd tools && uv run python -m eval.ws_client ../tests/audio/test_it.wav localhost "$PORT" auto 3 2>/dev/null)
    case "$ws_out" in
        *"riconoscimento vocale in italiano"*"done"*) echo "server ws-stream OK" ;;
        *) echo "server ws-stream FAIL: $ws_out"; fail=1 ;;
    esac
else
    echo "server ws-stream SKIP (uv assente)"
fi

exit $fail
