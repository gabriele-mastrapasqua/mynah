'use strict';
// Bindings Node per libmynah (koffi, FFI puro — NESSUNO step di build nel repo).
//
// Prerequisiti:
//   1) `make shared` nella root del repo  -> libmynah.dylib / libmynah.so
//   2) `npm i koffi`                        (l'utente, dove vuole: nessuna build nativa)
//
//   const { Mynah } = require('./mynah');
//   const m = new Mynah('models/parakeet-tdt-0.6b-v3');
//   console.log(m.transcribe('audio.wav'));
//   const { text, words } = m.transcribe('audio.wav', { timestamps: true });
//   // traduzione (modelli AED/Canary): lang "src>tgt"
//   console.log(new Mynah('models/canary-180m-flash').transcribe('de.wav', { lang: 'de>en' }));
//   m.close();
//
// Gemello di bindings/python/mynah.py: stessa superficie, stessi 7 simboli.

const fs = require('fs');
const path = require('path');

let koffi;
try {
  koffi = require('koffi');
} catch {
  throw new Error("koffi non installato: `npm i koffi` (solo FFI, nessuna build nativa)");
}

const QUANT = { f32: 0, int8: 1, int4: 2 };

function findLib() {
  const names = process.platform === 'darwin' ? ['libmynah.dylib']
              : process.platform === 'win32' ? ['mynah.dll', 'libmynah.dll']
              : ['libmynah.so'];
  const bases = [__dirname, path.resolve(__dirname, '..', '..'), process.cwd()];
  for (const b of bases) {
    for (const n of names) {
      const p = path.join(b, n);
      if (fs.existsSync(p)) return p;
    }
  }
  throw new Error('libmynah non trovata: compila con `make shared` nella root del repo');
}

const lib = koffi.load(findLib());

// libc.free: il testo restituito da mynah_transcribe_ts è malloc'd dal runtime e
// va liberato con la stessa free (come il ctypes.CDLL(None) del binding Python).
const libc = koffi.load(
  process.platform === 'darwin' ? 'libSystem.B.dylib'
  : process.platform === 'win32' ? 'msvcrt.dll'
  : 'libc.so.6');
const cfree = libc.func('void free(void *ptr)');

const Word = koffi.struct('mynah_word', { word: 'char *', t0: 'double', t1: 'double' });

const fn = {
  load_quant: lib.func('void *mynah_load_quant(const char *dir, int quant)'),
  free: lib.func('void mynah_free(void *m)'),
  version: lib.func('const char *mynah_version()'),
  set_target_lang: lib.func('int mynah_set_target_lang(void *m, const char *lang)'),
  can_translate: lib.func('int mynah_can_translate(void *m)'),
  set_segment_limit: lib.func('void mynah_set_segment_limit(void *m, double sec)'),
  words_free: lib.func('void mynah_words_free(mynah_word *w, int n)'),
  resample: lib.func(
    'float *mynah_resample(float *in, size_t n, int sr_in, int sr_out, _Out_ size_t *n_out)'),
  // stesso simbolo, due prototipi: solo-testo (words = NULL) e con timestamp.
  // lang_out e' un buffer di uscita: uint8_t* (puntatore vero) e non char* — che
  // koffi tratterebbe come stringa d'ingresso senza ricopiare la scrittura del C.
  transcribe_text: lib.func(
    'void *mynah_transcribe_ts(void *m, float *s, size_t n, const char *lang, int la, ' +
    'uint8_t *lang_out, void *words, void *n_words)'),
  transcribe_ts: lib.func(
    'void *mynah_transcribe_ts(void *m, float *s, size_t n, const char *lang, int la, ' +
    'uint8_t *lang_out, _Out_ mynah_word **words, _Out_ int *n_words)'),
};

// ---- WAV PCM16 -> Float32Array mono [-1,1] + sample rate (parser minimale) ----
function loadWav(file) {
  const b = fs.readFileSync(file);
  if (b.toString('latin1', 0, 4) !== 'RIFF' || b.toString('latin1', 8, 12) !== 'WAVE') {
    throw new Error('non è un WAV RIFF/WAVE');
  }
  let off = 12, fmt = null, data = null;
  while (off + 8 <= b.length) {
    const id = b.toString('latin1', off, off + 4);
    const size = b.readUInt32LE(off + 4);
    const body = off + 8;
    if (id === 'fmt ') {
      fmt = {
        audioFormat: b.readUInt16LE(body),
        channels: b.readUInt16LE(body + 2),
        sampleRate: b.readUInt32LE(body + 4),
        bits: b.readUInt16LE(body + 14),
      };
    } else if (id === 'data') {
      data = b.subarray(body, body + size);
    }
    off = body + size + (size & 1); // i chunk sono word-aligned
  }
  if (!fmt || !data) throw new Error('WAV senza chunk fmt/data');
  if (fmt.bits !== 16) throw new Error('serve WAV PCM16 (per mp3/altro: ffmpeg -ar 16000 -ac 1)');
  const nch = fmt.channels;
  const nSamp = Math.floor(data.length / 2 / nch);
  const out = new Float32Array(nSamp);
  for (let i = 0; i < nSamp; i++) {
    let s = 0;
    for (let c = 0; c < nch; c++) s += data.readInt16LE((i * nch + c) * 2);
    out[i] = s / nch / 32768.0;
  }
  return { samples: out, sampleRate: fmt.sampleRate };
}

class Mynah {
  // Un modello caricato. Thread-safety: usare da un thread alla volta.
  constructor(modelDir, quant = 'f32') {
    if (!(quant in QUANT)) throw new Error(`quant sconosciuto: ${quant}`);
    this._m = fn.load_quant(String(modelDir), QUANT[quant]);
    if (!this._m) throw new Error(`load fallita: ${modelDir}`);
  }

  close() {
    if (this._m) { fn.free(this._m); this._m = null; }
  }

  setTargetLang(lang) {
    // AED/Canary: lingua di uscita (≠ sorgente = traduzione). '' = ASR.
    if (fn.set_target_lang(this._m, lang) !== 0) throw new Error(`target lang non supportata: ${lang}`);
  }

  canTranslate() { return fn.can_translate(this._m) === 1; }

  setSegmentLimit(sec) { fn.set_segment_limit(this._m, sec); }

  // Trascrive un WAV (resample automatico a 16 kHz). opts: { lang, lookahead, timestamps }.
  // lang accetta "src>tgt" per la traduzione AED. Ritorna string, oppure
  // { text, words: [{ word, t0, t1 }, ...] } con timestamps: true.
  transcribe(wav, opts = {}) {
    const { lang = 'auto', lookahead = -1, timestamps = false } = opts;
    let { samples, sampleRate } = loadWav(wav);
    let n = samples.length;

    if (sampleRate !== 16000) {
      const nOut = [0n];
      const p = fn.resample(samples, n, sampleRate, 16000, nOut);
      if (!p) throw new Error('resampling fallito');
      const count = Number(nOut[0]);
      samples = Float32Array.from(koffi.decode(p, 'float', count));
      cfree(p);
      n = count;
    }

    const langOut = Buffer.alloc(16);
    let raw;
    let words = null;
    if (timestamps) {
      const wp = [null];
      const nw = [0];
      raw = fn.transcribe_ts(this._m, samples, n, lang, lookahead, langOut, wp, nw);
      if (!raw) throw new Error('trascrizione fallita (lingua non supportata?)');
      if (wp[0] && nw[0] > 0) {
        const arr = koffi.decode(wp[0], 'mynah_word', nw[0]);
        words = arr.map((w) => ({ word: w.word, t0: w.t0, t1: w.t1 }));
        fn.words_free(wp[0], nw[0]);
      } else {
        words = [];
      }
    } else {
      raw = fn.transcribe_text(this._m, samples, n, lang, lookahead, langOut, null, null);
      if (!raw) throw new Error('trascrizione fallita (lingua non supportata?)');
    }

    const text = koffi.decode(raw, 'char', -1); // stringa C NUL-terminata dal puntatore
    cfree(raw);                                 // liberata con la free del runtime (malloc)
    return timestamps ? { text, words, lang: cstr(langOut) } : text;
  }
}

function cstr(buf) {
  const z = buf.indexOf(0);
  return buf.toString('utf8', 0, z < 0 ? buf.length : z);
}

function version() {
  return fn.version();
}

module.exports = { Mynah, version, loadWav };
