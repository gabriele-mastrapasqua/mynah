"""Bindings Python per libmynah (ctypes, zero dipendenze).

Prerequisito: `make shared` nella root del repo (produce libmynah.dylib/.so).

    from mynah import Mynah
    m = Mynah("models/parakeet-tdt-0.6b-v3")
    print(m.transcribe("audio.wav"))
    text, words = m.transcribe("audio.wav", timestamps=True)
    # traduzione (modelli AED/Canary):
    print(Mynah("models/canary-180m-flash").transcribe("de.wav", lang="de>en"))
"""

from __future__ import annotations

import array
import ctypes
import wave
from pathlib import Path

QUANT = {"f32": 0, "int8": 1, "int4": 2}


def _find_lib() -> ctypes.CDLL:
    here = Path(__file__).resolve()
    candidates = []
    for base in (here.parent, here.parent.parent.parent, Path.cwd()):
        candidates += [base / "libmynah.dylib", base / "libmynah.so"]
    candidates += [Path("libmynah.dylib"), Path("libmynah.so")]
    for c in candidates:
        if c.exists():
            return ctypes.CDLL(str(c))
    raise OSError("libmynah non trovata: compila con `make shared` nella root del repo")


class _Word(ctypes.Structure):
    _fields_ = [("word", ctypes.c_char_p), ("t0", ctypes.c_double), ("t1", ctypes.c_double)]


_lib = None


def _api() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = _find_lib()
        _lib.mynah_load_quant.restype = ctypes.c_void_p
        _lib.mynah_load_quant.argtypes = [ctypes.c_char_p, ctypes.c_int]
        _lib.mynah_free.argtypes = [ctypes.c_void_p]
        _lib.mynah_transcribe_ts.restype = ctypes.c_void_p   # char* (da liberare noi)
        _lib.mynah_transcribe_ts.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
            ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
            ctypes.POINTER(ctypes.POINTER(_Word)), ctypes.POINTER(ctypes.c_int)]
        _lib.mynah_words_free.argtypes = [ctypes.POINTER(_Word), ctypes.c_int]
        _lib.mynah_set_target_lang.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        _lib.mynah_can_translate.argtypes = [ctypes.c_void_p]
        _lib.mynah_set_segment_limit.argtypes = [ctypes.c_void_p, ctypes.c_double]
        _lib.mynah_resample.restype = ctypes.POINTER(ctypes.c_float)
        _lib.mynah_resample.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
                                        ctypes.c_int, ctypes.c_int,
                                        ctypes.POINTER(ctypes.c_size_t)]
        _lib.mynah_version.restype = ctypes.c_char_p
    return _lib


def _load_wav(path: str) -> tuple[array.array, int]:
    """WAV PCM16 -> float32 [-1,1] mono (media dei canali) + sample rate."""
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("serve WAV PCM16 (per mp3/altro: ffmpeg -ar 16000 -ac 1)")
        nch, sr, n = w.getnchannels(), w.getframerate(), w.getnframes()
        pcm = array.array("h")
        pcm.frombytes(w.readframes(n))
    out = array.array("f", [0.0]) * (len(pcm) // nch)
    for i in range(len(out)):
        s = 0
        for c in range(nch):
            s += pcm[i * nch + c]
        out[i] = s / nch / 32768.0
    return out, sr


class Mynah:
    """Un modello caricato. Thread-safety: usare da un thread alla volta."""

    def __init__(self, model_dir: str, quant: str = "f32"):
        self._lib = _api()
        self._m = self._lib.mynah_load_quant(str(model_dir).encode(), QUANT[quant])
        if not self._m:
            raise RuntimeError(f"load fallita: {model_dir}")

    def close(self) -> None:
        if self._m:
            self._lib.mynah_free(self._m)
            self._m = None

    def __del__(self):  # noqa: D105
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def set_target_lang(self, lang: str) -> None:
        """AED/Canary: lingua di uscita (≠ sorgente = traduzione). '' = ASR."""
        if self._lib.mynah_set_target_lang(self._m, lang.encode()) != 0:
            raise ValueError(f"target lang non supportata: {lang}")

    def set_segment_limit(self, sec: float) -> None:
        self._lib.mynah_set_segment_limit(self._m, sec)

    def transcribe(self, wav: str, lang: str = "auto", lookahead: int = -1,
                   timestamps: bool = False):
        """Trascrive un WAV (resample automatico). lang accetta anche "src>tgt"
        per la traduzione AED. Ritorna str, o (str, [(word, t0, t1), ...])
        con timestamps=True."""
        samples, sr = _load_wav(wav)
        buf = (ctypes.c_float * len(samples)).from_buffer(samples)
        n = ctypes.c_size_t(len(samples))
        if sr != 16000:
            n_out = ctypes.c_size_t()
            p = self._lib.mynah_resample(buf, len(samples), sr, 16000,
                                         ctypes.byref(n_out))
            if not p:
                raise RuntimeError("resampling fallito")
            buf, n = p, n_out
        lang_out = ctypes.create_string_buffer(16)
        words_p = ctypes.POINTER(_Word)()
        n_words = ctypes.c_int(0)
        raw = self._lib.mynah_transcribe_ts(
            self._m, buf, n, lang.encode(), lookahead, lang_out,
            ctypes.byref(words_p) if timestamps else None,
            ctypes.byref(n_words) if timestamps else None)
        if not raw:
            raise RuntimeError("trascrizione fallita (lingua non supportata?)")
        text = ctypes.cast(raw, ctypes.c_char_p).value.decode()
        libc = ctypes.CDLL(None)
        libc.free(ctypes.c_void_p(raw))
        if not timestamps:
            return text
        words = [(words_p[i].word.decode(), words_p[i].t0, words_p[i].t1)
                 for i in range(n_words.value)]
        self._lib.mynah_words_free(words_p, n_words)
        return text, words


def version() -> str:
    return _api().mynah_version().decode()
