#!/usr/bin/env python3
"""Converte un checkpoint HF-native NeMo ASR nel layout Mynah (in-place).

Input: directory con il porting HuggingFace del modello
  (config.json, processor_config.json, tokenizer.json, model.safetensors).
Output (scritto NELLA stessa directory, i pesi non vengono toccati):
  - mynah.json              config unificata che guida il runtime C
  - tokens.json             array id -> piece (incluso il blank in coda)
  - mel_filters.safetensors filterbank mel (slaney) + finestra Hann precomputate

Uso: uv run python convert_nemo.py ../models/nemotron-3.5-asr-streaming-0.6b
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


# ---------------------------------------------------------------- mel filters
def hz_to_mel_slaney(f: np.ndarray) -> np.ndarray:
    """Scala mel 'slaney' (librosa/torchaudio htk=False): lineare sotto 1 kHz."""
    f = np.asarray(f, dtype=np.float64)
    mel = 3.0 * f / 200.0
    log_region = f >= 1000.0
    mel = np.where(log_region, 15.0 + 27.0 * np.log(np.maximum(f, 1e-10) / 1000.0) / np.log(6.4), mel)
    return mel


def mel_to_hz_slaney(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    f = 200.0 * m / 3.0
    log_region = m >= 15.0
    f = np.where(log_region, 1000.0 * np.exp(np.log(6.4) * (m - 15.0) / 27.0), f)
    return f


def melscale_fbanks(n_freqs: int, f_min: float, f_max: float, n_mels: int, sample_rate: int) -> np.ndarray:
    """Clone di torchaudio.functional.melscale_fbanks(norm='slaney', mel_scale='slaney').

    Identica alla costruzione NeMo (librosa htk=False, norm slaney); vedi
    docs/prior-art.md §B.2 (onnx-asr la valida vs NeMo con atol 5e-7).
    Ritorna [n_freqs, n_mels] float64.
    """
    all_freqs = np.linspace(0.0, sample_rate // 2, n_freqs, dtype=np.float64)
    m_pts = mel_to_hz_slaney(np.linspace(hz_to_mel_slaney(f_min), hz_to_mel_slaney(f_max), n_mels + 2))
    f_diff = m_pts[1:] - m_pts[:-1]                        # [n_mels+1]
    slopes = m_pts[None, :] - all_freqs[:, None]           # [n_freqs, n_mels+2]
    down = -slopes[:, :-2] / f_diff[:-1]                   # [n_freqs, n_mels]
    up = slopes[:, 2:] / f_diff[1:]
    fb = np.maximum(0.0, np.minimum(down, up))
    fb *= 2.0 / (m_pts[2:] - m_pts[:-2])                   # norm slaney
    return fb


# ------------------------------------------------------------------ tokenizer
def load_pieces(model_dir: Path, vocab_size: int, blank_id: int, blank_token: str) -> list[str]:
    """Estrae l'array id -> piece dal tokenizer.json (formato HF tokenizers).

    Nota (verificato sul checkpoint): il tokenizer.json elenca <pad>=13087 e <blank>=13088,
    ma nello spazio di output del modello (joint = vocab_size=13088 logit, blank_as_pad)
    l'id 13087 È il blank/pad e 13088 non esiste. Normalizziamo: pieces[blank_id]=blank_token,
    gli added token oltre vocab_size vengono ignorati.
    """
    tok = json.loads((model_dir / "tokenizer.json").read_text())
    pieces: list[str | None] = [None] * vocab_size
    for piece, idx in tok["model"]["vocab"].items():
        pieces[idx] = piece
    for added in tok.get("added_tokens", []):
        if added["id"] < blank_id:
            pieces[added["id"]] = added["content"]
    pieces[blank_id] = blank_token
    missing = [i for i, p in enumerate(pieces) if p is None]
    assert not missing, f"{len(missing)} id senza piece (primi: {missing[:5]})"
    return pieces  # type: ignore[return-value]


# ----------------------------------------------------------------------- main
def convert(model_dir: Path) -> None:
    cfg = json.loads((model_dir / "config.json").read_text())
    proc = json.loads((model_dir / "processor_config.json").read_text())
    tok_cfg = json.loads((model_dir / "tokenizer_config.json").read_text())
    enc = cfg["encoder_config"]
    fe = proc["feature_extractor"]

    assert cfg["model_type"] == "nemotron3_5_asr", f"model_type inatteso: {cfg['model_type']}"
    assert fe["feature_size"] == enc["num_mel_bins"]

    vocab_size = cfg["vocab_size"]          # 13088 = 13087 pieces + blank
    blank_id = cfg["blank_token_id"]        # 13087
    sub = enc["subsampling_factor"]
    left_context = enc["sliding_window"] - 1
    lookaheads = enc["supported_num_lookahead_tokens"]
    frame_ms = fe["hop_length"] / fe["sampling_rate"] * sub * 1000.0

    mynah = {
        "mynah_format": 1,
        "name": model_dir.name,
        "arch": "fastconformer_rnnt_streaming",
        "engine": "nemotron-streaming",
        "weights": "model.safetensors",
        "features": {
            "sample_rate": fe["sampling_rate"],
            "n_mels": fe["feature_size"],
            "n_fft": fe["n_fft"],
            "win_length": fe["win_length"],
            "hop_length": fe["hop_length"],
            "preemphasis": fe["preemphasis"],
            "log_zero_guard": 2.0 ** -24,
            "normalize": "NA",              # dal model_config.yaml nel .nemo (vedi docs/nemotron-arch.md)
            "dither": 0.0,                  # inference
            "mel_filters": "mel_filters.safetensors",
        },
        "encoder": {
            "n_layers": enc["num_hidden_layers"],
            "d_model": enc["hidden_size"],
            "n_heads": enc["num_attention_heads"],
            "ffn_dim": enc["intermediate_size"],
            "conv_kernel": enc["conv_kernel_size"],
            "conv_norm": "layer_norm",
            "subsampling": "dw_striding_causal",
            "subsampling_factor": sub,
            "subsampling_conv_channels": enc["subsampling_conv_channels"],
            "use_bias": enc["attention_bias"],
            "activation": enc["hidden_act"],          # silu
            "att_context_style": "chunked_limited",
            "pos_emb_max_len": enc["max_position_embeddings"],
            "xscaling": False,
        },
        "decoder": {
            "type": "rnnt_lstm",
            "pred_hidden": cfg["decoder_hidden_size"],
            "pred_layers": cfg["num_decoder_layers"],
            "joint_hidden": cfg["decoder_hidden_size"],
            "joint_activation": cfg["hidden_act"],    # relu
            "vocab_size": vocab_size,
            "blank_id": blank_id,
            "max_symbols_per_step": cfg["max_symbols_per_step"],
        },
        "streaming": {
            "att_context_presets": [[left_context, r] for r in lookaheads],
            "default_preset_index": lookaheads.index(enc["default_num_lookahead_tokens"]),
            "encoder_frame_ms": frame_ms,
        },
        "prompt": {
            "num_prompts": cfg["num_prompts"],
            "default_id": cfg["default_prompt_id"],
            "intermediate_size": cfg["prompt_intermediate_size"],
            "dictionary": proc["prompt_dictionary"],
        },
        "tokenizer": {"type": "spe_bpe", "pieces": "tokens.json"},
    }

    pieces = load_pieces(model_dir, vocab_size, blank_id, tok_cfg.get("blank_token", "<blank>"))

    fb = melscale_fbanks(fe["n_fft"] // 2 + 1, 0.0, fe["sampling_rate"] / 2, fe["feature_size"], fe["sampling_rate"])
    window = np.hanning(fe["win_length"])  # simmetrica, come torch.hann_window(periodic=False)

    (model_dir / "mynah.json").write_text(json.dumps(mynah, indent=1, ensure_ascii=False))
    (model_dir / "tokens.json").write_text(json.dumps(pieces, ensure_ascii=False))
    save_file(
        {"mel_fb": fb.astype(np.float32), "window": window.astype(np.float32)},
        model_dir / "mel_filters.safetensors",
    )
    print(f"OK {model_dir.name}: mynah.json, tokens.json ({len(pieces)} pieces), "
          f"mel_filters.safetensors (fb {fb.shape}, window {window.shape})")
    print(f"   presets latenza: {mynah['streaming']['att_context_presets']} "
          f"(default idx {mynah['streaming']['default_preset_index']}, frame {frame_ms:.0f} ms)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    convert(Path(sys.argv[1]).resolve())
