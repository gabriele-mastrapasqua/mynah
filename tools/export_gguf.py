#!/usr/bin/env python3
"""Esporta i pesi convertiti (model.safetensors, f32) in un container GGUF v3.

Il GGUF trasporta SOLO i pesi: mynah.json, mel_filters e tokens restano gli
stessi — il runtime li legge dalla dir del modello come sempre. Uso:

    uv run python export_gguf.py ../models/parakeet-tdt_ctc-110m [--dtype q8_0]

--dtype f32   (default) round-trip bit-esatto, stessa RAM
--dtype f16 | q8_0 | q4_0   file più piccolo; i tensori con l'ultima dim non
        multipla di 32 (bias, norm) restano f32, come fa llama.cpp.
Convenzioni ggml: dims invertite (ne[0] = la più veloce), alignment 32.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

ALIGN = 32
GGML = {"f32": 0, "f16": 1, "q4_0": 2, "q8_0": 8}


def gguf_string(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def quantize_q8_0(x: np.ndarray) -> bytes:
    """Blocchi da 32: d f16 + 32 int8 (q = round(x/d), d = amax/127)."""
    blocks = x.reshape(-1, 32).astype(np.float32)
    amax = np.abs(blocks).max(axis=1)
    d = amax / 127.0
    q = np.where(d[:, None] > 0, np.round(blocks / np.where(d[:, None] > 0, d[:, None], 1)), 0)
    q = np.clip(q, -127, 127).astype(np.int8)
    out = bytearray()
    d16 = d.astype(np.float16)
    for i in range(blocks.shape[0]):
        out += d16[i].tobytes() + q[i].tobytes()
    return bytes(out)


def quantize_q4_0(x: np.ndarray) -> bytes:
    """Blocchi da 32: d f16 + 16 byte nibble (ggml: d = max_signed/-8)."""
    blocks = x.reshape(-1, 32).astype(np.float32)
    idx = np.abs(blocks).argmax(axis=1)
    maxs = blocks[np.arange(blocks.shape[0]), idx]          # col segno, come ggml
    d = maxs / -8.0
    inv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1), 0)
    q = np.clip(np.round(blocks * inv[:, None]) + 8, 0, 15).astype(np.uint8)
    packed = (q[:, :16] | (q[:, 16:] << 4)).astype(np.uint8)
    out = bytearray()
    d16 = d.astype(np.float16)
    for i in range(blocks.shape[0]):
        out += d16[i].tobytes() + packed[i].tobytes()
    return bytes(out)


def encode_tensor(x: np.ndarray, dtype: str) -> tuple[int, bytes]:
    x = np.ascontiguousarray(x, dtype=np.float32)
    if dtype != "f32" and (x.ndim < 2 or x.shape[-1] % 32 != 0):
        # solo le matrici si quantizzano: bias/norm/running-stats 1-D restano f32
        # (stile llama.cpp; quantizzare running_var rompe 1/sqrt(var) del BatchNorm)
        dtype = "f32"
    if dtype == "f32":
        return GGML["f32"], x.tobytes()
    if dtype == "f16":
        return GGML["f16"], x.astype(np.float16).tobytes()
    if dtype == "q8_0":
        return GGML["q8_0"], quantize_q8_0(x)
    return GGML["q4_0"], quantize_q4_0(x)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--dtype", choices=list(GGML), default="f32")
    ap.add_argument("--out", type=Path, default=None, help="default: <model_dir>/model.gguf")
    args = ap.parse_args()

    st_path = args.model_dir / "model.safetensors"
    if not st_path.exists():
        print(f"errore: {st_path} assente (serve il modello convertito)", file=sys.stderr)
        return 1
    name = args.model_dir.name
    cfg = args.model_dir / "mynah.json"
    if cfg.exists():
        name = json.loads(cfg.read_text()).get("name", name)

    tensors = load_file(st_path)
    out_path = args.out or (args.model_dir / "model.gguf")

    meta = (
        gguf_string("general.architecture") + struct.pack("<I", 8) + gguf_string("mynah")
        + gguf_string("general.name") + struct.pack("<I", 8) + gguf_string(name)
        + gguf_string("general.alignment") + struct.pack("<I", 4) + struct.pack("<I", ALIGN)
    )
    header = struct.pack("<IIQQ", 0x46554747, 3, len(tensors), 3) + meta

    infos, payload = bytearray(), bytearray()
    for tname, x in tensors.items():
        ggml_type, blob = encode_tensor(x, args.dtype)
        while len(payload) % ALIGN != 0:
            payload += b"\0"
        dims = list(x.shape)[::-1] or [1]   # convenzione ggml: ne[0] = la più veloce
        infos += gguf_string(tname) + struct.pack("<I", len(dims))
        for d in dims:
            infos += struct.pack("<Q", d)
        infos += struct.pack("<I", ggml_type) + struct.pack("<Q", len(payload))
        payload += blob

    body = header + bytes(infos)
    pad = (ALIGN - len(body) % ALIGN) % ALIGN
    out_path.write_bytes(body + b"\0" * pad + bytes(payload))
    size_mb = out_path.stat().st_size / 1e6
    print(f"OK {out_path} [{args.dtype}] {len(tensors)} tensori, {size_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
