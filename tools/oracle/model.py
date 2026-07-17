"""Oracolo FastConformer in numpy puro (forward offline + greedy RNNT/TDT),
config-driven dal mynah.json del modello convertito.

Replica 1:1 le reference transformers:
- Nemotron (reference/transformers-nemotron_asr_streaming/): subsampling/conv causali,
  attention chunked_limited, conv norm layer_norm, prompt post-encoder, greedy RNNT.
- Parakeet TDT (reference/transformers-parakeet/): tutto non-causale (padding simmetrico),
  attention full, conv norm batch_norm (inference: affine con running stats),
  niente prompt, greedy TDT (duration head).

Scopo: riferimento numerico leggibile per il runtime C (dump per stadio via `dumps`),
NON performance. Convenzioni: pesi dal model.safetensors HF con nomi verbatim;
attivazioni [T, d]; batch = 1 implicito.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def layer_norm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


class Oracle:
    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        self.cfg = json.loads((model_dir / "mynah.json").read_text())
        self.pieces: list[str] = json.loads((model_dir / "tokens.json").read_text())
        f = safe_open(model_dir / self.cfg["weights"], framework="numpy")
        self.w = {k: f.get_tensor(k) for k in f.keys()}

        enc = self.cfg["encoder"]
        dec = self.cfg["decoder"]
        self.n_layers = enc["n_layers"]
        self.d_model = enc["d_model"]
        self.n_heads = enc["n_heads"]
        self.d_head = self.d_model // self.n_heads
        self.conv_k = enc["conv_kernel"]
        self.causal = "causal" in enc["subsampling"]
        self.conv_norm = enc.get("conv_norm", "layer_norm")
        self.blank = dec["blank_id"]
        self.max_symbols = dec["max_symbols_per_step"]
        self.durations: list[int] = dec.get("durations", [])   # TDT; [] = RNNT
        self.has_prompt = "prompt" in self.cfg
        # inv_freq del rel pos encoding (interleaved sin/cos)
        self.inv_freq = 1.0 / (10000.0 ** (np.arange(0, self.d_model, 2, dtype=np.float64) / self.d_model))

        # alias nomi subsampling: Nemotron usa conv_in/layers.{i}.{depthwise,pointwise}_conv,
        # il porting HF di Parakeet un Sequential piatto layers.{0,2,3,5,6}
        if "encoder.subsampling.conv_in.weight" in self.w:
            self.ss_names = {
                "conv_in": "encoder.subsampling.conv_in",
                "dw": ["encoder.subsampling.layers.0.depthwise_conv",
                       "encoder.subsampling.layers.1.depthwise_conv"],
                "pw": ["encoder.subsampling.layers.0.pointwise_conv",
                       "encoder.subsampling.layers.1.pointwise_conv"],
            }
        else:
            self.ss_names = {
                "conv_in": "encoder.subsampling.layers.0",
                "dw": ["encoder.subsampling.layers.2", "encoder.subsampling.layers.5"],
                "pw": ["encoder.subsampling.layers.3", "encoder.subsampling.layers.6"],
            }

    # ------------------------------------------------------------- subsampling
    def _conv2d_s2(self, x, weight, bias, groups=1):
        """x [C_in, T, F]; stride 2. Padding causale (2,1)x(2,1) o simmetrico (1,1)x(1,1)
        secondo self.causal. Ritorna [C_out, T', F']."""
        k = weight.shape[2]
        stride = 2
        pad = (k - 1, stride - 1) if self.causal else ((k - 1) // 2, (k - 1) // 2)
        x = np.pad(x, ((0, 0), pad, pad))
        C_in, T, F = x.shape
        C_out = weight.shape[0]
        To = (T - k) // stride + 1
        Fo = (F - k) // stride + 1
        # im2col semplice (oracolo: chiarezza > velocità)
        out = np.zeros((C_out, To, Fo), dtype=np.float64)
        if groups == 1:
            cols = np.zeros((C_in * k * k, To * Fo))
            idx = 0
            for dt in range(k):
                for df in range(k):
                    patch = x[:, dt:dt + To * stride:stride, df:df + Fo * stride:stride]
                    cols[idx * C_in:(idx + 1) * C_in] = patch.reshape(C_in, -1)
                    idx += 1
            wmat = weight.transpose(2, 3, 1, 0).reshape(k * k * C_in, C_out)  # [dt,df,cin] -> flat
            out = (cols.T @ wmat).T.reshape(C_out, To, Fo)
        else:  # depthwise: groups == C_in == C_out
            for dt in range(k):
                for df in range(k):
                    patch = x[:, dt:dt + To * stride:stride, df:df + Fo * stride:stride]
                    out += weight[:, 0, dt, df][:, None, None] * patch.reshape(C_in, To, Fo)
        return out + bias[:, None, None]

    def subsampling(self, feats: np.ndarray) -> np.ndarray:
        """feats [T, n_mels] (solo frame validi) -> [T', d_model]."""
        w = self.w
        nm = self.ss_names
        x = feats.astype(np.float64)[None, :, :]                      # [1, T, F]
        x = np.maximum(self._conv2d_s2(x, w[nm["conv_in"] + ".weight"],
                                       w[nm["conv_in"] + ".bias"]), 0.0)
        for i in range(2):
            x = self._conv2d_s2(x, w[nm["dw"][i] + ".weight"], w[nm["dw"][i] + ".bias"],
                                groups=x.shape[0])
            # pointwise 1x1
            pw = w[nm["pw"][i] + ".weight"][:, :, 0, 0]                # [C_out, C_in]
            x = np.einsum("oc,ctf->otf", pw, x) + w[nm["pw"][i] + ".bias"][:, None, None]
            x = np.maximum(x, 0.0)
        C, T, F = x.shape
        x = x.transpose(1, 0, 2).reshape(T, C * F)                     # [T, C*F] channel-major
        return x @ w["encoder.subsampling.linear.weight"].T.astype(np.float64) + w["encoder.subsampling.linear.bias"]

    # ---------------------------------------------------------------- encoder
    def rel_pos_emb(self, L: int) -> np.ndarray:
        """[2L-1, d_model], posizioni L-1 .. -(L-1), interleaved [sin, cos, ...]."""
        pos = np.arange(L - 1, -L, -1, dtype=np.float64)
        freqs = pos[:, None] * self.inv_freq[None, :]                  # [2L-1, d/2]
        emb = np.stack([np.sin(freqs), np.cos(freqs)], axis=-1).reshape(2 * L - 1, self.d_model)
        return emb

    @staticmethod
    def _rel_shift(bd: np.ndarray) -> np.ndarray:
        """bd [H, T, P] -> shift Transformer-XL -> [H, T, P]."""
        H, T, P = bd.shape
        bd = np.pad(bd, ((0, 0), (0, 0), (1, 0)))                      # [H, T, P+1]
        bd = bd.reshape(H, -1)[:, T:].reshape(H, T, P)                 # drop prima "riga"
        return bd

    def _attn_mask(self, T: int, left: int, right: int) -> np.ndarray:
        """Mask booleana chunked_limited [T, T] (True = visibile)."""
        chunk = right + 1
        lc = left // chunk
        q = np.arange(T)[:, None] // chunk
        k = np.arange(T)[None, :] // chunk
        d = q - k
        return (d >= 0) & (d <= lc)

    def attention(self, x, li: int, mask: np.ndarray, pos: np.ndarray):
        w = self.w
        p = f"encoder.layers.{li}.self_attn."
        T = x.shape[0]
        H, dk = self.n_heads, self.d_head

        def proj(name):
            return (x @ w[p + name + ".weight"].T.astype(np.float64)).reshape(T, H, dk).transpose(1, 0, 2)

        q, k, v = proj("q_proj"), proj("k_proj"), proj("v_proj")
        scaling = 1.0 / np.sqrt(dk)

        q_u = q + w[p + "bias_u"][:, None, :]
        q_v = q + w[p + "bias_v"][:, None, :]

        rk = (pos @ w[p + "relative_k_proj.weight"].T.astype(np.float64)).reshape(-1, H, dk).transpose(1, 0, 2)
        bd = q_v @ rk.transpose(0, 2, 1)                                # [H, T, 2T-1]
        bd = self._rel_shift(bd)[:, :, :T] * scaling
        if mask is not None:
            bd = np.where(mask[None, :, :], bd, -np.inf)

        ac = (q_u @ k.transpose(0, 2, 1)) * scaling
        att = softmax(ac + bd, axis=-1)
        out = (att @ v).transpose(1, 0, 2).reshape(T, H * dk)
        return out @ w[p + "o_proj.weight"].T.astype(np.float64)

    def conv_module(self, x, li: int):
        w = self.w
        p = f"encoder.layers.{li}.conv."
        # pointwise_conv1 [2d, d, 1] -> GLU
        h = x @ w[p + "pointwise_conv1.weight"][:, :, 0].T.astype(np.float64)   # [T, 2d]
        a, b = h[:, :self.d_model], h[:, self.d_model:]
        h = a * sigmoid(b)
        # depthwise k: causale (pad left k-1) o 'same' simmetrico (pad (k-1)/2 per lato)
        k = self.conv_k
        pad = (k - 1, 0) if self.causal else ((k - 1) // 2, (k - 1) // 2)
        hp = np.pad(h, (pad, (0, 0)))
        dw = w[p + "depthwise_conv.weight"][:, 0, :].astype(np.float64)          # [d, k]
        h = sum(hp[i:i + h.shape[0]] * dw[:, i] for i in range(k))
        if self.conv_norm == "batch_norm":
            # BatchNorm1d in inference: affine per-canale con le running stats (eps 1e-5)
            inv = 1.0 / np.sqrt(w[p + "norm.running_var"].astype(np.float64) + 1e-5)
            h = (h - w[p + "norm.running_mean"]) * inv * w[p + "norm.weight"] + w[p + "norm.bias"]
        else:
            h = layer_norm(h, w[p + "norm.weight"], w[p + "norm.bias"])
        h = silu(h)
        return h @ w[p + "pointwise_conv2.weight"][:, :, 0].T.astype(np.float64)

    def ffn(self, x, li: int, which: str):
        w = self.w
        p = f"encoder.layers.{li}.{which}."
        h = silu(x @ w[p + "linear1.weight"].T.astype(np.float64))
        return h @ w[p + "linear2.weight"].T.astype(np.float64)

    def encoder_layer(self, x, li: int, mask, pos):
        w = self.w
        p = f"encoder.layers.{li}."
        ln = lambda name, t: layer_norm(t, w[p + name + ".weight"], w[p + name + ".bias"])
        x = x + 0.5 * self.ffn(ln("norm_feed_forward1", x), li, "feed_forward1")
        x = x + self.attention(ln("norm_self_att", x), li, mask, pos)
        x = x + self.conv_module(ln("norm_conv", x), li)
        x = x + 0.5 * self.ffn(ln("norm_feed_forward2", x), li, "feed_forward2")
        return ln("norm_out", x)

    def encode(self, feats: np.ndarray, prompt_id: int | None, lookahead: int | None = None,
               dumps: dict | None = None) -> np.ndarray:
        """feats [T_mel, n_mels] validi -> enc [T_enc, 640] (dopo prompt + projector)."""
        w = self.w
        x = self.subsampling(feats)
        if dumps is not None:
            dumps["subsampling"] = x.copy()
        T = x.shape[0]
        pos = self.rel_pos_emb(T)
        if "streaming" in self.cfg:
            left, right = self.cfg["streaming"]["att_context_presets"][
                self.cfg["streaming"]["default_preset_index"]]
            if lookahead is not None:
                right = lookahead
            mask = self._attn_mask(T, left, right)
        else:
            mask = None                                    # attention full [-1, -1]
        for li in range(self.n_layers):
            x = self.encoder_layer(x, li, mask, pos)
            if dumps is not None and li in (0, self.n_layers // 2, self.n_layers - 1):
                dumps[f"layer_{li}"] = x.copy()
        if dumps is not None:
            dumps["encoder_out"] = x.copy()
        if self.has_prompt:
            # prompt POST-encoder (Nemotron)
            one_hot = np.zeros((T, self.cfg["prompt"]["num_prompts"]))
            one_hot[:, prompt_id] = 1.0
            fused = np.concatenate([x, one_hot], axis=-1)
            fused = np.maximum(fused @ w["prompt_projector.linear_1.weight"].T.astype(np.float64)
                               + w["prompt_projector.linear_1.bias"], 0.0)
            x = fused @ w["prompt_projector.linear_2.weight"].T.astype(np.float64) \
                + w["prompt_projector.linear_2.bias"]
        enc = x @ w["encoder_projector.weight"].T.astype(np.float64) + w["encoder_projector.bias"]
        if dumps is not None:
            dumps["enc_proj"] = enc.copy()
        return enc

    # ---------------------------------------------------------------- decoder
    def _lstm_step(self, x, state):
        """x [640]; state = [(h0,c0),(h1,c1)]. Ritorna (out [640], new_state)."""
        w = self.w
        new_state = []
        inp = x
        for l in range(self.cfg["decoder"]["pred_layers"]):
            h, c = state[l]
            z = (w[f"decoder.lstm.weight_ih_l{l}"].astype(np.float64) @ inp
                 + w[f"decoder.lstm.bias_ih_l{l}"]
                 + w[f"decoder.lstm.weight_hh_l{l}"].astype(np.float64) @ h
                 + w[f"decoder.lstm.bias_hh_l{l}"])
            H = self.cfg["decoder"]["pred_hidden"]
            i, f, g, o = sigmoid(z[:H]), sigmoid(z[H:2 * H]), np.tanh(z[2 * H:3 * H]), sigmoid(z[3 * H:])
            c = f * c + i * g
            h = o * np.tanh(c)
            new_state.append((h, c))
            inp = h
        return inp, new_state

    def _pred_step(self, token: int, state):
        emb = self.w["decoder.embedding.weight"][token].astype(np.float64)
        out, new_state = self._lstm_step(emb, state)
        g = self.w["decoder.decoder_projector.weight"].astype(np.float64) @ out \
            + self.w["decoder.decoder_projector.bias"]
        return g, new_state

    def greedy_decode(self, enc: np.ndarray) -> list[int]:
        """Greedy RNNT su enc [T, 640]. SOS = blank; stato LSTM avanza solo su non-blank."""
        if self.durations:
            return self.greedy_decode_tdt(enc)
        H = self.cfg["decoder"]["pred_hidden"]
        zeros = lambda: [(np.zeros(H), np.zeros(H)) for _ in range(self.cfg["decoder"]["pred_layers"])]
        head_w = self.w["joint.head.weight"].astype(np.float64)
        head_b = self.w["joint.head.bias"].astype(np.float64)

        g, state = self._pred_step(self.blank, zeros())   # primo step: input blank, stato zero
        tokens: list[int] = []
        for t in range(enc.shape[0]):
            emitted = 0
            while emitted < self.max_symbols:
                logits = head_w @ np.maximum(enc[t] + g, 0.0) + head_b
                k = int(np.argmax(logits))
                if k == self.blank:
                    break
                tokens.append(k)
                g, state = self._pred_step(k, state)
                emitted += 1
        return tokens

    def greedy_decode_tdt(self, enc: np.ndarray) -> list[int]:
        """Greedy TDT su enc [T, 640] (ParakeetTDTGenerationMixin): ogni step il joint
        predice token (argmax sui primi V logit, blank incluso) E duration (argmax sugli
        ultimi len(durations)); il puntatore frame avanza della duration a OGNI step
        (blank con duration 0 -> forzata a 1); lo stato LSTM avanza solo su non-blank.
        Guardia max_symbols per-frame come NeMo (forza avanzamento su loop dur=0)."""
        H = self.cfg["decoder"]["pred_hidden"]
        V = self.cfg["decoder"]["vocab_size"]              # 8193, blank incluso
        zeros = lambda: [(np.zeros(H), np.zeros(H)) for _ in range(self.cfg["decoder"]["pred_layers"])]
        head_w = self.w["joint.head.weight"].astype(np.float64)
        head_b = self.w["joint.head.bias"].astype(np.float64)

        g, state = self._pred_step(self.blank, zeros())
        tokens: list[int] = []
        t, emitted_here = 0, 0
        while t < enc.shape[0]:
            logits = head_w @ np.maximum(enc[t] + g, 0.0) + head_b
            k = int(np.argmax(logits[:V]))
            dur = self.durations[int(np.argmax(logits[V:]))]
            if k != self.blank:
                tokens.append(k)
                g, state = self._pred_step(k, state)
                emitted_here += 1
                if dur == 0 and emitted_here >= self.max_symbols:
                    dur = 1                                # sblocca il frame (NeMo)
            elif dur == 0:
                dur = 1                                    # blank: avanzamento minimo 1
            if dur > 0:
                emitted_here = 0
            t += dur
        return tokens

    # ------------------------------------------------------------------ testo
    def detokenize(self, tokens: list[int]) -> tuple[str, str | None]:
        """Testo + tag lingua (None se il modello non li emette, es. Parakeet)."""
        lang = None
        parts = []
        for t in tokens:
            piece = self.pieces[t]
            if piece.startswith("<") and piece.endswith(">"):
                if "-" in piece:
                    lang = piece[1:-1]
                continue
            parts.append(piece)
        text = "".join(parts).replace("▁", " ").lstrip(" ")
        return text, lang
