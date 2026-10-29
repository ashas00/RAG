"""Minimal Qwen3 inference in pure NumPy (no PyTorch needed).

Only what is needed to run Qwen/Qwen3-0.6B on CPU: read the official
safetensors weights, run the transformer forward pass and decode tokens
greedily with a key/value cache.

One decoder layer does:
    h = x + Attention(RMSNorm(x))      # grouped-query attention + RoPE
    y = h + MLP(RMSNorm(h))            # SwiGLU: down(silu(gate) * up)
"""
import json
import os
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float32]

# Prompt tokens processed per forward pass.
PREFILL_BLOCK = 256


@dataclass
class Qwen3Config:
    """The few hyper-parameters the forward pass needs (from config.json)."""

    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool

    @classmethod
    def from_file(cls, path: str) -> "Qwen3Config":
        """Read a Hugging Face config.json."""
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        # Newer configs store rope_theta inside "rope_parameters".
        rope = cfg.get("rope_parameters") or {}
        theta = cfg.get("rope_theta", rope.get("rope_theta", 1_000_000.0))
        hidden = int(cfg["hidden_size"])
        heads = int(cfg["num_attention_heads"])
        return cls(
            hidden_size=hidden,
            num_layers=int(cfg["num_hidden_layers"]),
            num_heads=heads,
            num_kv_heads=int(cfg["num_key_value_heads"]),
            head_dim=int(cfg.get("head_dim") or hidden // heads),
            rms_norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
            rope_theta=float(theta),
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
        )


def load_safetensors(path: str) -> dict[str, Array]:
    """Load every tensor of a .safetensors file as float32.

    File format: 8 bytes (header size N, little endian), N bytes of JSON
    describing each tensor (dtype, shape, byte offsets), then raw data.
    """
    with open(path, "rb") as f:
        header_size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_size))
    raw = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + header_size)
    tensors: dict[str, Array] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        begin, end = info["data_offsets"]
        chunk = raw[begin:end]
        dtype = info["dtype"]
        if dtype == "BF16":
            # bfloat16 is the upper half of a float32: shift it back.
            bits = chunk.view(np.uint16).astype(np.uint32) << 16
            values = bits.view(np.float32)
        elif dtype == "F16":
            values = chunk.view(np.float16).astype(np.float32)
        elif dtype == "F32":
            values = np.array(chunk.view(np.float32))
        else:
            raise ValueError(f"unsupported dtype {dtype} for {name}")
        tensors[name] = values.reshape(info["shape"])
    return tensors


def rms_norm(x: Array, weight: Array, eps: float) -> Array:
    """Scale x so its root-mean-square is 1, then multiply by weight."""
    variance = np.mean(x * x, axis=-1, keepdims=True)
    return (x / np.sqrt(variance + eps) * weight).astype(np.float32)


def silu(x: Array) -> Array:
    """SiLU activation: x * sigmoid(x)."""
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def softmax(x: Array) -> Array:
    """Numerically stable softmax over the last axis."""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return (e / np.sum(e, axis=-1, keepdims=True)).astype(np.float32)


@dataclass
class Layer:
    """Weights of one decoder layer (linear weights stored as [in, out])."""

    input_norm: Array
    q_proj: Array
    k_proj: Array
    v_proj: Array
    o_proj: Array
    q_norm: Array
    k_norm: Array
    post_norm: Array
    gate_proj: Array
    up_proj: Array
    down_proj: Array


class Qwen3:
    """A Qwen3 causal language model that runs with NumPy only."""

    def __init__(self, model_dir: str) -> None:
        """Load config.json and *.safetensors weights from model_dir."""
        self.config = Qwen3Config.from_file(
            os.path.join(model_dir, "config.json")
        )
        weights: dict[str, Array] = {}
        for name in sorted(os.listdir(model_dir)):
            if name.endswith(".safetensors"):
                weights.update(load_safetensors(os.path.join(model_dir, name)))

        def linear(name: str) -> Array:
            # Stored as [out, in]; transpose once so we can do x @ W.
            return np.ascontiguousarray(weights.pop(name).T)

        self.embed = np.ascontiguousarray(weights["model.embed_tokens.weight"])
        if "lm_head.weight" in weights and not self.config.tie_word_embeddings:
            self.lm_head = linear("lm_head.weight")
        else:
            self.lm_head = self.embed.T  # shared weights: a view, no copy
        self.norm = weights["model.norm.weight"]
        self.layers: list[Layer] = []
        for i in range(self.config.num_layers):
            p = f"model.layers.{i}."
            self.layers.append(Layer(
                input_norm=weights[p + "input_layernorm.weight"],
                q_proj=linear(p + "self_attn.q_proj.weight"),
                k_proj=linear(p + "self_attn.k_proj.weight"),
                v_proj=linear(p + "self_attn.v_proj.weight"),
                o_proj=linear(p + "self_attn.o_proj.weight"),
                q_norm=weights[p + "self_attn.q_norm.weight"],
                k_norm=weights[p + "self_attn.k_norm.weight"],
                post_norm=weights[p + "post_attention_layernorm.weight"],
                gate_proj=linear(p + "mlp.gate_proj.weight"),
                up_proj=linear(p + "mlp.up_proj.weight"),
                down_proj=linear(p + "mlp.down_proj.weight"),
            ))
        d = self.config.head_dim
        self.inv_freq = 1.0 / (
            self.config.rope_theta ** (np.arange(0, d, 2) / d)
        )

    def _rope(self, x: Array, start: int) -> Array:
        """Rotary position embedding for x of shape [heads, tokens, dim]."""
        positions = np.arange(start, start + x.shape[1])
        angles = np.outer(positions, self.inv_freq)
        angles = np.concatenate([angles, angles], axis=-1)
        cos = np.cos(angles).astype(np.float32)
        sin = np.sin(angles).astype(np.float32)
        half = x.shape[-1] // 2
        rotated = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
        return (x * cos + rotated * sin).astype(np.float32)

    def _attention(self, layer: Layer, x: Array, start: int,
                   k_cache: Array, v_cache: Array) -> Array:
        """Causal grouped-query attention; fills the cache in place."""
        c = self.config
        n = x.shape[0]
        q = (x @ layer.q_proj).reshape(n, c.num_heads, c.head_dim)
        k = (x @ layer.k_proj).reshape(n, c.num_kv_heads, c.head_dim)
        v = (x @ layer.v_proj).reshape(n, c.num_kv_heads, c.head_dim)
        # Qwen3 normalises queries and keys per head before RoPE.
        q = rms_norm(q, layer.q_norm, c.rms_norm_eps).transpose(1, 0, 2)
        k = rms_norm(k, layer.k_norm, c.rms_norm_eps).transpose(1, 0, 2)
        q = self._rope(q, start)
        k_cache[:, start:start + n] = self._rope(k, start)
        v_cache[:, start:start + n] = v.transpose(1, 0, 2)
        keys = k_cache[:, :start + n]
        values = v_cache[:, :start + n]

        # Each key/value head is shared by `group` query heads.
        group = c.num_heads // c.num_kv_heads
        grouped = q.reshape(c.num_kv_heads, group, n, c.head_dim)
        scores = grouped @ keys[:, None].transpose(0, 1, 3, 2)
        scores *= np.float32(c.head_dim ** -0.5)
        # Token i (absolute position start+i) may not look at later tokens.
        if n > 1:
            key_pos = np.arange(start + n)[None, :]
            future = key_pos > (start + np.arange(n))[:, None]
            scores += np.where(future, -np.inf, 0.0).astype(np.float32)
        out = softmax(scores) @ values[:, None]
        merged = out.reshape(c.num_heads, n, c.head_dim).transpose(1, 0, 2)
        result: Array = (merged.reshape(n, -1) @ layer.o_proj).astype(
            np.float32)
        return result

    def forward(self, tokens: list[int], start: int,
                k_cache: list[Array], v_cache: list[Array]) -> Array:
        """Run tokens placed at positions start.. and return the logits
        of the last one (shape [vocab])."""
        eps = self.config.rms_norm_eps
        x: Array = self.embed[tokens]
        for i, layer in enumerate(self.layers):
            h = rms_norm(x, layer.input_norm, eps)
            x = x + self._attention(layer, h, start, k_cache[i], v_cache[i])
            h = rms_norm(x, layer.post_norm, eps)
            mlp = silu(h @ layer.gate_proj) * (h @ layer.up_proj)
            x = (x + mlp @ layer.down_proj).astype(np.float32)
        last = rms_norm(x[-1:], self.norm, eps)
        logits: Array = (last @ self.lm_head)[0]
        return logits

    def generate(self, prompt: list[int], max_new_tokens: int,
                 stop_ids: set[int],
                 repetition_penalty: float = 1.0) -> list[int]:
        """Greedy decoding: always pick the most likely next token.

        Args:
            prompt: Token ids of the prompt.
            max_new_tokens: Upper bound on generated tokens.
            stop_ids: Generation ends when one of these is produced.
            repetition_penalty: > 1 makes already-used tokens less likely
                (same rule as Hugging Face), which avoids loops.

        Returns:
            Generated token ids, without the stop token.
        """
        c = self.config
        total = len(prompt) + max_new_tokens
        shape = (c.num_kv_heads, total, c.head_dim)
        k_cache = [np.zeros(shape, np.float32) for _ in self.layers]
        v_cache = [np.zeros(shape, np.float32) for _ in self.layers]

        # The prompt goes through in blocks (bounded memory), then the
        # answer is produced one token at a time.
        for start in range(0, len(prompt), PREFILL_BLOCK):
            block = prompt[start:start + PREFILL_BLOCK]
            logits = self.forward(block, start, k_cache, v_cache)
        seen = set(prompt)
        output: list[int] = []
        for _ in range(max_new_tokens):
            if repetition_penalty != 1.0 and seen:
                ids = np.fromiter(seen, dtype=np.int64)
                picked = logits[ids]
                logits[ids] = np.where(picked > 0,
                                       picked / repetition_penalty,
                                       picked * repetition_penalty)
            token = int(np.argmax(logits))
            if token in stop_ids:
                break
            output.append(token)
            seen.add(token)
            position = len(prompt) + len(output) - 1
            logits = self.forward([token], position, k_cache, v_cache)
        return output
