"""Tests for the NumPy Qwen3 runtime on a tiny random model."""
import json
from pathlib import Path

import numpy as np
import pytest

from src import qwen

from src.qwen import Qwen3, load_safetensors

HIDDEN, LAYERS, HEADS, KV_HEADS, HEAD_DIM, FFN, VOCAB = 32, 2, 4, 2, 8, 64, 50
CONFIG = {
    "hidden_size": HIDDEN, "num_hidden_layers": LAYERS,
    "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS,
    "head_dim": HEAD_DIM, "intermediate_size": FFN, "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0, "vocab_size": VOCAB, "tie_word_embeddings": True,
}


def save_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Write float32 tensors as bf16 (the format of the real weights)."""
    header: dict[str, object] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, value in tensors.items():
        bits = value.astype(np.float32).view(np.uint32) >> 16
        blob = bits.astype(np.uint16).tobytes()
        header[name] = {"dtype": "BF16", "shape": list(value.shape),
                        "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    head = json.dumps(header).encode()
    path.write_bytes(len(head).to_bytes(8, "little") + head + b"".join(blobs))


def make_model(tmp_path: Path) -> Qwen3:
    rng = np.random.default_rng(0)
    h, d, f = HIDDEN, HEAD_DIM, FFN
    q, kv = HEADS * d, KV_HEADS * d
    shapes: dict[str, tuple[int, ...]] = {
        "model.embed_tokens.weight": (VOCAB, h), "model.norm.weight": (h,)}
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        shapes.update({
            p + "input_layernorm.weight": (h,),
            p + "post_attention_layernorm.weight": (h,),
            p + "self_attn.q_proj.weight": (q, h),
            p + "self_attn.k_proj.weight": (kv, h),
            p + "self_attn.v_proj.weight": (kv, h),
            p + "self_attn.o_proj.weight": (h, q),
            p + "self_attn.q_norm.weight": (d,),
            p + "self_attn.k_norm.weight": (d,),
            p + "mlp.gate_proj.weight": (f, h),
            p + "mlp.up_proj.weight": (f, h),
            p + "mlp.down_proj.weight": (h, f),
        })
    tensors = {n: rng.normal(0, 0.5, s).astype(np.float32)
               for n, s in shapes.items()}
    save_safetensors(tmp_path / "model.safetensors", tensors)
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    return Qwen3(str(tmp_path))


def test_bf16_round_trip(tmp_path: Path) -> None:
    value = np.array([[1.0, -2.5], [0.15625, 3.0]], np.float32)
    save_safetensors(tmp_path / "t.safetensors", {"w": value})
    np.testing.assert_array_equal(
        load_safetensors(str(tmp_path / "t.safetensors"))["w"], value)


def test_kv_cache_matches_full_forward(tmp_path: Path) -> None:
    """Feeding tokens one by one with the cache must give the same logits
    as running the whole sequence at once."""
    model = make_model(tmp_path)
    tokens = [3, 7, 1, 42, 9, 0]
    c = model.config

    def fresh() -> list[np.ndarray]:
        shape = (c.num_kv_heads, len(tokens), c.head_dim)
        return [np.zeros(shape, np.float32) for _ in model.layers]

    full = model.forward(tokens, 0, fresh(), fresh())
    k_cache, v_cache = fresh(), fresh()
    model.forward(tokens[:3], 0, k_cache, v_cache)
    for pos in range(3, len(tokens)):
        step = model.forward([tokens[pos]], pos, k_cache, v_cache)
    np.testing.assert_allclose(step, full, rtol=1e-4, atol=1e-4)


def test_generate_stops(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    out = model.generate([1, 2, 3], max_new_tokens=5, stop_ids=set())
    assert len(out) == 5
    first = out[0]
    assert model.generate([1, 2, 3], 5, stop_ids={first}) == []


def test_block_prefill_matches_single_pass(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Splitting the prompt into blocks must not change the output."""
    model = make_model(tmp_path)
    prompt = [5, 8, 13, 21, 34, 2, 4, 6, 8, 10, 12]
    whole = model.generate(prompt, 8, set(), repetition_penalty=1.1)
    monkeypatch.setattr(qwen, "PREFILL_BLOCK", 3)
    assert model.generate(prompt, 8, set(), repetition_penalty=1.1) == whole
