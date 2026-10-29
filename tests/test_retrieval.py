"""Tests for tokenizing, indexing, retrieval and recall."""
from pathlib import Path

from src.evaluation import iou, recall_at_k
from src.indexer import build_index, tokenize
from src.models import MinimalSource
from src.retriever import Retriever, path_weight


def src(path: str, a: int, b: int) -> MinimalSource:
    return MinimalSource(file_path=path, first_character_index=a,
                         last_character_index=b)


def test_tokenize_splits_identifiers() -> None:
    tokens = tokenize("load_lora_adapter LoRAConfig")
    assert "load_lora_adapter" in tokens
    assert {"load", "lora", "adapter", "config"} <= set(tokens)
    assert tokenize("what is the") == []  # only stopwords


def test_path_weight() -> None:
    assert path_weight("data/raw/v/tests/test_x.py") < 1.0
    assert path_weight("data/raw/v/vllm/x.py") == 1.0


def test_build_and_search(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "docs").mkdir(parents=True)
    (raw / "docs" / "lora.md").write_text("# LoRA\n\nEnable LoRA adapters.\n")
    (raw / "sched.py").write_text("def schedule_requests():\n    pass\n")
    (raw / "image.png").write_bytes(b"\x89PNG")  # ignored extension
    index = tmp_path / "index"
    assert build_index(str(raw), str(index), 2000) == 2

    retriever = Retriever(str(index))
    found = retriever.search("How do I enable lora?", 5)
    assert found[0].file_path.endswith("lora.md")
    assert retriever.search("schedule requests", 1)[0].file_path.endswith(
        "sched.py")
    assert retriever.search("zzzz unknown", 5) == []
    assert retriever.search("", 5) == []
    assert retriever.search("lora", 0) == []


def test_build_empty_corpus(tmp_path: Path) -> None:
    assert build_index(str(tmp_path), str(tmp_path / "i"), 2000) == 0


def test_iou_and_recall() -> None:
    gold = [src("a", 0, 100), src("b", 0, 100)]
    assert iou(src("a", 0, 100), src("a", 50, 150)) == 50 / 150
    assert iou(src("a", 0, 100), src("b", 0, 100)) == 0.0
    assert recall_at_k([src("a", 90, 2000)], gold, 5) == 0.0  # iou < 0.05
    assert recall_at_k([src("a", 0, 500), src("b", 0, 10)], gold, 5) == 1.0
    assert recall_at_k([src("x", 0, 1), src("a", 0, 100)], gold, 1) == 0.0
    assert recall_at_k([], [], 5) == 0.0
