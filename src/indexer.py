"""Build and load the BM25 index."""
import json
import os
import re

import bm25s
from tqdm import tqdm

from src.chunker import Chunk, chunk_file

EXTENSIONS = (".py", ".md")

STOPWORDS = frozenset(
    "a an the is are was were be been of to in on for and or with by at "
    "as from that this it its what which how does do can when where why "
    "who into than then there these those used use using vllm".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase words, also splitting snake_case and CamelCase.

    "load_lora_adapter" -> load_lora_adapter, load, lora, adapter
    "LoRAConfig" -> loraconfig, lo, ra, config (good enough)
    """
    tokens: list[str] = []
    for word in re.findall(r"[A-Za-z0-9_]+", text):
        tokens.append(word.lower())
        parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", word)
        if len(parts) > 1:
            tokens.extend(p.lower() for p in parts)
    return [t for t in tokens if t not in STOPWORDS]


def collect_files(raw_dir: str) -> list[str]:
    """Return every indexable file path under raw_dir, sorted."""
    paths: list[str] = []
    for root, _, files in os.walk(raw_dir):
        for name in files:
            if name.endswith(EXTENSIONS):
                paths.append(os.path.join(root, name))
    return sorted(paths)


def build_index(raw_dir: str, out_dir: str, max_chunk_size: int) -> int:
    """Chunk all files, build BM25 and save to out_dir.

    Returns:
        Number of chunks indexed.
    """
    chunks: list[Chunk] = []
    for path in tqdm(collect_files(raw_dir), desc="Chunking"):
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except (OSError, UnicodeDecodeError):
            continue
        chunks.extend(chunk_file(path, text, max_chunk_size))

    # Path words help a lot: "lora" in docs/features/lora.md.
    corpus = [
        tokenize(c.file_path) + tokenize(c.text)
        for c in tqdm(chunks, desc="Tokenizing")
    ]
    retriever = bm25s.BM25()
    retriever.index(corpus, show_progress=False)

    os.makedirs(out_dir, exist_ok=True)
    retriever.save(os.path.join(out_dir, "bm25"))
    with open(os.path.join(out_dir, "chunks.json"), "w") as f:
        json.dump([c.model_dump(exclude={"text"}) for c in chunks], f)
    return len(chunks)


def load_index(out_dir: str) -> tuple[bm25s.BM25, list[dict[str, object]]]:
    """Load the BM25 model and chunk metadata from out_dir."""
    retriever = bm25s.BM25.load(os.path.join(out_dir, "bm25"))
    with open(os.path.join(out_dir, "chunks.json")) as f:
        chunks: list[dict[str, object]] = json.load(f)
    return retriever, chunks


def search(
    retriever: bm25s.BM25,
    chunks: list[dict[str, object]],
    query: str,
    k: int,
) -> list[dict[str, object]]:
    """Return metadata of the top-k chunks for a query."""
    tokens = tokenize(query)
    if not tokens or k <= 0:
        return []
    k = min(k, len(chunks))
    ids, _ = retriever.retrieve([tokens], k=k, show_progress=False)
    return [chunks[int(i)] for i in ids[0]]
