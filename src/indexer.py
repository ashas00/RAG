"""Indexing: read the corpus, chunk it and persist a BM25 index."""
import json
import os
import re

import bm25s
from tqdm import tqdm

from src.chunker import Chunk, chunk_file

# .txt: CMakeLists.txt and requirements files answer some docs questions.
EXTENSIONS = (".py", ".md", ".txt")
BM25_DIR = "bm25"
CHUNKS_FILE = "chunks.json"

STOPWORDS = frozenset(
    "a an the is are was were be been of to in on for and or with by at "
    "as from that this it its what which how does do can when where why "
    "who into than then there these those used use using vllm".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase words, also splitting snake_case and CamelCase.

    "load_lora_adapter" -> load_lora_adapter, load, lora, adapter
    "LoRAConfig"        -> loraconfig, lo, ra, config
    Keeping the full identifier lets a question that quotes it verbatim
    match exactly; the parts let a paraphrased question match too.
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


def build_index(raw_dir: str, index_dir: str, max_chunk_size: int) -> int:
    """Chunk all files of raw_dir, build BM25 and save it to index_dir.

    Returns:
        Number of chunks indexed (0 if there was nothing to index).
    """
    chunks: list[Chunk] = []
    for path in tqdm(collect_files(raw_dir), desc="Chunking"):
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except (OSError, UnicodeDecodeError):
            continue  # unreadable or binary file: skip it
        chunks.extend(chunk_file(path, text, max_chunk_size))
    if not chunks:
        return 0

    # Path words help a lot: "lora" in docs/features/lora.md.
    corpus = [
        tokenize(c.file_path) + tokenize(c.text)
        for c in tqdm(chunks, desc="Tokenizing")
    ]
    bm25 = bm25s.BM25()
    bm25.index(corpus, show_progress=False)

    os.makedirs(index_dir, exist_ok=True)
    bm25.save(os.path.join(index_dir, BM25_DIR))
    # Text is not stored: the span is enough to read it back from disk.
    with open(os.path.join(index_dir, CHUNKS_FILE), "w",
              encoding="utf-8") as f:
        json.dump([c.to_source().model_dump() for c in chunks], f)
    return len(chunks)
