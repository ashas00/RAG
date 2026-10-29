"""Retrieval: load the BM25 index and return the top-k sources."""
import os

import bm25s
from pydantic import TypeAdapter

from src.indexer import BM25_DIR, CHUNKS_FILE, tokenize
from src.models import MinimalSource

_SOURCES = TypeAdapter(list[MinimalSource])

# Tests and benchmarks repeat the words of the code they exercise, so they
# often outrank the real implementation. Questions are about vLLM itself:
# such chunks keep only part of their score.
LOW_PRIORITY_DIRS = ("/tests/", "/benchmarks/")
LOW_PRIORITY_WEIGHT = 0.6
# Re-rank this many BM25 candidates (at least) after weighting.
CANDIDATES = 50


def path_weight(file_path: str) -> float:
    """Score multiplier of a chunk, based on where its file lives."""
    if any(d in file_path for d in LOW_PRIORITY_DIRS):
        return LOW_PRIORITY_WEIGHT
    return 1.0


class Retriever:
    """Searches a BM25 index built by `build_index`."""

    def __init__(self, index_dir: str) -> None:
        """Load the index; raises if it is missing or corrupted."""
        self.bm25 = bm25s.BM25.load(os.path.join(index_dir, BM25_DIR))
        with open(os.path.join(index_dir, CHUNKS_FILE),
                  encoding="utf-8") as f:
            self.chunks = _SOURCES.validate_json(f.read())

    def search(self, query: str, k: int) -> list[MinimalSource]:
        """Return the k chunks with the best BM25 score, best first."""
        tokens = tokenize(query)
        pool = min(max(k, CANDIDATES), len(self.chunks))
        if not tokens or k <= 0 or pool <= 0:
            return []
        ids, scores = self.bm25.retrieve([tokens], k=pool,
                                         show_progress=False)
        ranked = sorted(
            ((float(score) * path_weight(self.chunks[int(i)].file_path),
              int(i))
             for i, score in zip(ids[0], scores[0])
             # A score of 0 means no query word is in the chunk.
             if score > 0),
            reverse=True,
        )
        return [self.chunks[i] for _, i in ranked[:k]]
