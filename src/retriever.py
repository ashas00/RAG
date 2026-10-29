"""Retrieval: load the BM25 index and return the top-k sources."""
import os

import bm25s
from pydantic import TypeAdapter

from src.indexer import BM25_DIR, CHUNKS_FILE, tokenize
from src.models import MinimalSource

_SOURCES = TypeAdapter(list[MinimalSource])


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
        k = min(k, len(self.chunks))
        if not tokens or k <= 0:
            return []
        ids, scores = self.bm25.retrieve([tokens], k=k, show_progress=False)
        # A score of 0 means no query word is in the chunk: not a match.
        return [self.chunks[int(i)]
                for i, score in zip(ids[0], scores[0]) if score > 0]
