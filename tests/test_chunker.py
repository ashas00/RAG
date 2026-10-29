"""Tests for the two chunking strategies."""
from src.chunker import chunk_file, markdown_boundaries, python_boundaries

PY = '''import os


def first():
    return 1


@decorator
class Thing:
    x = 1

    def method(self):
        return 2
'''

MD = "# Title\n\nIntro text.\n\n## Part\n\nMore text.\n"


def _covers(text: str, max_size: int, path: str) -> None:
    chunks = chunk_file(path, text, max_size)
    assert chunks, "at least one chunk"
    for c in chunks:
        assert 0 < c.end - c.start <= max_size
        assert text[c.start:c.end] == c.text
    # Chunks are in order and do not overlap.
    for a, b in zip(chunks, chunks[1:]):
        assert a.end <= b.start


def test_python_cuts_at_defs_and_decorators() -> None:
    cuts = python_boundaries(PY)
    assert PY.index("def first") in cuts
    assert PY.index("@decorator") in cuts  # decorator stays with class
    assert PY.index("    def method") in cuts


def test_python_invalid_syntax_falls_back() -> None:
    assert 7 in python_boundaries("def (:\n\nx = 1\n")  # blank line


def test_markdown_cuts_at_headings() -> None:
    cuts = markdown_boundaries(MD)
    assert MD.index("## Part") in cuts


def test_chunks_respect_max_size() -> None:
    _covers(PY * 20, 200, "a.py")
    _covers(MD * 50, 120, "a.md")
    _covers("x" * 5000, 2000, "long.md")  # no newline: hard cut


def test_empty_file_gives_no_chunk() -> None:
    assert chunk_file("a.py", "", 2000) == []
    assert chunk_file("a.md", "\n\n  \n", 2000) == []
