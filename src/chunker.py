"""Split files into chunks: one strategy for Python, one for Markdown."""
import ast
import re

from pydantic import BaseModel


class Chunk(BaseModel):
    """A piece of a corpus file with its character span."""

    file_path: str
    start: int
    end: int
    text: str


def _line_offsets(text: str) -> list[int]:
    """Return the character offset where each line starts."""
    offsets = [0]
    for i, char in enumerate(text):
        if char == "\n":
            offsets.append(i + 1)
    return offsets


def _split_by_boundaries(
    text: str, boundaries: list[int], max_size: int
) -> list[tuple[int, int]]:
    """Group text into spans <= max_size, cutting only at boundaries.

    Args:
        text: Full file content.
        boundaries: Character offsets where a cut is allowed.
        max_size: Maximum span length in characters.

    Returns:
        List of (start, end) spans covering the whole text.
    """
    points = sorted({0, len(text), *boundaries})
    # Segments between consecutive boundaries; split too-long ones.
    segments: list[tuple[int, int]] = []
    for a, b in zip(points, points[1:]):
        while b - a > max_size:
            # Too long: cut at the last newline that fits, else hard cut.
            cut = text.rfind("\n", a, a + max_size) + 1
            if cut <= a:
                cut = a + max_size
            segments.append((a, cut))
            a = cut
        if b > a:
            segments.append((a, b))
    # Greedily merge neighbouring segments while they fit.
    spans: list[tuple[int, int]] = []
    for a, b in segments:
        if spans and b - spans[-1][0] <= max_size:
            spans[-1] = (spans[-1][0], b)
        else:
            spans.append((a, b))
    return spans


def python_boundaries(text: str) -> list[int]:
    """Cut points for Python: starts of top-level statements and methods.

    Top-level defs/classes and methods inside classes are natural cut
    points. If the file does not parse, fall back to blank lines.
    """
    offsets = _line_offsets(text)
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return [offsets[i] for i, line in enumerate(text.split("\n"))
                if not line.strip() and i < len(offsets)]
    cuts: list[int] = []
    for node in tree.body:
        start = node.lineno
        # Include decorators in the chunk of the function/class.
        decorators = getattr(node, "decorator_list", [])
        if decorators:
            start = min(d.lineno for d in decorators)
        cuts.append(offsets[start - 1])
        if isinstance(node, ast.ClassDef):
            for sub in node.body:
                sub_start = sub.lineno
                sub_decs = getattr(sub, "decorator_list", [])
                if sub_decs:
                    sub_start = min(d.lineno for d in sub_decs)
                cuts.append(offsets[sub_start - 1])
    return cuts


HEADING = re.compile(r"^#{1,6} ", re.MULTILINE)


def markdown_boundaries(text: str) -> list[int]:
    """Cut points for Markdown: headings first, then paragraphs."""
    cuts = [m.start() for m in HEADING.finditer(text)]
    cuts += [m.end() for m in re.finditer(r"\n\s*\n", text)]
    return cuts


def chunk_file(file_path: str, text: str, max_size: int) -> list[Chunk]:
    """Chunk one file with the strategy matching its extension."""
    if file_path.endswith(".py"):
        boundaries = python_boundaries(text)
    else:
        boundaries = markdown_boundaries(text)
    return [
        Chunk(file_path=file_path, start=a, end=b, text=text[a:b])
        for a, b in _split_by_boundaries(text, boundaries, max_size)
        if text[a:b].strip()
    ]
