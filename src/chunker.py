"""Split files into chunks: one strategy for Python, one for Markdown."""
import ast
import re

from pydantic import BaseModel

from src.models import MinimalSource


class Chunk(BaseModel):
    """A piece of a corpus file: its character span and its text."""

    file_path: str
    start: int
    end: int
    text: str

    def to_source(self) -> MinimalSource:
        """Return the span as a MinimalSource (what retrieval outputs)."""
        return MinimalSource(file_path=self.file_path,
                             first_character_index=self.start,
                             last_character_index=self.end)


def _line_offsets(text: str) -> list[int]:
    """Return the character offset where each line starts."""
    return [0] + [m.end() for m in re.finditer("\n", text)]


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
                if not line.strip()]
    nodes: list[ast.stmt] = []
    for node in tree.body:
        nodes.append(node)
        if isinstance(node, ast.ClassDef):
            nodes.extend(node.body)
    return [offsets[_first_line(node) - 1] for node in nodes]


def _first_line(node: ast.stmt) -> int:
    """First line of a statement, decorators included."""
    decorators: list[ast.expr] = getattr(node, "decorator_list", [])
    return min([node.lineno] + [d.lineno for d in decorators])


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
