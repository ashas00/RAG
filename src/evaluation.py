"""Recall@k computation, same rule as the moulinette."""
from src.models import MinimalSource

MIN_IOU = 0.05


def iou(a: MinimalSource, b: MinimalSource) -> float:
    """Intersection over union of two spans (0 if different files)."""
    if a.file_path != b.file_path:
        return 0.0
    inter = (min(a.last_character_index, b.last_character_index)
             - max(a.first_character_index, b.first_character_index))
    if inter <= 0:
        return 0.0
    union = (max(a.last_character_index, b.last_character_index)
             - min(a.first_character_index, b.first_character_index))
    return inter / union


def recall_at_k(retrieved: list[MinimalSource],
                gold: list[MinimalSource], k: int) -> float:
    """Share of gold sources matched by at least one of the top-k."""
    if not gold:
        return 0.0
    top = retrieved[:k]
    hits = sum(
        1 for g in gold if any(iou(r, g) >= MIN_IOU for r in top)
    )
    return hits / len(gold)
