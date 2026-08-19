from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import AggregationPolicy, Annotation, PromptGroup, TaskType


@dataclass(slots=True)
class Candidate:
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    prompt: str
    polygons: list[list[float]] = field(default_factory=list)
    matched_prompts: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.matched_prompts:
            self.matched_prompts = [self.prompt]


def box_iou(left: Candidate, right: Candidate) -> float:
    lx1, ly1, lx2, ly2 = left.bbox_xyxy
    rx1, ry1, rx2, ry2 = right.bbox_xyxy
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def deduplicate(candidates: list[Candidate], threshold: float) -> list[Candidate]:
    """Drop cross-prompt duplicates while preserving same-prompt instances."""
    kept: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        duplicate_of = next(
            (
                previous
                for previous in kept
                if candidate.prompt != previous.prompt and box_iou(candidate, previous) >= threshold
            ),
            None,
        )
        if duplicate_of is None:
            kept.append(candidate)
        elif candidate.prompt not in duplicate_of.matched_prompts:
            duplicate_of.matched_prompts.append(candidate.prompt)
    return kept


def _to_xywh(box: tuple[float, float, float, float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [round(x1, 3), round(y1, 3), round(max(0.0, x2 - x1), 3), round(max(0.0, y2 - y1), 3)]


def _annotation(
    candidate: Candidate, group: PromptGroup, task_type: TaskType, annotation_id: int
) -> Annotation:
    return Annotation(
        id=annotation_id,
        label=group.label,
        class_id=group.class_id,
        score=round(candidate.score, 6),
        bbox=_to_xywh(candidate.bbox_xyxy),
        segmentation=candidate.polygons if task_type != TaskType.DETECT else None,
        source_prompts=candidate.matched_prompts,
    )


def aggregate(
    candidates: list[Candidate], group: PromptGroup, task_type: TaskType, start_id: int = 1
) -> list[Annotation]:
    if not candidates:
        return []

    if group.aggregation == AggregationPolicy.KEEP_ALL:
        selected = sorted(candidates, key=lambda item: item.score, reverse=True)
    elif group.aggregation == AggregationPolicy.BEST:
        selected = [max(candidates, key=lambda item: item.score)]
    else:
        selected = deduplicate(candidates, group.merge_iou)

    if group.aggregation != AggregationPolicy.UNION:
        return [
            _annotation(candidate, group, task_type, start_id + offset)
            for offset, candidate in enumerate(selected)
        ]

    x1 = min(candidate.bbox_xyxy[0] for candidate in selected)
    y1 = min(candidate.bbox_xyxy[1] for candidate in selected)
    x2 = max(candidate.bbox_xyxy[2] for candidate in selected)
    y2 = max(candidate.bbox_xyxy[3] for candidate in selected)
    polygons = [polygon for candidate in selected for polygon in candidate.polygons]
    return [
        Annotation(
            id=start_id,
            label=group.label,
            class_id=group.class_id,
            score=round(max(candidate.score for candidate in selected), 6),
            bbox=_to_xywh((x1, y1, x2, y2)),
            segmentation=polygons,
            source_prompts=sorted(
                {prompt for candidate in selected for prompt in candidate.matched_prompts}
            ),
            instance_count=len(selected),
        )
    ]
