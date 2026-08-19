from sam_api.postprocess import Candidate, aggregate
from sam_api.schemas import AggregationPolicy, PromptGroup, TaskType


def candidate(box, score, prompt):
    return Candidate(tuple(box), score, prompt, [[box[0], box[1], box[2], box[1], box[2], box[3]]])


def test_deduplicate_only_suppresses_cross_prompt_duplicates():
    group = PromptGroup(label="car", prompts=["car", "automobile"], merge_iou=0.5)
    candidates = [
        candidate((0, 0, 10, 10), 0.9, "car"),
        candidate((1, 1, 11, 11), 0.8, "automobile"),
        candidate((2, 2, 12, 12), 0.7, "car"),
    ]

    annotations = aggregate(candidates, group, TaskType.SEGMENT)

    assert len(annotations) == 2
    assert [annotation.score for annotation in annotations] == [0.9, 0.7]
    assert annotations[0].source_prompts == ["car", "automobile"]


def test_best_returns_one_annotation():
    group = PromptGroup(label="car", prompts=["car", "automobile"], aggregation=AggregationPolicy.BEST)
    annotations = aggregate(
        [candidate((0, 0, 10, 10), 0.6, "car"), candidate((20, 20, 30, 30), 0.95, "automobile")],
        group,
        TaskType.DETECT,
    )

    assert len(annotations) == 1
    assert annotations[0].score == 0.95
    assert annotations[0].segmentation is None


def test_union_combines_instances_for_semantic_output():
    group = PromptGroup(label="car", prompts=["car", "automobile"], aggregation=AggregationPolicy.UNION)
    annotations = aggregate(
        [candidate((0, 0, 10, 10), 0.9, "car"), candidate((20, 20, 30, 30), 0.8, "automobile")],
        group,
        TaskType.SEMANTIC,
    )

    assert len(annotations) == 1
    assert annotations[0].bbox == [0.0, 0.0, 30.0, 30.0]
    assert annotations[0].instance_count == 2
    assert len(annotations[0].segmentation) == 2
