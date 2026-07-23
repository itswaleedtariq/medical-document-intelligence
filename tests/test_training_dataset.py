import json
import random

from backend.app.schemas.answer import (
    ModelGroundedAnswer,
)
from training.generate_synthetic_dataset import (
    generate_dataset,
    generate_examples_for_document,
)
from training.prepare_dataset import (
    split_examples_by_source,
)


def test_generator_creates_all_task_types() -> None:
    examples = generate_examples_for_document(
        document_index=1,
        rng=random.Random(42),
    )

    assert len(examples) == 5

    task_types = {example.task_type for example in examples}

    assert task_types == {
        "value_lookup",
        "range_comparison",
        "summary",
        "insufficient_evidence",
        "safety_refusal",
    }


def test_generated_completions_match_schema() -> None:
    examples = generate_examples_for_document(
        document_index=1,
        rng=random.Random(42),
    )

    for example in examples:
        completion_payload = json.loads(example.completion[0].content)

        validated_answer = ModelGroundedAnswer.model_validate(completion_payload)

        assert validated_answer.answer


def test_dataset_split_prevents_source_leakage() -> None:
    examples = generate_dataset(
        document_count=20,
        seed=42,
    )

    splits = split_examples_by_source(
        examples=examples,
        seed=42,
    )

    train_sources = {example.source_document_id for example in splits["train"]}

    validation_sources = {
        example.source_document_id for example in splits["validation"]
    }

    test_sources = {example.source_document_id for example in splits["test"]}

    assert train_sources.isdisjoint(validation_sources)

    assert train_sources.isdisjoint(test_sources)

    assert validation_sources.isdisjoint(test_sources)
