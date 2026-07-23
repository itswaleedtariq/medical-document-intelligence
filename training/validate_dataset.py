import json
import re
from collections import Counter
from pathlib import Path

from backend.app.schemas.answer import (
    AnswerStatus,
    ModelGroundedAnswer,
)
from training.dataset_schema import TrainingExample

SPLIT_PATHS = {
    "train": Path("data/train/train.jsonl"),
    "validation": Path("data/validation/validation.jsonl"),
    "test": Path("data/test/test.jsonl"),
}

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@"
    r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

PHONE_PATTERN = re.compile(r"\b(?:\+?\d[\d -]{8,}\d)\b")

CNIC_PATTERN = re.compile(r"\b\d{5}-\d{7}-\d\b")

EVIDENCE_PATTERN = re.compile(r"\[(E\d+)\]")


def load_split(
    split_name: str,
    path: Path,
) -> list[TrainingExample]:
    if not path.exists():
        raise FileNotFoundError(f"{split_name} dataset not found: {path}")

    examples: list[TrainingExample] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        for line_number, line in enumerate(
            input_file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                examples.append(TrainingExample.model_validate_json(line))
            except Exception as error:
                raise ValueError(
                    f"{split_name}, line {line_number}: invalid training schema."
                ) from error

    if not examples:
        raise ValueError(f"{split_name} split is empty.")

    return examples


def validate_no_obvious_pii(
    example: TrainingExample,
) -> None:
    combined_text = "\n".join(
        [message.content for message in (example.prompt + example.completion)]
    )

    patterns = {
        "email address": EMAIL_PATTERN,
        "phone number": PHONE_PATTERN,
        "CNIC": CNIC_PATTERN,
    }

    for label, pattern in patterns.items():
        if pattern.search(combined_text):
            raise ValueError(f"{example.example_id}: possible {label} detected.")


def validate_completion(
    example: TrainingExample,
) -> None:
    completion_text = example.completion[0].content

    try:
        payload = json.loads(completion_text)

        answer = ModelGroundedAnswer.model_validate(payload)

    except Exception as error:
        raise ValueError(
            f"{example.example_id}: completion does not match ModelGroundedAnswer."
        ) from error

    prompt_text = "\n".join(message.content for message in example.prompt)

    allowed_evidence_ids = set(EVIDENCE_PATTERN.findall(prompt_text))

    if answer.status == AnswerStatus.ANSWERED:
        if not answer.facts:
            raise ValueError(
                f"{example.example_id}: answered response contains no facts."
            )

        for fact in answer.facts:
            if not fact.citations:
                raise ValueError(f"{example.example_id}: fact has no citation.")

            invalid_citations = set(fact.citations) - allowed_evidence_ids

            if invalid_citations:
                raise ValueError(
                    f"{example.example_id}: invalid citations {invalid_citations}."
                )

    if (
        answer.status
        in {
            AnswerStatus.REFUSED,
            AnswerStatus.INSUFFICIENT_EVIDENCE,
        }
        and answer.facts
    ):
        raise ValueError(
            f"{example.example_id}: refused or insufficient response contains facts."
        )


def validate_all_splits() -> None:
    splits = {
        split_name: load_split(
            split_name=split_name,
            path=path,
        )
        for split_name, path in (SPLIT_PATHS.items())
    }

    all_example_ids: set[str] = set()

    source_ids_by_split: dict[
        str,
        set[str],
    ] = {}

    for split_name, examples in splits.items():
        split_example_ids = {example.example_id for example in examples}

        duplicate_example_ids = all_example_ids & split_example_ids

        if duplicate_example_ids:
            raise ValueError(
                f"Duplicate example IDs across splits: {duplicate_example_ids}"
            )

        all_example_ids.update(split_example_ids)

        source_ids_by_split[split_name] = {
            example.source_document_id for example in examples
        }

        for example in examples:
            validate_no_obvious_pii(example)
            validate_completion(example)

        task_counts = Counter(example.task_type for example in examples)

        print(
            f"{split_name}: {len(examples)} examples, "
            f"{len(source_ids_by_split[split_name])} "
            "source documents"
        )

        print(
            "  Tasks:",
            dict(sorted(task_counts.items())),
        )

    split_names = list(source_ids_by_split)

    for first_index, first_name in enumerate(split_names):
        for second_name in split_names[first_index + 1 :]:
            source_overlap = (
                source_ids_by_split[first_name] & source_ids_by_split[second_name]
            )

            if source_overlap:
                raise ValueError(
                    "Source-document leakage between "
                    f"{first_name} and {second_name}: "
                    f"{source_overlap}"
                )

    print()
    print("Dataset validation passed.")
    print(f"Total examples: {len(all_example_ids)}")
    print("No duplicate IDs or source-document leakage detected.")


if __name__ == "__main__":
    validate_all_splits()
