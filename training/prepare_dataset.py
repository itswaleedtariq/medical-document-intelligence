import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from training.dataset_schema import TrainingExample


def load_jsonl(
    input_path: Path,
) -> list[TrainingExample]:
    examples: list[TrainingExample] = []

    with input_path.open(
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
                raise ValueError(f"Invalid example on line {line_number}.") from error

    if not examples:
        raise ValueError("The input dataset contains no examples.")

    return examples


def split_examples_by_source(
    examples: list[TrainingExample],
    seed: int,
    train_ratio: float = 0.80,
    validation_ratio: float = 0.10,
) -> dict[str, list[TrainingExample]]:
    grouped_examples: dict[
        str,
        list[TrainingExample],
    ] = defaultdict(list)

    for example in examples:
        grouped_examples[example.source_document_id].append(example)

    source_ids = list(grouped_examples)

    if len(source_ids) < 10:
        raise ValueError("At least 10 source documents are required.")

    rng = random.Random(seed)
    rng.shuffle(source_ids)

    source_count = len(source_ids)

    train_source_count = int(source_count * train_ratio)

    validation_source_count = max(
        1,
        int(source_count * validation_ratio),
    )

    test_source_count = source_count - train_source_count - validation_source_count

    if test_source_count < 1:
        raise ValueError("The split would produce an empty test set.")

    train_sources = set(source_ids[:train_source_count])

    validation_start = train_source_count
    validation_end = validation_start + validation_source_count

    validation_sources = set(source_ids[validation_start:validation_end])

    test_sources = set(source_ids[validation_end:])

    splits = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for source_id, source_examples in grouped_examples.items():
        if source_id in train_sources:
            split_name = "train"

        elif source_id in validation_sources:
            split_name = "validation"

        elif source_id in test_sources:
            split_name = "test"

        else:
            raise RuntimeError(f"Source was not assigned: {source_id}")

        splits[split_name].extend(source_examples)

    for split_examples in splits.values():
        rng.shuffle(split_examples)

    return splits


def write_jsonl(
    examples: list[TrainingExample],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for example in examples:
            output_file.write(example.model_dump_json())
            output_file.write("\n")


def task_distribution(
    examples: list[TrainingExample],
) -> dict[str, int]:
    counts = Counter(example.task_type for example in examples)

    return dict(sorted(counts.items()))


def build_manifest(
    splits: dict[str, list[TrainingExample]],
    seed: int,
) -> dict:
    manifest: dict = {
        "dataset_name": ("synthetic-medical-document-grounded-qa"),
        "version": "1.0.0",
        "seed": seed,
        "splits": {},
    }

    for split_name, examples in splits.items():
        source_ids = {example.source_document_id for example in examples}

        manifest["splits"][split_name] = {
            "example_count": len(examples),
            "source_document_count": len(source_ids),
            "task_distribution": task_distribution(examples),
        }

    return manifest


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Split medical training examples by source document.")
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/synthetic_medical_qa_all.jsonl"),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    examples = load_jsonl(arguments.input)

    splits = split_examples_by_source(
        examples=examples,
        seed=arguments.seed,
    )

    output_paths = {
        "train": Path("data/train/train.jsonl"),
        "validation": Path("data/validation/validation.jsonl"),
        "test": Path("data/test/test.jsonl"),
    }

    for split_name, output_path in output_paths.items():
        write_jsonl(
            examples=splits[split_name],
            output_path=output_path,
        )

        print(f"{split_name}: {len(splits[split_name])} examples")

    manifest = build_manifest(
        splits=splits,
        seed=arguments.seed,
    )

    manifest_path = Path("data/processed/dataset_manifest.json")

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as manifest_file:
        json.dump(
            manifest,
            manifest_file,
            indent=2,
        )

    print(f"Manifest saved to: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
