import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from training.dataset_schema import (
    ChatMessage,
    TrainingExample,
)

SYSTEM_MESSAGE = """
You are a medical document information assistant.

Use only the supplied document evidence.

Rules:
1. Do not invent values or findings.
2. Do not independently diagnose a disease.
3. Do not prescribe medication.
4. Do not recommend dosage changes.
5. Compare values only with reference ranges printed in the evidence.
6. Every factual statement must cite a supplied evidence identifier.
7. Return one valid JSON object without Markdown code fences.
""".strip()


@dataclass(frozen=True)
class LabDefinition:
    name: str
    unit: str
    low: float
    high: float
    decimals: int


@dataclass(frozen=True)
class GeneratedLabResult:
    name: str
    value: float
    displayed_value: str
    unit: str
    low: float
    high: float
    displayed_low: str
    displayed_high: str
    status: str


LAB_CATALOG = (
    LabDefinition(
        name="Hemoglobin",
        unit="g/dL",
        low=12.0,
        high=15.5,
        decimals=1,
    ),
    LabDefinition(
        name="White Blood Cell Count",
        unit="x10^9/L",
        low=4.0,
        high=11.0,
        decimals=1,
    ),
    LabDefinition(
        name="Platelet Count",
        unit="x10^9/L",
        low=150.0,
        high=400.0,
        decimals=0,
    ),
    LabDefinition(
        name="Mean Corpuscular Volume",
        unit="fL",
        low=80.0,
        high=100.0,
        decimals=0,
    ),
    LabDefinition(
        name="Fasting Glucose",
        unit="mg/dL",
        low=70.0,
        high=99.0,
        decimals=0,
    ),
    LabDefinition(
        name="Creatinine",
        unit="mg/dL",
        low=0.6,
        high=1.2,
        decimals=1,
    ),
    LabDefinition(
        name="Sodium",
        unit="mmol/L",
        low=135.0,
        high=145.0,
        decimals=0,
    ),
    LabDefinition(
        name="Potassium",
        unit="mmol/L",
        low=3.5,
        high=5.1,
        decimals=1,
    ),
)


def format_number(
    value: float,
    decimals: int,
) -> str:
    return f"{value:.{decimals}f}"


def sample_lab_result(
    definition: LabDefinition,
    rng: random.Random,
) -> GeneratedLabResult:
    selected_status = rng.choices(
        population=[
            "below_provided_range",
            "within_provided_range",
            "above_provided_range",
        ],
        weights=[
            0.25,
            0.50,
            0.25,
        ],
        k=1,
    )[0]

    range_width = definition.high - definition.low

    if selected_status == "below_provided_range":
        value = rng.uniform(
            definition.low - (range_width * 0.35),
            definition.low - (range_width * 0.05),
        )

    elif selected_status == "above_provided_range":
        value = rng.uniform(
            definition.high + (range_width * 0.05),
            definition.high + (range_width * 0.35),
        )

    else:
        value = rng.uniform(
            definition.low,
            definition.high,
        )

    rounded_value = round(
        value,
        definition.decimals,
    )

    # Recalculate status after rounding.
    if rounded_value < definition.low:
        final_status = "below_provided_range"
    elif rounded_value > definition.high:
        final_status = "above_provided_range"
    else:
        final_status = "within_provided_range"

    return GeneratedLabResult(
        name=definition.name,
        value=rounded_value,
        displayed_value=format_number(
            rounded_value,
            definition.decimals,
        ),
        unit=definition.unit,
        low=definition.low,
        high=definition.high,
        displayed_low=format_number(
            definition.low,
            definition.decimals,
        ),
        displayed_high=format_number(
            definition.high,
            definition.decimals,
        ),
        status=final_status,
    )


def status_phrase(status: str) -> str:
    phrases = {
        "below_provided_range": ("below the printed reference range"),
        "within_provided_range": ("within the printed reference range"),
        "above_provided_range": ("above the printed reference range"),
    }

    return phrases[status]


def build_document_text(
    source_document_id: str,
    results: list[GeneratedLabResult],
) -> str:
    sections = [
        "SYNTHETIC LABORATORY REPORT",
        f"Report ID: {source_document_id}",
        "",
    ]

    for result in results:
        sections.extend(
            [
                result.name,
                (f"Result: {result.displayed_value} {result.unit}"),
                (
                    "Reference range: "
                    f"{result.displayed_low}-"
                    f"{result.displayed_high} "
                    f"{result.unit}"
                ),
                "",
            ]
        )

    sections.extend(
        [
            "No medications are listed.",
            "No confirmed diagnosis is listed.",
            "This report contains synthetic information only.",
        ]
    )

    return "\n".join(sections)


def build_user_prompt(
    document_text: str,
    question: str,
) -> str:
    return f"""
DOCUMENT EVIDENCE:
------------------
[E1]
File: synthetic_lab_report.pdf
Page: 1
Text:
{document_text}
------------------

USER QUESTION:
{question}

Required JSON format:
{{
  "status": "answered, insufficient_evidence, or refused",
  "answer": "Direct plain-language response",
  "facts": [
    {{
      "statement": "One factual statement",
      "citations": ["E1"]
    }}
  ],
  "safety_note": "Optional note or null",
  "limitations": []
}}
""".strip()


def make_completion(
    payload: dict,
) -> list[ChatMessage]:
    return [
        ChatMessage(
            role="assistant",
            content=json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
    ]


def create_training_example(
    *,
    example_id: str,
    source_document_id: str,
    task_type: str,
    document_text: str,
    question: str,
    response_payload: dict,
) -> TrainingExample:
    return TrainingExample(
        example_id=example_id,
        source_document_id=source_document_id,
        task_type=task_type,
        prompt=[
            ChatMessage(
                role="system",
                content=SYSTEM_MESSAGE,
            ),
            ChatMessage(
                role="user",
                content=build_user_prompt(
                    document_text=document_text,
                    question=question,
                ),
            ),
        ],
        completion=make_completion(response_payload),
        metadata={
            "synthetic": True,
            "document_type": "laboratory_report",
            "page_number": 1,
            "evidence_ids": "E1",
        },
    )


def generate_examples_for_document(
    document_index: int,
    rng: random.Random,
) -> list[TrainingExample]:
    source_document_id = f"synthetic-lab-{document_index:05d}"

    selected_definitions = rng.sample(
        list(LAB_CATALOG),
        k=4,
    )

    results = [
        sample_lab_result(
            definition=definition,
            rng=rng,
        )
        for definition in selected_definitions
    ]

    document_text = build_document_text(
        source_document_id=source_document_id,
        results=results,
    )

    selected_result = rng.choice(results)

    value_payload = {
        "status": "answered",
        "answer": (
            f"The reported {selected_result.name} result was "
            f"{selected_result.displayed_value} "
            f"{selected_result.unit}."
        ),
        "facts": [
            {
                "statement": (
                    f"{selected_result.name} was reported as "
                    f"{selected_result.displayed_value} "
                    f"{selected_result.unit}."
                ),
                "citations": ["E1"],
            }
        ],
        "safety_note": None,
        "limitations": [],
    }

    comparison_payload = {
        "status": "answered",
        "answer": (
            f"The reported {selected_result.name} result was "
            f"{selected_result.displayed_value} "
            f"{selected_result.unit}. It was "
            f"{status_phrase(selected_result.status)} of "
            f"{selected_result.displayed_low}-"
            f"{selected_result.displayed_high} "
            f"{selected_result.unit}."
        ),
        "facts": [
            {
                "statement": (
                    f"{selected_result.name} was reported as "
                    f"{selected_result.displayed_value} "
                    f"{selected_result.unit}."
                ),
                "citations": ["E1"],
            },
            {
                "statement": (
                    "The printed reference range was "
                    f"{selected_result.displayed_low}-"
                    f"{selected_result.displayed_high} "
                    f"{selected_result.unit}."
                ),
                "citations": ["E1"],
            },
            {
                "statement": (
                    f"The result was {status_phrase(selected_result.status)}."
                ),
                "citations": ["E1"],
            },
        ],
        "safety_note": None,
        "limitations": [
            (
                "The comparison uses only the reference range "
                "printed in the supplied report."
            )
        ],
    }

    summary_statements = [
        (
            f"{result.name}: {result.displayed_value} "
            f"{result.unit}, {status_phrase(result.status)}."
        )
        for result in results
    ]

    summary_payload = {
        "status": "answered",
        "answer": " ".join(summary_statements),
        "facts": [
            {
                "statement": statement,
                "citations": ["E1"],
            }
            for statement in summary_statements
        ],
        "safety_note": (
            "This summary reports document information and "
            "does not provide a diagnosis."
        ),
        "limitations": [
            ("Only the supplied results and printed reference ranges were used.")
        ],
    }

    selected_names = {result.name for result in results}

    missing_definition = next(
        definition
        for definition in LAB_CATALOG
        if definition.name not in selected_names
    )

    insufficient_payload = {
        "status": "insufficient_evidence",
        "answer": (
            f"The supplied document does not report a {missing_definition.name} result."
        ),
        "facts": [],
        "safety_note": None,
        "limitations": [
            ("The requested test is not present in the supplied evidence.")
        ],
    }

    refusal_payload = {
        "status": "refused",
        "answer": (
            "I can explain information explicitly written in "
            "the report, but I cannot diagnose a disease, "
            "prescribe medication, or recommend dosage changes."
        ),
        "facts": [],
        "safety_note": (
            "Personalized medical decisions should be discussed "
            "with a qualified healthcare professional."
        ),
        "limitations": [("This assistant is limited to document explanation.")],
    }

    examples = [
        create_training_example(
            example_id=(f"{source_document_id}-value"),
            source_document_id=source_document_id,
            task_type="value_lookup",
            document_text=document_text,
            question=(f"What was the {selected_result.name} result?"),
            response_payload=value_payload,
        ),
        create_training_example(
            example_id=(f"{source_document_id}-comparison"),
            source_document_id=source_document_id,
            task_type="range_comparison",
            document_text=document_text,
            question=(
                f"How did the {selected_result.name} result "
                f"compare with the printed reference range?"
            ),
            response_payload=comparison_payload,
        ),
        create_training_example(
            example_id=(f"{source_document_id}-summary"),
            source_document_id=source_document_id,
            task_type="summary",
            document_text=document_text,
            question=(
                "Summarize the laboratory results using only "
                "the printed values and reference ranges."
            ),
            response_payload=summary_payload,
        ),
        create_training_example(
            example_id=(f"{source_document_id}-missing"),
            source_document_id=source_document_id,
            task_type="insufficient_evidence",
            document_text=document_text,
            question=(f"What was the {missing_definition.name} result?"),
            response_payload=insufficient_payload,
        ),
        create_training_example(
            example_id=(f"{source_document_id}-refusal"),
            source_document_id=source_document_id,
            task_type="safety_refusal",
            document_text=document_text,
            question=(
                "Diagnose me from this report and prescribe "
                "the medication I should take."
            ),
            response_payload=refusal_payload,
        ),
    ]

    return examples


def generate_dataset(
    document_count: int,
    seed: int,
) -> list[TrainingExample]:
    if document_count < 10:
        raise ValueError("Generate at least 10 source documents.")

    rng = random.Random(seed)
    examples: list[TrainingExample] = []

    for document_index in range(
        1,
        document_count + 1,
    ):
        examples.extend(
            generate_examples_for_document(
                document_index=document_index,
                rng=rng,
            )
        )

    return examples


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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Generate synthetic grounded medical-document training examples.")
    )

    parser.add_argument(
        "--documents",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/synthetic_medical_qa_all.jsonl"),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    examples = generate_dataset(
        document_count=arguments.documents,
        seed=arguments.seed,
    )

    write_jsonl(
        examples=examples,
        output_path=arguments.output,
    )

    print(
        f"Generated {len(examples)} examples from "
        f"{arguments.documents} synthetic reports."
    )
    print(f"Saved to: {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
