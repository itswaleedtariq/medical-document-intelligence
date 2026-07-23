# Synthetic Medical Document Grounded QA Dataset

## Overview

This dataset supports supervised fine-tuning of the Medical Document
Intelligence Assistant.

It contains synthetic laboratory reports and grounded question-answer
examples. No real patient records are included.

## Intended tasks

- Laboratory value lookup
- Comparison with printed reference ranges
- Document summarization
- Insufficient-evidence handling
- Diagnosis and treatment refusal
- Evidence citation generation
- Structured JSON response generation

## Data source

All examples are generated programmatically from predefined synthetic
laboratory test templates.

The dataset does not contain real names, contact information, medical
record numbers, or identifiable patient information.

## Dataset structure

Each example contains:

- `example_id`
- `source_document_id`
- `task_type`
- `prompt`
- `completion`
- `metadata`

The prompt and completion use conversational role-based messages.

## Splitting policy

Examples are split by `source_document_id`.

All examples generated from one synthetic report remain in the same
training, validation, or test split. This reduces source-document
leakage between evaluation splits.

## Safety behavior

The dataset teaches the model to:

- Use only supplied evidence
- Avoid unsupported medical claims
- Avoid independent diagnosis
- Avoid prescribing medication
- Avoid dosage recommendations
- Return insufficient evidence when information is absent
- Cite supplied evidence identifiers

## Limitations

- Laboratory ranges are synthetic.
- The dataset is not clinically validated.
- It does not represent the full diversity of real medical documents.
- It does not establish clinical safety or effectiveness.
- It must not be used as a substitute for professional medical review.