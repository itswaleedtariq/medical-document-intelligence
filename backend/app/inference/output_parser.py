import json
import re

from pydantic import ValidationError

from backend.app.inference.exceptions import ModelOutputError
from backend.app.schemas.answer import (
    AnswerStatus,
    ModelGroundedAnswer,
)


def _extract_json_text(raw_output: str) -> str:
    cleaned = raw_output.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    ).strip()

    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")

    if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
        raise ModelOutputError("The model did not return a JSON object.")

    return cleaned[first_brace : last_brace + 1]


def parse_grounded_model_output(
    raw_output: str,
    allowed_evidence_ids: set[str],
) -> ModelGroundedAnswer:
    json_text = _extract_json_text(raw_output)

    try:
        payload = json.loads(json_text)
        answer = ModelGroundedAnswer.model_validate(payload)

    except (
        json.JSONDecodeError,
        ValidationError,
    ) as error:
        raise ModelOutputError(
            "The model returned invalid structured output."
        ) from error

    if answer.status == AnswerStatus.ANSWERED:
        if not answer.facts:
            raise ModelOutputError("An answered response must contain cited facts.")

        for fact in answer.facts:
            if not fact.citations:
                raise ModelOutputError("Every answered fact must contain a citation.")

            invalid_citations = set(fact.citations) - allowed_evidence_ids

            if invalid_citations:
                raise ModelOutputError(
                    "The model cited evidence that was not supplied: "
                    + ", ".join(sorted(invalid_citations))
                )

    elif answer.facts:
        raise ModelOutputError(
            "Refused or insufficient responses must not contain factual claims."
        )

    return answer
