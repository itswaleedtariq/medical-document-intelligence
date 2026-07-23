import re

PROHIBITED_PATTERNS = (
    r"\bdiagnose me\b",
    r"\bwhat disease do i have\b",
    r"\btell me exactly what i have\b",
    r"\bprescribe\b",
    r"\bwhat medicine should i take\b",
    r"\bwhat medication should i take\b",
    r"\bwhat dosage should i take\b",
    r"\bchange my dosage\b",
    r"\bincrease my dosage\b",
    r"\bdecrease my dosage\b",
    r"\bstop taking my medication\b",
    r"\bshould i stop taking\b",
)


def get_medical_request_refusal(
    question: str,
) -> str | None:
    """
    Detect direct requests for personalized diagnosis or treatment.

    Questions asking what is explicitly written in a document remain
    allowed, such as: "What diagnosis is mentioned in the report?"
    """
    normalized_question = " ".join(question.lower().split())

    for pattern in PROHIBITED_PATTERNS:
        if re.search(pattern, normalized_question):
            return (
                "I can explain information explicitly written in the "
                "uploaded document, but I cannot diagnose a disease, "
                "prescribe medication, or recommend dosage changes. "
                "Please discuss personalized medical decisions with a "
                "qualified healthcare professional."
            )

    return None
