from typing import Literal

from pydantic import BaseModel, Field, model_validator

MessageRole = Literal[
    "system",
    "user",
    "assistant",
]

TaskType = Literal[
    "value_lookup",
    "range_comparison",
    "summary",
    "insufficient_evidence",
    "safety_refusal",
]


class ChatMessage(BaseModel):
    role: MessageRole
    content: str = Field(min_length=1)


class TrainingExample(BaseModel):
    example_id: str = Field(min_length=1)
    source_document_id: str = Field(min_length=1)
    task_type: TaskType

    prompt: list[ChatMessage] = Field(min_length=1)
    completion: list[ChatMessage] = Field(
        min_length=1,
        max_length=1,
    )

    metadata: dict[
        str,
        str | int | float | bool | None,
    ] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_conversation(
        self,
    ) -> "TrainingExample":
        if self.completion[0].role != "assistant":
            raise ValueError("Completion must contain one assistant message.")

        if not any(message.role == "user" for message in self.prompt):
            raise ValueError("Prompt must contain a user message.")

        return self
