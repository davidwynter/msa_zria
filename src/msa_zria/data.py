from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

InputMode = Literal["triples", "text", "hybrid"]


class Triple(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    predicate: str
    object: str

    def as_line(self) -> str:
        return f"{self.subject} | {self.predicate} | {self.object}"

    def as_sentence(self) -> str:
        predicate = self.predicate.replace("_", " ")
        return f"{self.subject} {predicate} {self.object}."


class TrainingExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str
    response: str
    triples: list[Triple] = Field(default_factory=list)
    natural_language_context: list[str] = Field(default_factory=list)
    system_prompt: str = (
        "You are a customer support reasoning assistant. "
        "Use the provided facts and return only the requested result."
    )

    def render_triples(self) -> str:
        return "\n".join(triple.as_line() for triple in self.triples)

    def render_text(self) -> str:
        if self.natural_language_context:
            return "\n".join(self.natural_language_context)
        return " ".join(triple.as_sentence() for triple in self.triples)

    def render_user_content(self, input_mode: InputMode) -> str:
        parts = [self.instruction.strip()]

        if input_mode == "triples":
            parts.append("Facts:\n" + self.render_triples())
        elif input_mode == "text":
            parts.append("Context:\n" + self.render_text())
        else:
            parts.append("Facts:\n" + self.render_triples())
            parts.append("Context:\n" + self.render_text())

        return "\n\n".join(part for part in parts if part.strip())

    def to_messages(self, input_mode: InputMode) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.render_user_content(input_mode)},
            {"role": "assistant", "content": self.response},
        ]
