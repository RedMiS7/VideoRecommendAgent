from dataclasses import dataclass, field
from typing import Literal, Protocol


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


class ModelError(RuntimeError):
    def __init__(self, kind: str, message: str):
        self.kind = kind
        super().__init__(message)


class LanguageModel(Protocol):
    def generate_json(self, messages: list[Message], schema: dict) -> dict: ...


class EmbeddingModel(Protocol):
    @property
    def space_id(self) -> str: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class CallLog:
    events: list[dict] = field(default_factory=list)
