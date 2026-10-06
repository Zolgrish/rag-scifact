"""Local LLM generator abstraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence


@dataclass(frozen=True)
class GeneratorResult:
    text: str
    model_id: str
    usage: Mapping[str, int] | None = None
    generation_ms: float | None = None


class Generator(Protocol):
    """Runtime-independent local generation interface."""

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> GeneratorResult:
        """Generate text from messages or raise an infrastructure exception."""

