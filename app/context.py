"""Immutable citation authority, budgeted as whole chunks in retrieval rank order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from app.models import RetrievedDocument
from app.prompt import build_messages


class ContextBudgetError(RuntimeError):
    """The rendered chat cannot fit, or the token counter/config is invalid."""


@dataclass(frozen=True)
class ContextItem:
    doc_id: str
    chunk_id: str
    rank: int
    title: str
    text: str


@dataclass(frozen=True)
class ContextBundle:
    included: tuple[ContextItem, ...]
    excluded: tuple[ContextItem, ...]
    truncated: bool
    input_token_count: int


def build_context(
    query: str, retrieved: Sequence[RetrievedDocument], *,
    count_prompt_tokens: Callable[[Sequence[Mapping[str, str]]], int],
    context_length: int, max_new_tokens: int,
) -> ContextBundle:
    if (type(context_length) is not int or type(max_new_tokens) is not int
            or not 0 < max_new_tokens < context_length):
        raise ContextBudgetError("Require 0 < max_new_tokens < context_length")
    items = tuple(ContextItem(r.doc_id, r.chunk_id, r.rank, r.title, r.text)
                  for r in retrieved)
    if any(left.rank >= right.rank for left, right in zip(items, items[1:])):
        raise ContextBudgetError("Retrieved evidence must be ordered by ascending rank")
    counts: dict[int, int] = {}

    def count(prefix: int) -> int:
        if prefix not in counts:
            value = count_prompt_tokens(build_messages(query, items[:prefix]))
            if type(value) is not int or value <= 0:
                raise ContextBudgetError("Prompt counter must return a positive integer")
            counts[prefix] = value
        return counts[prefix]

    budget = context_length - max_new_tokens
    size = len(items)
    if count(size) <= budget:
        return ContextBundle(items, (), False, count(size))
    if count(0) > budget:
        raise ContextBudgetError("Base request plus output reserve exceeds context_length")
    # top_k is bounded (<=10), so inspect every remaining prefix instead of
    # assuming tokenizer/chat-template counts are monotonic as chunks are added.
    largest_fitting = 0
    for prefix in range(1, size):
        if count(prefix) <= budget:
            largest_fitting = prefix
    return ContextBundle(
        items[:largest_fitting],
        items[largest_fitting:],
        True,
        count(largest_fitting),
    )
