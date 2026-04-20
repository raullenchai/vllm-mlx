# SPDX-License-Identifier: Apache-2.0
"""Pipeline stage interfaces — contracts for pluggable decode strategies.

Default implementation (StandardDecode) wraps mlx-lm's BatchGenerator
public API. Optimized implementations can add MTP, speculative decode,
Medusa heads, etc. by implementing DecodeStrategy or DecodePlugin.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class TokenResult:
    """Result from one decode step for one sequence."""

    uid: int
    token: int
    logprobs: Any = None
    finish_reason: str | None = None
    prompt_cache: list | None = None


@dataclass
class DecodeRequest:
    """Request to insert into the decode stage."""

    tokens: list[int]
    max_tokens: int
    sampler: Callable | None = None
    logits_processors: list[Callable] | None = None
    cache: Any = None


class DecodeStrategy(ABC):
    """Token generation — the core decode loop.

    Default implementation wraps mlx-lm BatchGenerator's public API
    (insert/next/remove). Optimized implementations add MTP, speculative
    decode, etc. on top.
    """

    @abstractmethod
    def insert(self, request: DecodeRequest) -> int:
        """Insert a request. Returns assigned UID."""
        ...

    @abstractmethod
    def step(self) -> list[TokenResult]:
        """Run one decode step. Returns tokens for all active sequences."""
        ...

    @abstractmethod
    def remove(self, uid: int) -> Any | None:
        """Remove a sequence. Returns prompt cache if available."""
        ...

    @abstractmethod
    def has_active(self) -> bool:
        """Whether there are active sequences."""
        ...

    def close(self) -> None:  # noqa: B027
        """Release resources."""


class DecodePlugin(ABC):
    """Optional decode-time optimization that wraps a DecodeStrategy.

    Lifecycle: on_insert → wrap_step (repeated) → on_remove → on_close
    """

    @abstractmethod
    def wrap_step(
        self, base_step: Callable[[], list[TokenResult]]
    ) -> list[TokenResult]:
        """Wrap the base decode step with custom logic."""
        ...

    def on_insert(self, request: DecodeRequest, uid: int) -> None:  # noqa: B027
        """Called when a new sequence is inserted."""

    def on_remove(self, uid: int) -> None:  # noqa: B027
        """Called when a sequence is removed."""

    def on_close(self) -> None:  # noqa: B027
        """Called when the decode strategy is closed."""
