# SPDX-License-Identifier: Apache-2.0
"""
Logits processors for jump-forward decoding of tool call structural tokens.

When models generate tool calls in XML format (e.g., MiniMax's
<minimax:tool_call>), many tokens are predictable structural markup.
By biasing logits toward the expected next token, we accelerate generation
of these structural sequences without constraining the model's free choices
for argument values.

Usage:
    processor = create_tool_logits_processor("minimax", tokenizer)
    if processor:
        # Pass to BatchGenerator via logits_processors
        ...
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class ToolLogitsProcessor(Protocol):
    """Protocol for tool call logits processors."""

    def __call__(self, token_ids: Any, logits: Any) -> Any:
        """Apply logits bias based on current generation state."""
        ...

    def reset(self) -> None:
        """Reset state for a new generation."""
        ...


class MiniMaxToolLogitsProcessor:
    """
    Logits processor that biases structural tokens in MiniMax tool calls.

    MiniMax tool call format:
        <minimax:tool_call>
        <invoke name="function_name">
        <parameter name="param">value</parameter>
        </invoke>
        </minimax:tool_call>

    After detecting the start of a structural sequence, biases the logits
    toward the expected continuation tokens. Does not constrain the model
    for free-form content (function names, parameter names, values).

    State machine:
        idle -> after_invoke (saw "<invoke")
        after_invoke -> idle (saw ' name="')
        idle -> after_param_value (saw '">...something not starting with <')
        ... etc.

    The processor uses a simpler approach: it pre-tokenizes known structural
    patterns and when the recent tokens match a pattern prefix, biases toward
    the next token in the pattern.
    """

    # Structural patterns that follow predictable sequences
    PATTERNS = [
        # After <invoke → expect ' name="'
        (' name="', "<invoke"),
        # After param value closing → expect </parameter>
        ("</parameter>", None),  # Triggered by seeing '">' after param value
        # After </parameter> block → could be another <parameter or </invoke>
        ("</invoke>", None),  # Triggered contextually
        # After </invoke> → expect </minimax:tool_call>
        ("</minimax:tool_call>", "</invoke>"),
    ]

    def __init__(self, tokenizer: Any, bias_strength: float = 20.0):
        """
        Initialize the MiniMax tool logits processor.

        Args:
            tokenizer: The tokenizer to use for encoding patterns.
            bias_strength: Logits bias to add to expected tokens.
        """
        self.tokenizer = tokenizer
        self.bias_strength = bias_strength

        # Pre-tokenize structural fragments
        self._pattern_tokens: dict[str, list[int]] = {}
        for pattern, _ in self.PATTERNS:
            tokens = tokenizer.encode(pattern, add_special_tokens=False)
            if tokens:
                self._pattern_tokens[pattern] = tokens

        # State tracking
        self._recent_text = ""
        self._active_pattern: str | None = None
        self._pattern_pos = 0  # Position within active pattern's token sequence
        self._last_param_close_pos = -1  # Track last </parameter> position to avoid re-triggering

    def reset(self) -> None:
        """Reset state for a new generation."""
        self._recent_text = ""
        self._active_pattern = None
        self._pattern_pos = 0
        self._last_param_close_pos = -1

    def __call__(self, token_ids: Any, logits: Any) -> Any:
        """
        Apply logits bias for structural tool call tokens.

        Args:
            token_ids: Previously generated token IDs.
            logits: Current logits tensor (1, vocab_size).

        Returns:
            Modified logits tensor.
        """
        import mlx.core as mx

        # Decode last few tokens to track context
        if hasattr(token_ids, "tolist"):
            id_list = token_ids.tolist()
        else:
            id_list = list(token_ids)

        if not id_list:
            return logits

        # Decode last token to update recent text
        last_token_text = self.tokenizer.decode(
            [id_list[-1]], skip_special_tokens=False
        )
        self._recent_text += last_token_text
        # Keep only last 200 chars for matching
        if len(self._recent_text) > 200:
            self._recent_text = self._recent_text[-200:]

        # If we're tracking an active pattern, bias toward next token
        if self._active_pattern is not None:
            pattern_tokens = self._pattern_tokens.get(self._active_pattern, [])
            if self._pattern_pos < len(pattern_tokens):
                target_token = pattern_tokens[self._pattern_pos]
                self._pattern_pos += 1

                # Add bias to the expected token
                bias = mx.zeros_like(logits)
                if logits.ndim == 2:
                    bias[0, target_token] = self.bias_strength
                else:
                    bias[target_token] = self.bias_strength
                return logits + bias
            else:
                # Pattern complete — skip trigger check this call to avoid
                # re-activating on stale _recent_text
                self._active_pattern = None
                self._pattern_pos = 0
                return logits

        # Check if we should start tracking a pattern
        for pattern, trigger in self.PATTERNS:
            if trigger and self._recent_text.rstrip().endswith(trigger):
                pattern_tokens = self._pattern_tokens.get(pattern, [])
                if pattern_tokens:
                    self._active_pattern = pattern
                    self._pattern_pos = 0
                    # Bias first token
                    target_token = pattern_tokens[0]
                    self._pattern_pos = 1

                    bias = mx.zeros_like(logits)
                    if logits.ndim == 2:
                        bias[0, target_token] = self.bias_strength
                    else:
                        bias[target_token] = self.bias_strength
                    return logits + bias

        # Check for </invoke> trigger: after seeing </parameter>\n or similar
        # Only trigger once per </parameter> occurrence to avoid repeated bias
        param_close_pos = self._recent_text.rfind("</parameter>")
        if param_close_pos > self._last_param_close_pos:
            after_param = self._recent_text[param_close_pos + len("</parameter>"):]
            # If the text after </parameter> is whitespace only, we might
            # be about to see </invoke> or another <parameter
            stripped = after_param.strip()
            if not stripped:
                self._last_param_close_pos = param_close_pos
                pattern = "</invoke>"
                pattern_tokens = self._pattern_tokens.get(pattern, [])
                if pattern_tokens:
                    target_token = pattern_tokens[0]
                    bias = mx.zeros_like(logits)
                    if logits.ndim == 2:
                        bias[0, target_token] = self.bias_strength * 0.5
                    else:
                        bias[target_token] = self.bias_strength * 0.5
                    return logits + bias

        return logits


def create_tool_logits_processor(
    parser_name: str, tokenizer: Any, bias_strength: float = 20.0
) -> ToolLogitsProcessor | None:
    """
    Factory function to create a tool logits processor for a given parser.

    Args:
        parser_name: Name of the tool call parser (e.g., "minimax").
        tokenizer: The tokenizer instance.
        bias_strength: Logits bias strength.

    Returns:
        A logits processor instance, or None if not supported for this parser.
    """
    if parser_name == "minimax":
        return MiniMaxToolLogitsProcessor(tokenizer, bias_strength=bias_strength)
    # Future: add support for other parsers (hermes, llama, etc.)
    return None
