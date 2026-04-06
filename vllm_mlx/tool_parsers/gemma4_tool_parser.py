# SPDX-License-Identifier: Apache-2.0
"""
Gemma 4 tool call parser for vllm-mlx.

Handles Gemma 4's native tool calling format:
  <|tool_call>call:func_name{key:<|"|>value<|"|>, key2:<|"|>value2<|"|>}<tool_call|>

Multiple tool calls can appear in a single response. The parser extracts
function name and arguments from each <|tool_call>...<tool_call|> block.
"""

import json
import re
import uuid
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)


def _generate_tool_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:8]}"


# Match <|tool_call>call:name{...}<tool_call|>
# The outer group captures the full block, inner groups get name and args.
GEMMA4_TOOL_CALL_PATTERN = re.compile(
    r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>", re.DOTALL
)

# Match key:<|"|>value<|"|> pairs inside the argument block.
# Values may contain nested quotes or special characters.
GEMMA4_PARAM_PATTERN = re.compile(
    r"(\w+):<\|\"\|>(.*?)<\|\"\|>", re.DOTALL
)

# Thinking tags used by Gemma 4
GEMMA4_THINK_PATTERN = re.compile(
    r"<\|channel>thought\n.*?(?=<\|tool_call>|$)", re.DOTALL
)


def _parse_gemma4_args(args_block: str) -> dict[str, Any]:
    """Parse Gemma 4 argument format into a dict.

    Gemma 4 uses: key:<|"|>value<|"|>, key2:<|"|>value2<|"|>
    Also handles bare key:value without quote markers.

    Args:
        args_block: Raw argument string from inside the braces.

    Returns:
        Dictionary of parsed arguments.
    """
    arguments: dict[str, Any] = {}

    # First try the quoted format: key:<|"|>value<|"|>
    for match in GEMMA4_PARAM_PATTERN.finditer(args_block):
        key = match.group(1)
        value = match.group(2)
        # Try to parse as JSON for typed values (numbers, bools, etc.)
        try:
            arguments[key] = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            arguments[key] = value

    # If no quoted params found, try bare key:value format
    if not arguments:
        # Remove any remaining quote markers
        cleaned = args_block.replace('<|"|>', '"')
        # Try parsing as JSON directly
        try:
            parsed = json.loads("{" + cleaned + "}")
            if isinstance(parsed, dict):
                arguments = parsed
        except (json.JSONDecodeError, ValueError):
            # Try comma-separated key:value pairs
            for pair in args_block.split(","):
                pair = pair.strip()
                if ":" in pair:
                    key, _, value = pair.partition(":")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key:
                        try:
                            arguments[key] = json.loads(value)
                        except (json.JSONDecodeError, ValueError):
                            arguments[key] = value

    return arguments


@ToolParserManager.register_module(["gemma4", "gemma"])
class Gemma4ToolParser(ToolParser):
    """
    Tool call parser for Google Gemma 4 models.

    Supports Gemma 4's native tool call format:
      <|tool_call>call:func_name{key:<|"|>value<|"|>}<tool_call|>

    Also handles:
      - Multiple tool calls in a single response
      - Thinking/reasoning blocks before tool calls
      - Streaming extraction

    Used when --enable-auto-tool-choice --tool-call-parser gemma4 are set.
    """

    SUPPORTS_NATIVE_TOOL_FORMAT = True

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from a complete Gemma 4 model response."""
        tool_calls: list[dict[str, Any]] = []
        cleaned_text = model_output

        # Strip thinking blocks
        cleaned_text = GEMMA4_THINK_PATTERN.sub("", cleaned_text).strip()
        cleaned_text = self.strip_think_tags(cleaned_text)

        # Extract all tool calls
        matches = GEMMA4_TOOL_CALL_PATTERN.findall(cleaned_text)
        for func_name, args_block in matches:
            arguments = _parse_gemma4_args(args_block)
            tool_calls.append(
                {
                    "id": _generate_tool_id(),
                    "name": func_name.strip(),
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                }
            )

        # Remove matched tool call blocks from content
        if matches:
            cleaned_text = GEMMA4_TOOL_CALL_PATTERN.sub("", cleaned_text).strip()

        # Also check for text-format fallback
        if not tool_calls and self.has_text_format_tool_call(cleaned_text):
            tool_calls = self.extract_text_format_tool_calls(cleaned_text)
            if tool_calls:
                cleaned_text = ""

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=cleaned_text if cleaned_text else None,
            )
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=cleaned_text
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int] | None = None,
        current_token_ids: Sequence[int] | None = None,
        delta_token_ids: Sequence[int] | None = None,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Extract tool calls from streaming Gemma 4 output."""
        # Count open/close tool call tags
        open_count = current_text.count("<|tool_call>")
        close_count = current_text.count("<tool_call|>")
        prev_close_count = previous_text.count("<tool_call|>")

        if open_count > 0:
            if open_count > close_count:
                # Inside an incomplete tool call block — suppress output
                return None

            if close_count > prev_close_count:
                # New tool call(s) completed in this delta
                result = self.extract_tool_calls(current_text, request)
                if result.tools_called:
                    new_calls = result.tool_calls[prev_close_count:]
                    if new_calls:
                        return {
                            "tool_calls": [
                                {
                                    "index": prev_close_count + i,
                                    "id": tc["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tc["name"],
                                        "arguments": tc["arguments"],
                                    },
                                }
                                for i, tc in enumerate(new_calls)
                            ]
                        }

            return {"content": delta_text}

        return {"content": delta_text}

    def has_pending_tool_call(self, text: str) -> bool:
        """Check if text contains incomplete Gemma 4 tool call markup."""
        return (
            "<|tool_call>" in text
            and "<tool_call|>" not in text
        ) or self.has_text_format_tool_call(text)
