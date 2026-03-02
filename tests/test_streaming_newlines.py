# SPDX-License-Identifier: Apache-2.0
"""
Tests for streaming markdown newline preservation and reasoning parser behavior.

Reproduces two bugs reported by users:
1. Markdown newlines stripped in streaming mode (whitespace-only chunks dropped)
2. Qwen3 reasoning parser eating all content when model doesn't use <think> tags

These tests exercise the reasoning parser directly, independent of the HTTP server.
"""

import pytest

from vllm_mlx.reasoning import DeltaMessage, get_parser


class TestQwen3NoTagStreaming:
    """Test Qwen3 parser behavior when model output has NO <think> tags.

    Bug: When --reasoning-parser qwen3 is active but the model doesn't
    emit <think> tags (e.g., 8-bit quantized models that think inline),
    ALL output goes to reasoning stream and content is empty.
    """

    @pytest.fixture
    def parser(self):
        return get_parser("qwen3")()

    def test_no_tags_streaming_corrected_by_finalize(self, parser):
        """When no <think> tags appear, finalize_streaming corrects to content.

        During streaming, the base parser defaults all output to reasoning
        (to support implicit think mode where </think> hasn't arrived yet).
        At stream end, finalize_streaming detects no tags were ever seen and
        emits the full text as a content correction.
        """
        parser.reset_state()

        text = "Hello! Here is a markdown example:\n\n# Heading\n\n- Item 1\n- Item 2\n"

        accumulated = ""
        reasoning_parts = []

        for char in text:
            prev = accumulated
            accumulated += char
            result = parser.extract_reasoning_streaming(prev, accumulated, char)
            if result and result.reasoning:
                reasoning_parts.append(result.reasoning)

        # During streaming: everything goes to reasoning (correct — can't know
        # yet whether </think> will come)
        assert "".join(reasoning_parts) == text

        # finalize_streaming corrects: no tags seen → reclassify as content
        correction = parser.finalize_streaming(accumulated)
        assert correction is not None
        assert correction.content == text

    def test_no_tags_nonstreaming_is_fine(self, parser):
        """Non-streaming extraction correctly handles no-tag output."""
        text = "Hello! Here is a markdown example."
        reasoning, content = parser.extract_reasoning(text)
        assert reasoning is None
        assert content == text

    def test_with_tags_still_works(self, parser):
        """Ensure fix doesn't break normal <think>...</think> flow."""
        parser.reset_state()

        tokens = ["<think>", "Let me think", "</think>", "The answer is 42."]
        accumulated = ""
        content_parts = []
        reasoning_parts = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result:
                if result.content:
                    content_parts.append(result.content)
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)

        assert "Let me think" in "".join(reasoning_parts)
        assert "The answer is 42." in "".join(content_parts)

    def test_short_no_tags_finalized_as_content(self, parser):
        """Short no-tag output (under threshold) should be corrected by finalize."""
        parser.reset_state()

        text = "Short answer."
        accumulated = ""

        for char in text:
            prev = accumulated
            accumulated += char
            parser.extract_reasoning_streaming(prev, accumulated, char)

        # finalize_streaming should emit correction
        correction = parser.finalize_streaming(accumulated)
        assert correction is not None
        assert correction.content == text

    def test_implicit_mode_still_works(self, parser):
        """Ensure fix doesn't break implicit mode (only </think> in output)."""
        parser.reset_state()

        tokens = ["thinking", " about ", "it", "</think>", "The answer."]
        accumulated = ""
        content_parts = []
        reasoning_parts = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result:
                if result.content:
                    content_parts.append(result.content)
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)

        assert "thinking about it" in "".join(reasoning_parts)
        assert "The answer." in "".join(content_parts)


class TestNewlinePreservation:
    """Test that newline-only chunks survive the streaming pipeline.

    Bug: `\n` chunks were being dropped by whitespace suppression,
    breaking markdown formatting (headings, bullet lists, code blocks).
    """

    @pytest.fixture
    def parser(self):
        return get_parser("qwen3")()

    def test_newline_chunks_in_content(self, parser):
        """Newlines in content stream should not be dropped."""
        parser.reset_state()

        # Simulate: <think>ok</think>Hello\n\n# Heading\n
        tokens = ["<think>", "ok", "</think>", "Hello", "\n", "\n", "# Heading", "\n"]
        accumulated = ""
        content_parts = []

        for token in tokens:
            prev = accumulated
            accumulated += token
            result = parser.extract_reasoning_streaming(prev, accumulated, token)
            if result and result.content is not None:
                content_parts.append(result.content)

        full = "".join(content_parts)
        # Newlines should be preserved
        assert "\n\n" in full, f"Double newline lost in streaming. Got: {full!r}"
        assert "# Heading" in full

    def test_newline_only_delta_not_dropped(self, parser):
        """A delta that is exactly '\n' should produce content, not be skipped."""
        parser.reset_state()

        # After think tags, a \n-only delta should be content
        prev = "<think>x</think>Hello"
        delta = "\n"
        curr = prev + delta

        # First process up to "Hello" so parser knows we're past </think>
        accumulated = ""
        for char in prev:
            p = accumulated
            accumulated += char
            parser.extract_reasoning_streaming(p, accumulated, char)

        # Now the \n delta
        result = parser.extract_reasoning_streaming(prev, curr, delta)
        assert result is not None, "Newline delta should not be None"
        assert result.content == "\n", f"Expected content='\\n', got {result!r}"


class TestDeepSeekNoTagComparison:
    """Verify DeepSeek-R1 already handles no-tag case correctly (for reference)."""

    @pytest.fixture
    def parser(self):
        return get_parser("deepseek_r1")()

    def test_no_tags_streaming_becomes_content(self, parser):
        """DeepSeek-R1 correctly switches to content after threshold."""
        parser.reset_state()

        text = "This is a regular response without any thinking tags. It should be content."
        accumulated = ""
        content_parts = []
        reasoning_parts = []

        for char in text:
            prev = accumulated
            accumulated += char
            result = parser.extract_reasoning_streaming(prev, accumulated, char)
            if result:
                if result.content:
                    content_parts.append(result.content)
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)

        full_content = "".join(content_parts)
        # DeepSeek-R1 has NO_TAG_CONTENT_THRESHOLD = 64, so after 64 chars
        # it starts treating as content
        assert len(full_content) > 0, "DeepSeek should have content for no-tag output"
