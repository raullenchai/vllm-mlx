# SPDX-License-Identifier: Apache-2.0
"""
End-to-end streaming simulator that reproduces the server's SSE pipeline.

Simulates the full flow: model tokens → reasoning parser → content filtering
→ SSE emission → client-side reassembly.

Tests three scenarios matching real-world usage:
1. Normal Qwen3 with <think> tags (should work)
2. OpenCode implicit think mode (<think> in prompt, only </think> in output)
3. No-tag model (8-bit quantized, never emits <think> tags)
4. Markdown with newlines (the original bug report)

Each test checks both the streaming content AND the final assembled output.
"""

import pytest

from vllm_mlx.reasoning import get_parser


def simulate_server_streaming(tokens: list[str], use_reasoning_parser: str = "qwen3"):
    """
    Simulate the server's streaming pipeline from server.py.

    This mirrors the actual logic in stream_chat_completion():
    - Reasoning parser extracts content/reasoning from each delta
    - Reasoning-only chunks are DROPPED (line 2615: `if not content: continue`)
    - finalize_streaming correction is emitted at the end
    - Empty-string content is filtered (line 2729)

    Returns:
        list of SSE content strings the client would receive
    """
    parser = get_parser(use_reasoning_parser)()
    parser.reset_state()

    accumulated_text = ""
    sse_chunks = []  # What the client receives as content

    for i, token in enumerate(tokens):
        delta_text = token
        is_finished = (i == len(tokens) - 1)

        previous_text = accumulated_text
        accumulated_text += delta_text

        delta_msg = parser.extract_reasoning_streaming(
            previous_text, accumulated_text, delta_text
        )

        if delta_msg is None:
            continue

        content = delta_msg.content
        reasoning = delta_msg.reasoning

        # Skip empty-string content (server.py line 2729)
        if content is not None and content == "":
            content = None

        # Server line 2615: skip if no content and not finished
        finish_reason = "stop" if is_finished else None
        if not content and not finish_reason:
            continue

        if content:
            sse_chunks.append(content)

    # Server line 2761: finalize_streaming correction
    if hasattr(parser, 'finalize_streaming'):
        correction = parser.finalize_streaming(accumulated_text)
        if correction and correction.content:
            sse_chunks.append(correction.content)

    return sse_chunks


class TestScenario1_ExplicitThinkTags:
    """Normal Qwen3 usage with <think>reasoning</think>content."""

    def test_basic_think_then_content(self):
        """Standard flow: think tags → reasoning extracted, content streamed."""
        tokens = ["<think>", "Let me ", "analyze.", "</think>", "The ", "answer ", "is 42."]
        chunks = simulate_server_streaming(tokens)
        full = "".join(chunks)
        assert "The answer is 42." in full
        assert "<think>" not in full
        assert "analyze" not in full  # reasoning should NOT be in content

    def test_multiline_reasoning_then_content(self):
        tokens = [
            "<think>", "Step 1\n", "Step 2\n", "Step 3\n", "</think>",
            "# Result\n", "\n", "The answer is **42**.\n",
        ]
        chunks = simulate_server_streaming(tokens)
        full = "".join(chunks)
        assert "# Result" in full
        assert "Step 1" not in full  # reasoning excluded
        assert "\n" in full  # newlines preserved


class TestScenario2_ImplicitThinkMode:
    """OpenCode injects <think> in prompt. Only </think> appears in output.

    This is the most critical scenario — OpenCode/Cursor/Continue.dev all do this.
    The model output starts with reasoning text (no <think>), then </think>, then content.
    """

    def test_short_implicit_reasoning(self):
        """Short reasoning (< 64 chars) before </think>."""
        tokens = ["Let ", "me ", "think.", "</think>", "Answer: ", "42"]
        chunks = simulate_server_streaming(tokens)
        full = "".join(chunks)
        assert "Answer: 42" in full
        assert "Let me think" not in full  # reasoning excluded from content

    def test_long_implicit_reasoning(self):
        """Long reasoning (> 64 chars) before </think>.

        This is the critical regression test — the NO_TAG_CONTENT_THRESHOLD
        must NOT kick in and start routing reasoning as content.
        """
        # Generate reasoning > 64 chars
        reasoning_tokens = [
            "Let me think about this problem step by step.\n",  # 47 chars
            "First, I need to consider the constraints.\n",     # 44 chars (total: 91)
            "Then apply the algorithm.\n",                       # 27 chars (total: 118)
            "Finally verify the result.\n",                      # 28 chars (total: 146)
        ]
        content_tokens = ["The ", "answer ", "is ", "42."]

        tokens = reasoning_tokens + ["</think>"] + content_tokens
        chunks = simulate_server_streaming(tokens)
        full = "".join(chunks)

        # Content should have ONLY the actual content
        assert "The answer is 42." in full
        # Reasoning should NOT leak into content
        assert "Let me think" not in full, f"Reasoning leaked into content: {full!r}"
        assert "constraints" not in full, f"Reasoning leaked into content: {full!r}"
        assert "algorithm" not in full, f"Reasoning leaked into content: {full!r}"

    def test_very_long_implicit_reasoning(self):
        """Very long reasoning (> 500 chars) before </think>."""
        reasoning = "Analyzing step by step. " * 30  # ~720 chars
        # Break into tokens
        reasoning_tokens = [reasoning[i:i+20] for i in range(0, len(reasoning), 20)]
        content_tokens = ["Here is the answer."]

        tokens = reasoning_tokens + ["</think>"] + content_tokens
        chunks = simulate_server_streaming(tokens)
        full = "".join(chunks)

        assert "Here is the answer." in full
        assert "Analyzing" not in full, f"Reasoning leaked: {full!r}"


class TestScenario3_NoTagModel:
    """Model never emits <think> tags (e.g., 8-bit quantized Qwen3).

    This is the original user-reported bug: content is empty because
    the reasoning parser classifies everything as reasoning.
    """

    def test_short_no_tag_output(self):
        """Short output (< 64 chars) with no tags at all."""
        tokens = ["Hello ", "world!"]
        chunks = simulate_server_streaming(tokens)
        full = "".join(chunks)
        assert full == "Hello world!", f"Expected 'Hello world!', got {full!r}"

    def test_long_no_tag_output(self):
        """Long output (> 64 chars) with no tags — the core bug."""
        text = "Here is a markdown example:\n\n# Heading\n\n- Item 1\n- Item 2\n\nDone."
        tokens = [text[i:i+5] for i in range(0, len(text), 5)]
        chunks = simulate_server_streaming(tokens)
        full = "".join(chunks)

        assert "# Heading" in full, f"Content missing: {full!r}"
        assert "- Item 1" in full, f"Content missing: {full!r}"
        assert "Done." in full, f"Content missing: {full!r}"
        # ALL text should be in content, nothing lost
        assert full == text, f"Content mismatch:\n  Expected: {text!r}\n  Got:      {full!r}"

    def test_very_long_no_tag_output(self):
        """Very long output with no tags — no chars should be lost."""
        text = "The quick brown fox. " * 20  # 420 chars
        tokens = [text[i:i+10] for i in range(0, len(text), 10)]
        chunks = simulate_server_streaming(tokens)
        full = "".join(chunks)
        assert full == text, f"Length mismatch: expected {len(text)}, got {len(full)}"

    def test_no_tag_with_newlines(self):
        """No-tag output with markdown newlines — the original Reddit bug."""
        tokens = [
            "# Title", "\n", "\n",
            "Some text.", "\n", "\n",
            "- bullet 1", "\n",
            "- bullet 2", "\n", "\n",
            "```python", "\n",
            "print('hello')", "\n",
            "```", "\n",
        ]
        chunks = simulate_server_streaming(tokens)
        full = "".join(chunks)

        assert "# Title\n\n" in full, f"Heading newlines lost: {full!r}"
        assert "- bullet 1\n- bullet 2" in full, f"Bullets lost: {full!r}"
        assert "```python\nprint('hello')\n```" in full, f"Code block lost: {full!r}"

    def test_no_tag_emojis(self):
        """Emoji characters should pass through correctly."""
        tokens = ["Hello ", "🎉", " ", "🚀", " world!"]
        chunks = simulate_server_streaming(tokens)
        full = "".join(chunks)
        assert "🎉" in full
        assert "🚀" in full


class TestScenario4_NewlinePreservation:
    """Test that newline-only chunks survive the pipeline.

    Original bug: \n chunks were dropped by whitespace suppression.
    """

    def test_newline_between_paragraphs(self):
        """Double newline between paragraphs must be preserved."""
        tokens = ["<think>", "ok", "</think>", "Para 1.", "\n", "\n", "Para 2."]
        chunks = simulate_server_streaming(tokens)
        full = "".join(chunks)
        assert "Para 1.\n\nPara 2." in full, f"Newlines lost: {full!r}"

    def test_newline_in_markdown_list(self):
        tokens = ["<think>", "ok", "</think>", "- a", "\n", "- b", "\n", "- c"]
        chunks = simulate_server_streaming(tokens)
        full = "".join(chunks)
        assert "- a\n- b\n- c" in full

    def test_newline_in_code_block(self):
        tokens = [
            "<think>", "ok", "</think>",
            "```", "\n", "line1", "\n", "line2", "\n", "```",
        ]
        chunks = simulate_server_streaming(tokens)
        full = "".join(chunks)
        assert "```\nline1\nline2\n```" in full


class TestScenario5_EdgeCases:
    """Edge cases and mixed scenarios."""

    def test_empty_think_tags(self):
        """<think></think>content — empty reasoning."""
        tokens = ["<think>", "</think>", "Just content."]
        chunks = simulate_server_streaming(tokens)
        full = "".join(chunks)
        assert full == "Just content."

    def test_deepseek_no_tag_threshold(self):
        """DeepSeek-R1 should also handle no-tag output correctly."""
        text = "A regular response without thinking tags, should be content."
        tokens = [text[i:i+5] for i in range(0, len(text), 5)]
        chunks = simulate_server_streaming(tokens, use_reasoning_parser="deepseek_r1")
        full = "".join(chunks)
        assert full == text, f"DeepSeek no-tag mismatch: {full!r}"

    def test_single_char_no_tag(self):
        """Single character output, no tags."""
        chunks = simulate_server_streaming(["Y"], use_reasoning_parser="qwen3")
        full = "".join(chunks)
        assert full == "Y"

    def test_whitespace_only_not_emitted_as_content(self):
        """Pure empty string should be filtered, but whitespace should pass."""
        tokens = ["<think>", "ok", "</think>", "a", "\n", "\t", "b"]
        chunks = simulate_server_streaming(tokens)
        full = "".join(chunks)
        assert "a\n\tb" in full
