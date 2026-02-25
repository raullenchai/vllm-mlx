# SPDX-License-Identifier: Apache-2.0
"""Tests for reasoning parsers (base, think_parser, deepseek_r1, gpt_oss)."""

import pytest

from vllm_mlx.reasoning.base import DeltaMessage, ReasoningParser
from vllm_mlx.reasoning.think_parser import BaseThinkingReasoningParser
from vllm_mlx.reasoning.deepseek_r1_parser import DeepSeekR1ReasoningParser
from vllm_mlx.reasoning.gpt_oss_parser import (
    GptOssReasoningParser,
    _extract_channel,
    _CHANNEL_RE,
    _STRUCTURAL_TOKENS,
)


# ---------------------------------------------------------------------------
# DeltaMessage
# ---------------------------------------------------------------------------

class TestDeltaMessage:

    def test_reasoning_only(self):
        dm = DeltaMessage(reasoning="thinking")
        assert dm.reasoning == "thinking"
        assert dm.content is None

    def test_content_only(self):
        dm = DeltaMessage(content="answer")
        assert dm.content == "answer"
        assert dm.reasoning is None

    def test_both(self):
        dm = DeltaMessage(reasoning="r", content="c")
        assert dm.reasoning == "r"
        assert dm.content == "c"

    def test_reasoning_content_alias(self):
        dm = DeltaMessage(reasoning="r")
        assert dm.reasoning_content == "r"

    def test_defaults(self):
        dm = DeltaMessage()
        assert dm.role is None
        assert dm.content is None
        assert dm.reasoning is None


# ---------------------------------------------------------------------------
# ReasoningParser (abstract base)
# ---------------------------------------------------------------------------

class TestReasoningParserBase:

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            ReasoningParser()

    def test_reset_state_noop(self):
        class Dummy(ReasoningParser):
            def extract_reasoning(self, model_output):
                return None, model_output
            def extract_reasoning_streaming(self, prev, curr, delta):
                return None
        d = Dummy()
        d.reset_state()  # should not raise

    def test_finalize_streaming_default_none(self):
        class Dummy(ReasoningParser):
            def extract_reasoning(self, model_output):
                return None, model_output
            def extract_reasoning_streaming(self, prev, curr, delta):
                return None
        d = Dummy()
        assert d.finalize_streaming("some text") is None


# ---------------------------------------------------------------------------
# BaseThinkingReasoningParser (via DeepSeek-R1 as concrete subclass)
# ---------------------------------------------------------------------------

class TestBaseThinkExtractReasoning:
    """Tests for extract_reasoning using DeepSeekR1ReasoningParser."""

    def setup_method(self):
        self.parser = DeepSeekR1ReasoningParser()

    def test_both_tags(self):
        text = "<think>step by step</think>The answer is 42."
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning == "step by step"
        assert content == "The answer is 42."

    def test_both_tags_empty_reasoning(self):
        text = "<think></think>Just content"
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning is None
        assert content == "Just content"

    def test_both_tags_empty_content(self):
        text = "<think>reasoning only</think>"
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning == "reasoning only"
        assert content is None

    def test_both_tags_whitespace_reasoning(self):
        text = "<think>   </think>content"
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning is None
        assert content == "content"

    def test_only_end_tag_implicit(self):
        text = "implicit reasoning</think>final answer"
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning == "implicit reasoning"
        assert content == "final answer"

    def test_only_start_tag(self):
        text = "<think>incomplete reasoning without close"
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning == "incomplete reasoning without close"
        assert content is None

    def test_no_tags_pure_content(self):
        text = "Just a simple response with no thinking."
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning is None
        assert content == text

    def test_multiline_reasoning(self):
        text = "<think>Line 1\nLine 2\nLine 3</think>Answer"
        reasoning, content = self.parser.extract_reasoning(text)
        assert "Line 1" in reasoning
        assert "Line 3" in reasoning
        assert content == "Answer"

    def test_multiple_think_tags_uses_first(self):
        text = "<think>first</think>middle<think>second</think>end"
        reasoning, content = self.parser.extract_reasoning(text)
        # partition finds first occurrence
        assert reasoning == "first"
        assert "middle" in content


# ---------------------------------------------------------------------------
# BaseThinkingReasoningParser streaming
# ---------------------------------------------------------------------------

class TestBaseThinkStreaming:

    def setup_method(self):
        self.parser = DeepSeekR1ReasoningParser()
        self.parser.reset_state()

    def test_skip_start_token(self):
        result = self.parser.extract_reasoning_streaming("", "<think>", "<think>")
        assert result is None

    def test_skip_end_token(self):
        result = self.parser.extract_reasoning_streaming(
            "<think>reasoning", "<think>reasoning</think>", "</think>"
        )
        assert result is None

    def test_reasoning_after_start(self):
        prev = "<think>"
        delta = "step 1"
        curr = prev + delta
        result = self.parser.extract_reasoning_streaming(prev, curr, delta)
        assert result.reasoning == "step 1"
        assert result.content is None

    def test_content_after_end(self):
        prev = "<think>reasoning</think>"
        delta = "content"
        curr = prev + delta
        result = self.parser.extract_reasoning_streaming(prev, curr, delta)
        assert result.content == "content"
        assert result.reasoning is None

    def test_transition_in_delta(self):
        prev = "<think>reasoning"
        delta = " more</think>content"
        curr = prev + delta
        result = self.parser.extract_reasoning_streaming(prev, curr, delta)
        assert result.reasoning == " more"
        assert result.content == "content"

    def test_both_tags_in_single_delta(self):
        prev = ""
        delta = "<think>reason</think>content"
        curr = delta
        result = self.parser.extract_reasoning_streaming(prev, curr, delta)
        assert result.reasoning == "reason"
        assert result.content == "content"

    def test_start_tag_only_in_delta(self):
        prev = ""
        delta = "<think>beginning"
        curr = delta
        result = self.parser.extract_reasoning_streaming(prev, curr, delta)
        assert result.reasoning == "beginning"

    def test_no_tags_early_defaults_to_reasoning(self):
        """Before any tags seen, base class defaults to reasoning."""
        prev = ""
        delta = "hello"
        curr = "hello"
        result = self.parser.extract_reasoning_streaming(prev, curr, delta)
        # DeepSeek has threshold logic, but under threshold defaults to reasoning
        assert result.reasoning == "hello" or result.content == "hello"

    def test_implicit_end_only(self):
        """Implicit mode: </think> without <think>."""
        prev = "some reasoning"
        delta = "</think>answer"
        curr = prev + delta
        result = self.parser.extract_reasoning_streaming(prev, curr, delta)
        # Should transition from reasoning to content
        assert result is not None

    def test_reset_state(self):
        self.parser._saw_any_tag = True
        self.parser.reset_state()
        assert self.parser._saw_any_tag is False


# ---------------------------------------------------------------------------
# DeepSeekR1ReasoningParser specifics
# ---------------------------------------------------------------------------

class TestDeepSeekR1:

    def setup_method(self):
        self.parser = DeepSeekR1ReasoningParser()

    def test_tokens(self):
        assert self.parser.start_token == "<think>"
        assert self.parser.end_token == "</think>"

    def test_no_tag_threshold_constant(self):
        assert self.parser.NO_TAG_CONTENT_THRESHOLD == 64

    def test_no_start_only_end(self):
        """DeepSeek-R1 handles implicit start tag."""
        text = "thinking about it</think>42"
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning == "thinking about it"
        assert content == "42"

    def test_no_tags_returns_content(self):
        text = "direct answer"
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning is None
        assert content == "direct answer"

    def test_standard_both_tags(self):
        text = "<think>r</think>c"
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning == "r"
        assert content == "c"

    def test_streaming_no_tag_past_threshold(self):
        """After threshold chars without tags, treat as content."""
        self.parser.reset_state()
        long_text = "x" * 100
        result = self.parser.extract_reasoning_streaming("", long_text, long_text)
        assert result.content == long_text

    def test_streaming_no_tag_under_threshold(self):
        """Under threshold without tags, delegates to base (reasoning)."""
        self.parser.reset_state()
        short = "hi"
        result = self.parser.extract_reasoning_streaming("", short, short)
        assert result.reasoning == short

    def test_finalize_short_no_tag_correction(self):
        """Short output without tags gets corrected from reasoning to content."""
        self.parser.reset_state()
        self.parser._saw_any_tag = False
        result = self.parser.finalize_streaming("short answer")
        assert result is not None
        assert result.content == "short answer"

    def test_finalize_long_no_tag_no_correction(self):
        """Long output without tags: no correction (already handled by threshold)."""
        self.parser.reset_state()
        self.parser._saw_any_tag = False
        result = self.parser.finalize_streaming("x" * 100)
        assert result is None

    def test_finalize_with_tags_no_correction(self):
        """Output with tags: no correction needed."""
        self.parser.reset_state()
        self.parser._saw_any_tag = True
        result = self.parser.finalize_streaming("<think>r</think>c")
        assert result is None

    def test_finalize_empty_no_correction(self):
        self.parser.reset_state()
        result = self.parser.finalize_streaming("")
        assert result is None


# ---------------------------------------------------------------------------
# GptOssReasoningParser
# ---------------------------------------------------------------------------

class TestGptOssHelpers:

    def test_extract_channel_analysis(self):
        text = "<|channel|>analysis<|message|>my reasoning<|start|>assistant"
        result = _extract_channel(text, "analysis")
        assert result == "my reasoning"

    def test_extract_channel_final(self):
        text = "<|channel|>final<|message|>the answer<|return|>"
        result = _extract_channel(text, "final")
        assert result == "the answer"

    def test_extract_channel_not_found(self):
        text = "<|channel|>analysis<|message|>reasoning"
        result = _extract_channel(text, "final")
        assert result is None

    def test_extract_channel_empty_content(self):
        text = "<|channel|>analysis<|message|><|start|>"
        result = _extract_channel(text, "analysis")
        assert result is None

    def test_extract_channel_with_constrain(self):
        text = "<|channel|>final <|constrain|>JSON<|message|>content here<|return|>"
        result = _extract_channel(text, "final")
        assert result == "content here"

    def test_channel_regex_matches_analysis(self):
        text = "<|channel|>analysis<|message|>"
        m = _CHANNEL_RE.search(text)
        assert m is not None
        assert m.group(1) == "analysis"

    def test_channel_regex_matches_final(self):
        text = "<|channel|>final<|message|>"
        m = _CHANNEL_RE.search(text)
        assert m is not None
        assert m.group(1) == "final"

    def test_channel_regex_matches_constrain(self):
        text = "<|channel|>final <|constrain|>JSON<|message|>"
        m = _CHANNEL_RE.search(text)
        assert m is not None
        assert m.group(1) == "final"

    def test_structural_tokens_regex(self):
        for tok in ["<|start|>", "<|end|>", "<|channel|>", "<|return|>", "<|call|>", "<|constrain|>"]:
            assert _STRUCTURAL_TOKENS.search(tok) is not None


class TestGptOssExtractReasoning:

    def setup_method(self):
        self.parser = GptOssReasoningParser()

    def test_full_format(self):
        text = (
            "<|channel|>analysis<|message|>Step by step reasoning"
            "<|start|>assistant<|channel|>final<|message|>The answer is 42<|return|>"
        )
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning == "Step by step reasoning"
        assert content == "The answer is 42"

    def test_analysis_only(self):
        text = "<|channel|>analysis<|message|>just reasoning"
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning == "just reasoning"
        assert content is None

    def test_final_only(self):
        text = "<|channel|>final<|message|>just content<|return|>"
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning is None
        assert content == "just content"

    def test_no_channels(self):
        text = "plain text without channels"
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning is None
        assert content == text

    def test_empty_input(self):
        reasoning, content = self.parser.extract_reasoning("")
        assert reasoning is None
        assert content is None

    def test_none_like_empty(self):
        reasoning, content = self.parser.extract_reasoning("")
        assert reasoning is None

    def test_constrain_format(self):
        text = (
            "<|channel|>analysis<|message|>thinking"
            "<|start|>assistant<|channel|>final <|constrain|>JSON<|message|>{\"key\": \"val\"}<|return|>"
        )
        reasoning, content = self.parser.extract_reasoning(text)
        assert reasoning == "thinking"
        assert content == '{"key": "val"}'

    def test_structural_tokens_stripped(self):
        text = (
            "<|channel|>analysis<|message|>reason<|start|>"
            "<|channel|>final<|message|>answer<|return|>"
        )
        reasoning, content = self.parser.extract_reasoning(text)
        assert "<|" not in (reasoning or "")
        assert "<|" not in (content or "")


class TestGptOssStreaming:

    def setup_method(self):
        self.parser = GptOssReasoningParser()

    def test_detect_phase_init(self):
        assert GptOssReasoningParser._detect_phase("") == "init"
        assert GptOssReasoningParser._detect_phase("random text") == "init"

    def test_detect_phase_analysis(self):
        text = "<|channel|>analysis<|message|>reasoning"
        assert GptOssReasoningParser._detect_phase(text) == "analysis"

    def test_detect_phase_final(self):
        text = "<|channel|>analysis<|message|>r<|start|>assistant<|channel|>final<|message|>c"
        assert GptOssReasoningParser._detect_phase(text) == "final"

    def test_detect_phase_transition(self):
        text = "<|channel|>analysis<|message|>reason<|start|>"
        assert GptOssReasoningParser._detect_phase(text) == "transition"

    def test_streaming_analysis_phase(self):
        prev = "<|channel|>analysis<|message|>part1"
        delta = " part2"
        curr = prev + delta
        result = self.parser.extract_reasoning_streaming(prev, curr, delta)
        assert result is not None
        assert result.reasoning == " part2"

    def test_streaming_final_phase(self):
        prev = "<|channel|>analysis<|message|>r<|start|>assistant<|channel|>final<|message|>part1"
        delta = " part2"
        curr = prev + delta
        result = self.parser.extract_reasoning_streaming(prev, curr, delta)
        assert result is not None
        assert result.content == " part2"

    def test_streaming_phase_transition_to_analysis(self):
        prev = ""
        delta = "<|channel|>analysis<|message|>reasoning start"
        curr = delta
        result = self.parser.extract_reasoning_streaming(prev, curr, delta)
        assert result is not None
        assert result.reasoning is not None
        assert "reasoning start" in result.reasoning

    def test_streaming_phase_transition_to_final(self):
        prev = "<|channel|>analysis<|message|>reason<|start|>assistant"
        delta = "<|channel|>final<|message|>content start"
        curr = prev + delta
        result = self.parser.extract_reasoning_streaming(prev, curr, delta)
        assert result is not None
        assert result.content is not None
        assert "content start" in result.content

    def test_streaming_init_phase_skips(self):
        prev = ""
        delta = "<|start|>"
        curr = delta
        result = self.parser.extract_reasoning_streaming(prev, curr, delta)
        assert result is None

    def test_streaming_structural_token_stripped(self):
        prev = "<|channel|>analysis<|message|>reasoning"
        delta = "<|start|>"
        curr = prev + delta
        result = self.parser.extract_reasoning_streaming(prev, curr, delta)
        # Phase transitions to "transition", delta is structural → skip
        assert result is None or (result and "<|start|>" not in (result.reasoning or ""))

    def test_strip_return(self):
        assert GptOssReasoningParser._strip_return("text<|return|>") == "text"
        assert GptOssReasoningParser._strip_return("no return") == "no return"

    def test_extract_content_after_marker(self):
        text = "<|channel|>analysis<|message|>the content"
        result = GptOssReasoningParser._extract_content_after_marker_in_delta(text, "analysis")
        assert result == "the content"

    def test_extract_content_after_marker_not_found(self):
        text = "<|channel|>analysis<|message|>content"
        result = GptOssReasoningParser._extract_content_after_marker_in_delta(text, "final")
        assert result is None


# ---------------------------------------------------------------------------
# Full streaming simulation tests
# ---------------------------------------------------------------------------

class TestFullStreamingSimulation:
    """Simulate realistic streaming token-by-token delivery."""

    def test_think_parser_full_stream(self):
        """Simulate: <think>step 1\nstep 2</think>The answer."""
        parser = DeepSeekR1ReasoningParser()
        parser.reset_state()

        chunks = ["<think>", "step ", "1\n", "step 2", "</think>", "The ", "answer."]
        accumulated = ""
        reasoning_parts = []
        content_parts = []

        for chunk in chunks:
            prev = accumulated
            accumulated += chunk
            result = parser.extract_reasoning_streaming(prev, accumulated, chunk)
            if result:
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)
                if result.content:
                    content_parts.append(result.content)

        assert "".join(reasoning_parts) == "step 1\nstep 2"
        assert "".join(content_parts) == "The answer."

    def test_deepseek_implicit_stream(self):
        """Simulate implicit mode: reasoning</think>content (no <think>)."""
        parser = DeepSeekR1ReasoningParser()
        parser.reset_state()

        chunks = ["reas", "oning", "</think>", "content"]
        accumulated = ""
        reasoning_parts = []
        content_parts = []

        for chunk in chunks:
            prev = accumulated
            accumulated += chunk
            result = parser.extract_reasoning_streaming(prev, accumulated, chunk)
            if result:
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)
                if result.content:
                    content_parts.append(result.content)

        assert "reas" in "".join(reasoning_parts)
        assert "content" in "".join(content_parts)

    def test_gpt_oss_full_stream(self):
        """Simulate GPT-OSS channel-based streaming."""
        parser = GptOssReasoningParser()

        chunks = [
            "<|channel|>analysis<|message|>",
            "reasoning ",
            "here",
            "<|start|>",
            "assistant",
            "<|channel|>final<|message|>",
            "the ",
            "answer",
            "<|return|>",
        ]
        accumulated = ""
        reasoning_parts = []
        content_parts = []

        for chunk in chunks:
            prev = accumulated
            accumulated += chunk
            result = parser.extract_reasoning_streaming(prev, accumulated, chunk)
            if result:
                if result.reasoning:
                    reasoning_parts.append(result.reasoning)
                if result.content:
                    content_parts.append(result.content)

        reasoning_text = "".join(reasoning_parts)
        content_text = "".join(content_parts)
        assert "reasoning" in reasoning_text
        assert "answer" in content_text
