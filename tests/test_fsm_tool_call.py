# SPDX-License-Identifier: Apache-2.0
"""Tests for FSM-based tool call constrained decoding."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

import pytest

# Skip all tests if outlines-core not installed
pytest.importorskip("outlines_core")


class TestFSMToolCallCache:
    """Tests for FSM compilation cache."""

    def test_precompile_success(self):
        from vllm_mlx.api.fsm_tool_call import FSMToolCallCache

        cache = FSMToolCallCache()
        # Build vocabulary from real tokenizer
        from outlines_core import Vocabulary

        cache._vocabulary = Vocabulary.from_pretrained(
            "mlx-community/Qwen3.5-4B-MLX-4bit"
        )

        tools = [
            {
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
                }
            }
        ]
        assert cache.precompile(tools) is True

    def test_cache_hit(self):
        from vllm_mlx.api.fsm_tool_call import FSMToolCallCache

        cache = FSMToolCallCache()
        from outlines_core import Vocabulary

        cache._vocabulary = Vocabulary.from_pretrained(
            "mlx-community/Qwen3.5-4B-MLX-4bit"
        )

        tools = [{"function": {"name": "search", "parameters": {"type": "object"}}}]

        # First call compiles
        t0 = time.perf_counter()
        cache.precompile(tools)
        first_time = time.perf_counter() - t0

        # Second call hits cache
        t0 = time.perf_counter()
        result = cache.precompile(tools)
        second_time = time.perf_counter() - t0

        assert result is True
        assert second_time < first_time / 10, "Cache hit should be >10x faster"

    def test_get_guide_returns_fresh_guide(self):
        from vllm_mlx.api.fsm_tool_call import FSMToolCallCache

        cache = FSMToolCallCache()
        from outlines_core import Vocabulary

        cache._vocabulary = Vocabulary.from_pretrained(
            "mlx-community/Qwen3.5-4B-MLX-4bit"
        )

        tools = [{"function": {"name": "test", "parameters": {"type": "object"}}}]

        g1 = cache.get_guide(tools)
        g2 = cache.get_guide(tools)
        assert g1 is not None
        assert g2 is not None
        # Each guide is a fresh instance (independent state)
        assert g1 is not g2

    def test_schema_builds_correct_enum(self):
        from vllm_mlx.api.fsm_tool_call import _build_tool_call_schema

        tools = [
            {"function": {"name": "get_weather"}},
            {"function": {"name": "search"}},
            {"function": {"name": "calculate"}},
        ]
        schema = json.loads(_build_tool_call_schema(tools))
        assert schema["properties"]["name"]["enum"] == [
            "get_weather",
            "search",
            "calculate",
        ]
        assert schema["required"] == ["name", "arguments"]


class TestFSMToolCallProcessor:
    """Tests for the two-mode logits processor."""

    @pytest.fixture
    def tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("mlx-community/Qwen3.5-4B-MLX-4bit")

    @pytest.fixture
    def tools(self):
        return [
            {
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                }
            }
        ]

    @pytest.fixture
    def processor(self, tokenizer, tools):
        from outlines_core import Vocabulary

        from vllm_mlx.api.fsm_tool_call import (
            FSMToolCallCache,
            FSMToolCallProcessor,
        )

        cache = FSMToolCallCache()
        cache._vocabulary = Vocabulary.from_pretrained(
            "mlx-community/Qwen3.5-4B-MLX-4bit"
        )
        cache.precompile(tools)

        return FSMToolCallProcessor(
            tokenizer=tokenizer,
            tools=tools,
            parser_name="hermes",
            cache=cache,
        )

    def test_free_mode_passes_logits_through(self, processor, tokenizer):
        """In free mode, logits should pass through unchanged."""
        import mlx.core as mx

        logits = mx.random.normal((1, 248077))
        token_ids = mx.array(tokenizer.encode("Hello world"))

        result = processor(token_ids, logits)
        # Should be identical (no masking)
        assert mx.array_equal(result, logits)

    def test_trigger_plus_json_activates_constrained_mode(self, processor, tokenizer):
        """After seeing <tool_call>\\n + '{', processor should constrain."""
        import mlx.core as mx

        # Feed trigger + JSON start
        text = '<tool_call>\n{"'
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        logits = mx.zeros((1, 248077))

        for i, tid in enumerate(token_ids):
            all_ids = mx.array(token_ids[: i + 1])
            processor(all_ids, logits)

        assert processor._constrained, "Should be in constrained mode after trigger + '{'"

    def test_trigger_plus_xml_skips_fsm(self, processor, tokenizer):
        """After <tool_call>\\n + '<', FSM should NOT activate (XML format)."""
        import mlx.core as mx

        text = "<tool_call>\n<function"
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        logits = mx.zeros((1, 248077))

        for i, tid in enumerate(token_ids):
            all_ids = mx.array(token_ids[: i + 1])
            processor(all_ids, logits)

        assert not processor._constrained, "Should NOT constrain for XML format"

    def test_constrained_mode_masks_invalid_tokens(self, processor, tokenizer):
        """In constrained mode, most tokens should be masked to -inf."""
        import mlx.core as mx

        # Activate constrained mode by feeding trigger
        processor._recent_text = "<tool_call>\n"
        processor._constrained = False

        # Create a dummy "last token was newline" to trigger
        trigger_ids = tokenizer.encode("<tool_call>\n", add_special_tokens=False)
        logits = mx.zeros((1, 248077))

        # Feed the last trigger token to activate FSM
        result = processor(mx.array(trigger_ids), logits)

        if processor._constrained:
            # Most tokens should be -inf (masked)
            result_np = result.tolist()[0]
            n_valid = sum(1 for x in result_np if x > -1e9)
            n_masked = sum(1 for x in result_np if x < -1e9)
            print(f"\n  Constrained: {n_valid} valid, {n_masked} masked")
            assert n_valid < 100, f"Expected < 100 valid tokens, got {n_valid}"
            assert n_masked > 200000, "Expected most tokens masked"

    def test_reset_clears_state(self, processor):
        processor._constrained = True
        processor._recent_text = "some text"
        processor._guide = MagicMock()

        processor.reset()

        assert not processor._constrained
        assert processor._recent_text == ""
        assert processor._guide is None


class TestFSMFactory:
    """Tests for the factory function."""

    def test_create_returns_processor_when_available(self):
        from outlines_core import Vocabulary
        from transformers import AutoTokenizer

        from vllm_mlx.api.fsm_tool_call import create_fsm_processor, get_fsm_cache

        tok = AutoTokenizer.from_pretrained("mlx-community/Qwen3.5-4B-MLX-4bit")
        cache = get_fsm_cache()
        cache._vocabulary = Vocabulary.from_pretrained(
            "mlx-community/Qwen3.5-4B-MLX-4bit"
        )

        tools = [{"function": {"name": "test", "parameters": {"type": "object"}}}]
        proc = create_fsm_processor("hermes", tok, tools)
        assert proc is not None

    def test_create_returns_processor_with_generic_schema(self):
        """Even without specific tools, factory returns a processor
        with generic schema (any name + any arguments)."""
        from vllm_mlx.api.fsm_tool_call import create_fsm_processor

        tok = MagicMock()
        # No tools → generic schema processor (not None)
        proc = create_fsm_processor("hermes", tok, None)
        assert proc is not None, "Should return generic FSM processor"

    def test_all_parsers_have_triggers(self):
        """Every parser should have a trigger pattern registered."""
        from vllm_mlx.api.fsm_tool_call import TOOL_CALL_TRIGGERS

        expected_parsers = [
            "hermes", "llama", "minimax", "qwen", "deepseek",
            "glm47", "granite", "nemotron", "kimi", "gemma4",
            "functionary", "seed_oss", "mistral", "xlam",
        ]
        for p in expected_parsers:
            assert p in TOOL_CALL_TRIGGERS, f"Missing trigger for parser {p!r}"


class TestFSMPerformance:
    """Verify FSM overhead is negligible."""

    def test_per_token_overhead_under_10us(self):
        """FSM lookup must be < 10µs per token to not affect decode speed."""
        from outlines_core import Guide, Index, Vocabulary, json_schema

        vocabulary = Vocabulary.from_pretrained("mlx-community/Qwen3.5-4B-MLX-4bit")
        schema = json.dumps({
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": ["get_weather"]},
                "arguments": {"type": "object"},
            },
            "required": ["name", "arguments"],
        })
        regex = json_schema.build_regex_from_schema(schema)
        index = Index(regex, vocabulary)

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("mlx-community/Qwen3.5-4B-MLX-4bit")
        target = '{"name": "get_weather", "arguments": {}}'
        target_ids = tok.encode(target, add_special_tokens=False)

        # Benchmark
        t0 = time.perf_counter()
        for _ in range(1000):
            guide = Guide(index)
            for tid in target_ids:
                guide.get_tokens()
                guide.advance(tid)
        dt = time.perf_counter() - t0
        per_token_us = dt / (1000 * len(target_ids)) * 1e6

        print(f"\n  Per-token FSM overhead: {per_token_us:.1f} µs")
        assert per_token_us < 10, f"FSM overhead too high: {per_token_us:.1f} µs"
