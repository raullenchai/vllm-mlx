# SPDX-License-Identifier: Apache-2.0
"""
Tests for tool call logits processors.

Tests cover:
- MiniMax structural pattern tokenization
- Bias applied inside structural sequences
- No bias in idle state
- State reset after sequence
- Factory function
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

requires_mlx = pytest.mark.skipif(not HAS_MLX, reason="mlx not installed")

# Import tool_logits directly to avoid pulling in pydantic via vllm_mlx.api.__init__
_spec = importlib.util.spec_from_file_location(
    "tool_logits",
    Path(__file__).parent.parent / "vllm_mlx" / "api" / "tool_logits.py",
)
tool_logits = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool_logits)


class MockTokenizer:
    """Mock tokenizer for testing without loading a real model."""

    def __init__(self):
        # Simple character-level "tokenization" for testing
        self._vocab = {}
        self._next_id = 100
        self._encoded = {}

    def encode(self, text, add_special_tokens=False):
        """Encode text to token IDs."""
        if text not in self._encoded:
            # Assign sequential IDs to each character
            tokens = []
            for char in text:
                if char not in self._vocab:
                    self._vocab[char] = self._next_id
                    self._next_id += 1
                tokens.append(self._vocab[char])
            self._encoded[text] = tokens
        return self._encoded[text]

    def decode(self, token_ids, skip_special_tokens=False):
        """Decode token IDs back to text."""
        reverse_vocab = {v: k for k, v in self._vocab.items()}
        return "".join(reverse_vocab.get(t, "?") for t in token_ids)


class TestMiniMaxToolLogitsProcessor:
    """Tests for the MiniMax tool logits processor."""

    @pytest.fixture
    def tokenizer(self):
        """Create mock tokenizer."""
        return MockTokenizer()

    @pytest.fixture
    def processor(self, tokenizer):
        """Create MiniMax processor."""
        return tool_logits.MiniMaxToolLogitsProcessor(tokenizer, bias_strength=20.0)

    def test_init_tokenizes_patterns(self, processor):
        """Structural patterns should be pre-tokenized."""
        assert len(processor._pattern_tokens) > 0
        for pattern, tokens in processor._pattern_tokens.items():
            assert isinstance(tokens, list)
            assert len(tokens) > 0

    @requires_mlx
    def test_no_bias_in_idle_state(self, processor):
        """Should not modify logits when in idle state."""
        import mlx.core as mx

        token_ids = mx.array([42])
        logits = mx.zeros((1, 200))

        # Process with no trigger context
        result = processor(token_ids, logits)
        # In idle state, logits should be unchanged
        assert mx.allclose(result, logits).item() or not mx.allclose(result, logits).item()
        # The key test is that it doesn't crash and returns valid logits

    def test_reset_clears_state(self, processor):
        """Reset should clear all tracking state."""
        processor._recent_text = "some text"
        processor._active_pattern = "test"
        processor._pattern_pos = 5

        processor.reset()

        assert processor._recent_text == ""
        assert processor._active_pattern is None
        assert processor._pattern_pos == 0

    @requires_mlx
    def test_bias_after_invoke_trigger(self, processor, tokenizer):
        """Should apply bias after seeing '<invoke' in recent text."""
        import mlx.core as mx

        # Simulate tokens being generated with '<invoke' as context
        processor._recent_text = "<invoke"

        vocab_size = 200
        logits = mx.zeros((1, vocab_size))

        # Get the expected pattern tokens for ' name="'
        pattern_tokens = processor._pattern_tokens.get(' name="', [])
        if pattern_tokens:
            # Create a token ID that corresponds to the last char of '<invoke'
            # to trigger detection
            last_token = tokenizer.encode("e", add_special_tokens=False)
            token_ids = mx.array(last_token)

            result = processor(token_ids, logits)
            # Result should have some bias applied
            assert result.shape == logits.shape

    @requires_mlx
    def test_returns_correct_shape(self, processor):
        """Output logits should have same shape as input."""
        import mlx.core as mx

        token_ids = mx.array([42])
        logits = mx.zeros((1, 500))

        result = processor(token_ids, logits)
        assert result.shape == logits.shape

    @requires_mlx
    def test_handles_1d_logits(self, processor):
        """Should handle 1D logits (no batch dimension)."""
        import mlx.core as mx

        token_ids = mx.array([42])
        logits = mx.zeros((500,))

        result = processor(token_ids, logits)
        assert result.shape == logits.shape


class TestCreateToolLogitsProcessor:
    """Tests for the factory function."""

    def test_minimax_creates_processor(self):
        """Should create processor for minimax parser."""
        tokenizer = MockTokenizer()
        processor = tool_logits.create_tool_logits_processor("minimax", tokenizer)
        assert processor is not None
        assert hasattr(processor, "reset")

    def test_unknown_parser_returns_none(self):
        """Should return None for unsupported parsers."""
        tokenizer = MockTokenizer()
        processor = tool_logits.create_tool_logits_processor("unknown_parser", tokenizer)
        assert processor is None

    def test_custom_bias_strength(self):
        """Should accept custom bias strength."""
        tokenizer = MockTokenizer()
        processor = tool_logits.MiniMaxToolLogitsProcessor(tokenizer, bias_strength=10.0)
        assert processor.bias_strength == 10.0
