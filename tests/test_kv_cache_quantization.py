# SPDX-License-Identifier: Apache-2.0
"""Tests for KV cache quantization in prefix cache."""

import mlx.core as mx
from mlx_lm.models.cache import KVCache, QuantizedKVCache

from vllm_mlx.memory_cache import (
    MemoryAwarePrefixCache,
    MemoryCacheConfig,
    _dequantize_cache,
    _quantize_cache,
    _trim_to_offset,
    estimate_kv_cache_memory,
)


def _make_kv_cache(
    n_layers: int = 4, seq_len: int = 100, n_heads: int = 8, head_dim: int = 64
):
    """Create a list of KVCache layers with random data."""
    cache = []
    for _ in range(n_layers):
        kv = KVCache()
        kv.keys = mx.random.normal((1, n_heads, seq_len, head_dim))
        kv.values = mx.random.normal((1, n_heads, seq_len, head_dim))
        kv.offset = seq_len
        cache.append(kv)
    mx.eval(*[kv.keys for kv in cache], *[kv.values for kv in cache])
    return cache


class TestQuantizeDequantize:
    """Test the quantize/dequantize helper functions."""

    def test_quantize_produces_quantized_cache(self):
        cache = _make_kv_cache()
        quantized = _quantize_cache(cache, bits=8, group_size=64)
        assert len(quantized) == len(cache)
        for layer in quantized:
            assert isinstance(layer, QuantizedKVCache)

    def test_dequantize_produces_kv_cache(self):
        cache = _make_kv_cache()
        quantized = _quantize_cache(cache, bits=8, group_size=64)
        restored = _dequantize_cache(quantized)
        assert len(restored) == len(cache)
        for layer in restored:
            assert isinstance(layer, KVCache)

    def test_round_trip_preserves_shapes(self):
        cache = _make_kv_cache(n_layers=4, seq_len=128, n_heads=8, head_dim=64)
        quantized = _quantize_cache(cache, bits=8, group_size=64)
        restored = _dequantize_cache(quantized)
        for orig, rest in zip(cache, restored):
            assert rest.keys.shape == orig.keys.shape
            assert rest.values.shape == orig.values.shape

    def test_round_trip_preserves_offset(self):
        cache = _make_kv_cache(seq_len=200)
        quantized = _quantize_cache(cache, bits=8, group_size=64)
        restored = _dequantize_cache(quantized)
        for orig, rest in zip(cache, restored):
            assert rest.offset == orig.offset

    def test_round_trip_values_close(self):
        cache = _make_kv_cache()
        quantized = _quantize_cache(cache, bits=8, group_size=64)
        restored = _dequantize_cache(quantized)
        for orig, rest in zip(cache, restored):
            mx.eval(orig.keys, rest.keys)
            diff = mx.abs(orig.keys - rest.keys).mean().item()
            assert diff < 0.05, f"Mean abs error too high: {diff}"

    def test_4bit_quantization(self):
        cache = _make_kv_cache()
        quantized = _quantize_cache(cache, bits=4, group_size=64)
        restored = _dequantize_cache(quantized)
        for orig, rest in zip(cache, restored):
            assert rest.keys.shape == orig.keys.shape

    def test_empty_cache_passthrough(self):
        assert _quantize_cache([], bits=8, group_size=64) == []
        assert _dequantize_cache([]) == []

    def test_none_keys_passthrough(self):
        """KVCache with None keys should not be quantized."""
        kv = KVCache()  # keys=None, values=None
        result = _quantize_cache([kv], bits=8, group_size=64)
        assert isinstance(result[0], KVCache)
        assert result[0].keys is None


class TestMixedCacheLayers:
    """Test that non-KVCache layers are preserved."""

    def test_non_kvcache_layers_preserved(self):
        """Simulate a hybrid cache with dict layers (like MambaCache state)."""
        kv = KVCache()
        kv.keys = mx.random.normal((1, 4, 50, 64))
        kv.values = mx.random.normal((1, 4, 50, 64))
        kv.offset = 50
        mx.eval(kv.keys, kv.values)

        # Simulate a non-KVCache layer (dict-based state)
        fake_mamba = {"state": mx.zeros((1, 16, 64)), "type": "mamba"}

        cache = [kv, fake_mamba]
        quantized = _quantize_cache(cache, bits=8, group_size=64)

        assert isinstance(quantized[0], QuantizedKVCache)
        assert isinstance(quantized[1], dict)  # Preserved as-is

        restored = _dequantize_cache(quantized)
        assert isinstance(restored[0], KVCache)
        assert isinstance(restored[1], dict)


class TestMemoryReduction:
    """Test that quantization reduces memory estimates."""

    def test_quantized_uses_less_memory(self):
        cache = _make_kv_cache(n_layers=8, seq_len=512, n_heads=32, head_dim=128)
        original_mem = estimate_kv_cache_memory(cache)

        quantized = _quantize_cache(cache, bits=8, group_size=64)
        quantized_mem = estimate_kv_cache_memory(quantized)

        assert quantized_mem < original_mem
        ratio = original_mem / quantized_mem
        assert ratio > 2.0, f"Expected >2x reduction, got {ratio:.2f}x"

    def test_4bit_uses_less_than_8bit(self):
        cache = _make_kv_cache(n_layers=4, seq_len=256)
        q8 = _quantize_cache(cache, bits=8, group_size=64)
        q4 = _quantize_cache(cache, bits=4, group_size=64)
        mem_8 = estimate_kv_cache_memory(q8)
        mem_4 = estimate_kv_cache_memory(q4)
        assert mem_4 < mem_8


class TestMemoryCacheConfig:
    """Test config fields for quantization."""

    def test_default_quantization_off(self):
        config = MemoryCacheConfig()
        assert config.kv_quantize is False
        assert config.kv_bits == 8
        assert config.kv_group_size == 64

    def test_config_with_quantization(self):
        config = MemoryCacheConfig(kv_quantize=True, kv_bits=4, kv_group_size=32)
        assert config.kv_quantize is True
        assert config.kv_bits == 4
        assert config.kv_group_size == 32


class TestPrefixCacheIntegration:
    """Test store/fetch with quantization enabled and disabled."""

    def _make_cache_and_model(self):
        class FakeModel:
            pass

        return FakeModel()

    def test_store_fetch_without_quantization(self):
        model = self._make_cache_and_model()
        config = MemoryCacheConfig(kv_quantize=False, max_memory_mb=500)
        pc = MemoryAwarePrefixCache(model, config)

        cache = _make_kv_cache(n_layers=2, seq_len=50)
        tokens = list(range(50))

        pc.store(tokens, cache)
        fetched, remaining = pc.fetch(tokens)

        assert fetched is not None
        assert remaining == []
        for layer in fetched:
            assert isinstance(layer, KVCache)

    def test_store_fetch_with_quantization(self):
        model = self._make_cache_and_model()
        config = MemoryCacheConfig(
            kv_quantize=True,
            kv_bits=8,
            kv_min_quantize_tokens=0,
            max_memory_mb=500,
        )
        pc = MemoryAwarePrefixCache(model, config)

        cache = _make_kv_cache(n_layers=2, seq_len=50)
        tokens = list(range(50))

        pc.store(tokens, cache)
        # Internally stored as quantized
        stored_entry = list(pc._entries.values())[0]
        for layer in stored_entry.cache:
            assert isinstance(layer, QuantizedKVCache)

        # Fetched as dequantized KVCache
        fetched, remaining = pc.fetch(tokens)
        assert fetched is not None
        assert remaining == []
        for layer in fetched:
            assert isinstance(layer, KVCache)

    def test_quantized_store_reduces_tracked_memory(self):
        model = self._make_cache_and_model()

        config_fp16 = MemoryCacheConfig(kv_quantize=False, max_memory_mb=500)
        pc_fp16 = MemoryAwarePrefixCache(model, config_fp16)

        config_q8 = MemoryCacheConfig(kv_quantize=True, kv_bits=8, max_memory_mb=500)
        pc_q8 = MemoryAwarePrefixCache(model, config_q8)

        cache = _make_kv_cache(n_layers=4, seq_len=256)
        tokens = list(range(256))

        pc_fp16.store(tokens, cache)
        pc_q8.store(tokens, cache)

        assert pc_q8._current_memory < pc_fp16._current_memory


class TestTrimToOffset:
    """Test the _trim_to_offset helper function."""

    def test_trim_oversized_arrays(self):
        """KV arrays larger than offset should be trimmed."""
        cache = []
        for _ in range(2):
            kv = KVCache()
            kv.keys = mx.random.normal((1, 8, 4096, 64))
            kv.values = mx.random.normal((1, 8, 4096, 64))
            kv.offset = 512
            cache.append(kv)
        mx.eval(*[kv.keys for kv in cache], *[kv.values for kv in cache])

        trimmed = _trim_to_offset(cache)
        for layer in trimmed:
            assert layer.keys.shape[2] == 512
            assert layer.values.shape[2] == 512
            assert layer.offset == 512

    def test_no_trim_when_exact(self):
        """No trimming needed when arrays match offset."""
        cache = _make_kv_cache(n_layers=2, seq_len=100)
        trimmed = _trim_to_offset(cache)
        for orig, tr in zip(cache, trimmed):
            assert tr.keys.shape == orig.keys.shape
            assert tr.values.shape == orig.values.shape

    def test_non_kvcache_layers_preserved(self):
        """Non-KVCache layers pass through unchanged."""
        fake_layer = {"state": mx.zeros((1, 16, 64)), "type": "mamba"}
        result = _trim_to_offset([fake_layer])
        assert result[0] is fake_layer

    def test_none_keys_passthrough(self):
        """KVCache with None keys should pass through."""
        kv = KVCache()
        result = _trim_to_offset([kv])
        assert result[0] is kv


class TestMinQuantizeTokensThreshold:
    """Test that short sequences skip quantization."""

    def _make_model(self):
        class FakeModel:
            pass

        return FakeModel()

    def test_store_skips_quantization_below_threshold(self):
        """Sequences shorter than min_quantize_tokens should not be quantized."""
        model = self._make_model()
        config = MemoryCacheConfig(
            kv_quantize=True,
            kv_bits=8,
            kv_min_quantize_tokens=256,
            max_memory_mb=500,
        )
        pc = MemoryAwarePrefixCache(model, config)

        cache = _make_kv_cache(n_layers=2, seq_len=50)
        tokens = list(range(50))
        pc.store(tokens, cache)

        stored_entry = list(pc._entries.values())[0]
        for layer in stored_entry.cache:
            assert isinstance(layer, KVCache), (
                "Short sequences should remain as KVCache (not quantized)"
            )

    def test_store_quantizes_above_threshold(self):
        """Sequences >= min_quantize_tokens should be quantized."""
        model = self._make_model()
        config = MemoryCacheConfig(
            kv_quantize=True,
            kv_bits=8,
            kv_min_quantize_tokens=256,
            max_memory_mb=500,
        )
        pc = MemoryAwarePrefixCache(model, config)

        cache = _make_kv_cache(n_layers=2, seq_len=300)
        tokens = list(range(300))
        pc.store(tokens, cache)

        stored_entry = list(pc._entries.values())[0]
        for layer in stored_entry.cache:
            assert isinstance(layer, QuantizedKVCache), (
                "Long sequences should be quantized"
            )

    def test_trim_applied_without_quantization(self):
        """Oversized arrays should be trimmed even without quantization."""
        model = self._make_model()
        config = MemoryCacheConfig(kv_quantize=False, max_memory_mb=500)
        pc = MemoryAwarePrefixCache(model, config)

        # Create oversized cache: arrays have 4096 but offset is 100
        cache = []
        for _ in range(2):
            kv = KVCache()
            kv.keys = mx.random.normal((1, 8, 4096, 64))
            kv.values = mx.random.normal((1, 8, 4096, 64))
            kv.offset = 100
            cache.append(kv)
        mx.eval(*[kv.keys for kv in cache], *[kv.values for kv in cache])

        tokens = list(range(100))
        pc.store(tokens, cache)

        stored_entry = list(pc._entries.values())[0]
        for layer in stored_entry.cache:
            assert layer.keys.shape[2] == 100, (
                f"Expected trimmed to 100, got {layer.keys.shape[2]}"
            )
            assert layer.values.shape[2] == 100
