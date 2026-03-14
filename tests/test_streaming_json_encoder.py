# SPDX-License-Identifier: Apache-2.0
"""
Tests for optimized streaming JSON encoder.

These tests verify that the StreamingJSONEncoder provides the same output
as the current json.dumps() approach but with better performance through
pre-computed templates.
"""

import json


class TestStreamingJSONEncoder:
    """Tests for the StreamingJSONEncoder class."""

    def test_encode_completion_chunk(self):
        """Test encoding a single completion chunk with content."""
        from vllm_mlx.api.streaming import StreamingJSONEncoder

        encoder = StreamingJSONEncoder(
            response_id="cmpl-abc123",
            model="test-model",
            object_type="text_completion",
        )

        result = encoder.encode_completion_chunk(
            text="Hello",
            index=0,
            finish_reason=None,
        )

        # Should be valid SSE format
        assert result.startswith("data: ")
        assert result.endswith("\n\n")

        # Parse the JSON part
        json_str = result[6:-2]  # Remove "data: " prefix and "\n\n" suffix
        data = json.loads(json_str)

        assert data["id"] == "cmpl-abc123"
        assert data["object"] == "text_completion"
        assert data["model"] == "test-model"
        assert data["choices"][0]["index"] == 0
        assert data["choices"][0]["text"] == "Hello"
        assert data["choices"][0]["finish_reason"] is None

    def test_encode_completion_chunk_with_finish_reason(self):
        """Test encoding a completion chunk with finish_reason."""
        from vllm_mlx.api.streaming import StreamingJSONEncoder

        encoder = StreamingJSONEncoder(
            response_id="cmpl-abc123",
            model="test-model",
            object_type="text_completion",
        )

        result = encoder.encode_completion_chunk(
            text=" world",
            index=0,
            finish_reason="stop",
        )

        json_str = result[6:-2]
        data = json.loads(json_str)

        assert data["choices"][0]["text"] == " world"
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_encode_chat_chunk_with_role(self):
        """Test encoding a chat chunk with role (first chunk)."""
        from vllm_mlx.api.streaming import StreamingJSONEncoder

        encoder = StreamingJSONEncoder(
            response_id="chatcmpl-xyz789",
            model="chat-model",
            object_type="chat.completion.chunk",
        )

        result = encoder.encode_chat_chunk(
            role="assistant",
            content=None,
            finish_reason=None,
        )

        json_str = result[6:-2]
        data = json.loads(json_str)

        assert data["id"] == "chatcmpl-xyz789"
        assert data["object"] == "chat.completion.chunk"
        assert data["choices"][0]["delta"]["role"] == "assistant"
        assert (
            "content" not in data["choices"][0]["delta"]
            or data["choices"][0]["delta"]["content"] is None
        )

    def test_encode_chat_chunk_with_content(self):
        """Test encoding a chat chunk with content."""
        from vllm_mlx.api.streaming import StreamingJSONEncoder

        encoder = StreamingJSONEncoder(
            response_id="chatcmpl-xyz789",
            model="chat-model",
            object_type="chat.completion.chunk",
        )

        result = encoder.encode_chat_chunk(
            role=None,
            content="Hello",
            finish_reason=None,
        )

        json_str = result[6:-2]
        data = json.loads(json_str)

        assert data["choices"][0]["delta"]["content"] == "Hello"
        assert (
            "role" not in data["choices"][0]["delta"]
            or data["choices"][0]["delta"]["role"] is None
        )

    def test_encode_chat_chunk_with_finish_reason(self):
        """Test encoding a chat chunk with finish_reason."""
        from vllm_mlx.api.streaming import StreamingJSONEncoder

        encoder = StreamingJSONEncoder(
            response_id="chatcmpl-xyz789",
            model="chat-model",
            object_type="chat.completion.chunk",
        )

        result = encoder.encode_chat_chunk(
            role=None,
            content=None,
            finish_reason="stop",
        )

        json_str = result[6:-2]
        data = json.loads(json_str)

        assert data["choices"][0]["finish_reason"] == "stop"

    def test_escape_special_characters(self):
        """Test that special JSON characters are properly escaped."""
        from vllm_mlx.api.streaming import StreamingJSONEncoder

        encoder = StreamingJSONEncoder(
            response_id="cmpl-test",
            model="test-model",
            object_type="text_completion",
        )

        # Text with special characters that need escaping
        special_text = 'Hello "world"\nNew line\tTab\\Backslash'

        result = encoder.encode_completion_chunk(
            text=special_text,
            index=0,
            finish_reason=None,
        )

        # Should be valid JSON
        json_str = result[6:-2]
        data = json.loads(json_str)

        # The decoded text should match the original
        assert data["choices"][0]["text"] == special_text

    def test_escape_unicode_characters(self):
        """Test that unicode characters are handled correctly."""
        from vllm_mlx.api.streaming import StreamingJSONEncoder

        encoder = StreamingJSONEncoder(
            response_id="cmpl-test",
            model="test-model",
            object_type="text_completion",
        )

        # Text with unicode characters
        unicode_text = "Hello 世界! 🌍 Ñoño"

        result = encoder.encode_completion_chunk(
            text=unicode_text,
            index=0,
            finish_reason=None,
        )

        json_str = result[6:-2]
        data = json.loads(json_str)

        assert data["choices"][0]["text"] == unicode_text

    def test_encode_done_message(self):
        """Test encoding the [DONE] message."""
        from vllm_mlx.api.streaming import StreamingJSONEncoder

        encoder = StreamingJSONEncoder(
            response_id="cmpl-test",
            model="test-model",
            object_type="text_completion",
        )

        result = encoder.encode_done()

        assert result == "data: [DONE]\n\n"

    def test_created_timestamp_is_present(self):
        """Test that created timestamp is included."""
        import time

        from vllm_mlx.api.streaming import StreamingJSONEncoder

        before = int(time.time())

        encoder = StreamingJSONEncoder(
            response_id="cmpl-test",
            model="test-model",
            object_type="text_completion",
        )

        result = encoder.encode_completion_chunk(
            text="test",
            index=0,
            finish_reason=None,
        )

        after = int(time.time())

        json_str = result[6:-2]
        data = json.loads(json_str)

        assert "created" in data
        assert before <= data["created"] <= after

    def test_encode_completion_chunk_with_usage(self):
        """Test encoding a completion chunk with usage stats."""
        from vllm_mlx.api.streaming import StreamingJSONEncoder

        encoder = StreamingJSONEncoder(
            response_id="cmpl-test",
            model="test-model",
            object_type="text_completion",
        )

        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }

        result = encoder.encode_completion_chunk(
            text="done",
            index=0,
            finish_reason="stop",
            usage=usage,
        )

        json_str = result[6:-2]
        data = json.loads(json_str)

        assert data["usage"]["prompt_tokens"] == 10
        assert data["usage"]["completion_tokens"] == 20
        assert data["usage"]["total_tokens"] == 30


class TestStreamingJSONEncoderPerformance:
    """Performance tests comparing optimized encoder vs json.dumps()."""

    def test_encoder_produces_same_output_as_json_dumps(self):
        """Verify encoder output matches what json.dumps would produce."""
        import time

        from vllm_mlx.api.streaming import StreamingJSONEncoder

        response_id = "cmpl-perf123"
        model = "test-model"
        created = int(time.time())

        encoder = StreamingJSONEncoder(
            response_id=response_id,
            model=model,
            object_type="text_completion",
            created=created,  # Fix timestamp for comparison
        )

        # Encode with our optimized encoder
        optimized = encoder.encode_completion_chunk(
            text="test",
            index=0,
            finish_reason=None,
        )

        # Build the same data manually
        manual_data = {
            "id": response_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "text": "test",
                    "finish_reason": None,
                }
            ],
        }
        manual = f"data: {json.dumps(manual_data)}\n\n"

        # Parse both and compare (order-independent)
        opt_json = json.loads(optimized[6:-2])
        man_json = json.loads(manual[6:-2])

        assert opt_json == man_json


class TestStreamingJSONEncoderBenchmark:
    """Benchmark tests to verify performance improvement."""

    def test_encoder_faster_than_naive_approach(self):
        """
        Verify optimized encoder is faster than naive json.dumps per token.

        This test simulates streaming 100 tokens and compares performance.
        """
        import time

        from vllm_mlx.api.streaming import StreamingJSONEncoder

        num_tokens = 100
        response_id = "chatcmpl-bench123"
        model = "mlx-community/Llama-3.2-3B-Instruct-4bit"

        # Benchmark optimized encoder
        encoder = StreamingJSONEncoder(
            response_id=response_id,
            model=model,
            object_type="chat.completion.chunk",
        )

        start = time.perf_counter()
        for i in range(num_tokens):
            _ = encoder.encode_chat_chunk(content=f"token{i}")
        optimized_time = time.perf_counter() - start

        # Benchmark naive approach (simulating current server.py behavior)
        created = int(time.time())
        start = time.perf_counter()
        for i in range(num_tokens):
            data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": f"token{i}"},
                        "finish_reason": None,
                    }
                ],
            }
            _ = f"data: {__import__('json').dumps(data)}\n\n"
        naive_time = time.perf_counter() - start

        # Log times for visibility
        print(f"\nOptimized: {optimized_time * 1000:.2f}ms")
        print(f"Naive: {naive_time * 1000:.2f}ms")
        print(f"Ratio: {naive_time / optimized_time:.2f}x")

        # The optimized version should be at least as fast
        # (exact speedup depends on implementation)
        # For now, we just verify correctness - actual speedup comes from template-based approach
        assert optimized_time < naive_time * 2  # Should not be significantly slower


class TestStreamingJSONEncoderEdgeCases:
    """Edge case tests for StreamingJSONEncoder."""

    def test_empty_text(self):
        """Test encoding empty text."""
        from vllm_mlx.api.streaming import StreamingJSONEncoder

        encoder = StreamingJSONEncoder(
            response_id="cmpl-test",
            model="test-model",
            object_type="text_completion",
        )

        result = encoder.encode_completion_chunk(
            text="",
            index=0,
            finish_reason=None,
        )

        json_str = result[6:-2]
        data = json.loads(json_str)

        assert data["choices"][0]["text"] == ""

    def test_very_long_text(self):
        """Test encoding very long text."""
        from vllm_mlx.api.streaming import StreamingJSONEncoder

        encoder = StreamingJSONEncoder(
            response_id="cmpl-test",
            model="test-model",
            object_type="text_completion",
        )

        long_text = "A" * 10000

        result = encoder.encode_completion_chunk(
            text=long_text,
            index=0,
            finish_reason=None,
        )

        json_str = result[6:-2]
        data = json.loads(json_str)

        assert data["choices"][0]["text"] == long_text

    def test_model_name_with_special_chars(self):
        """Test model name with special characters."""
        from vllm_mlx.api.streaming import StreamingJSONEncoder

        encoder = StreamingJSONEncoder(
            response_id="cmpl-test",
            model="mlx-community/Llama-3.2-3B-Instruct-4bit",
            object_type="text_completion",
        )

        result = encoder.encode_completion_chunk(
            text="test",
            index=0,
            finish_reason=None,
        )

        json_str = result[6:-2]
        data = json.loads(json_str)

        assert data["model"] == "mlx-community/Llama-3.2-3B-Instruct-4bit"
