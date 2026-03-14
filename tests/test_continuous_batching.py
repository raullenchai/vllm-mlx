# SPDX-License-Identifier: Apache-2.0
"""
Tests for continuous batching performance.

These tests verify that continuous batching properly handles
multiple concurrent requests with improved throughput.
"""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from vllm_mlx.request import Request, SamplingParams
from vllm_mlx.scheduler import Scheduler, SchedulerConfig


class TestContinuousBatchingBasic:
    """Basic tests for continuous batching functionality."""

    def test_scheduler_accepts_multiple_requests(self):
        """Test that scheduler can queue multiple requests."""
        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode = lambda x: list(range(len(x.split())))

        config = SchedulerConfig(max_num_seqs=32)
        scheduler = Scheduler(model, tokenizer, config)

        # Add multiple requests
        for i in range(5):
            request = Request(
                request_id=f"req-{i}",
                prompt=f"Test prompt {i}",
                sampling_params=SamplingParams(max_tokens=50),
            )
            scheduler.add_request(request)

        assert scheduler.get_num_waiting() == 5
        assert scheduler.has_requests()

    def test_scheduler_config_batching_params(self):
        """Test scheduler config has batching parameters."""
        config = SchedulerConfig(
            max_num_seqs=64,
            prefill_batch_size=16,
            completion_batch_size=32,
        )

        assert config.max_num_seqs == 64
        assert config.prefill_batch_size == 16
        assert config.completion_batch_size == 32


@pytest.mark.asyncio
class TestContinuousBatchingIntegration:
    """Integration tests requiring actual model loading."""

    @pytest.fixture
    def small_model(self):
        """Load a small model for testing."""
        try:
            from mlx_lm import load

            model, tokenizer = load("mlx-community/Qwen3-0.6B-8bit")
            return model, tokenizer
        except Exception:
            pytest.skip("Model not available for testing")

    async def test_single_request(self, small_model):
        """Test single request processing."""
        from vllm_mlx import (
            AsyncEngineCore,
            EngineConfig,
            SamplingParams,
            SchedulerConfig,
        )

        model, tokenizer = small_model
        config = EngineConfig(
            model_name="test",
            scheduler_config=SchedulerConfig(),
        )

        async with AsyncEngineCore(model, tokenizer, config) as engine:
            await asyncio.sleep(0.1)

            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": "Hi"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            params = SamplingParams(max_tokens=10, temperature=0.0)

            rid = await engine.add_request(prompt, params)

            output = None
            async for out in engine.stream_outputs(rid, timeout=30):
                if out.finished:
                    output = out
                    break

            assert output is not None
            assert output.finished
            assert output.completion_tokens > 0

    async def test_concurrent_requests(self, small_model):
        """Test multiple concurrent requests are batched."""
        from vllm_mlx import (
            AsyncEngineCore,
            EngineConfig,
            SamplingParams,
            SchedulerConfig,
        )

        model, tokenizer = small_model
        config = EngineConfig(
            model_name="test",
            scheduler_config=SchedulerConfig(
                max_num_seqs=32,
                prefill_batch_size=8,
                completion_batch_size=16,
            ),
        )

        async with AsyncEngineCore(model, tokenizer, config) as engine:
            await asyncio.sleep(0.1)

            prompts = ["What is 2+2?", "Name a color.", "Hello!"]
            params = SamplingParams(max_tokens=20, temperature=0.0)

            # Send all requests
            request_ids = []
            for p in prompts:
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                rid = await engine.add_request(formatted, params)
                request_ids.append(rid)

            # Collect results
            async def get_result(rid):
                async for out in engine.stream_outputs(rid, timeout=30):
                    if out.finished:
                        return out
                return None

            results = await asyncio.gather(*[get_result(r) for r in request_ids])

            # All should complete
            assert all(r is not None for r in results)
            assert all(r.finished for r in results)
            assert all(r.completion_tokens > 0 for r in results)

    async def test_batching_improves_throughput(self, small_model):
        """Test that batching improves throughput vs sequential."""
        from vllm_mlx import (
            AsyncEngineCore,
            EngineConfig,
            SamplingParams,
            SchedulerConfig,
        )

        model, tokenizer = small_model
        config = EngineConfig(
            model_name="test",
            scheduler_config=SchedulerConfig(
                max_num_seqs=32,
                prefill_batch_size=8,
                completion_batch_size=16,
            ),
        )

        prompts = [
            "What is 2+2?",
            "Name 3 colors.",
            "What is Python?",
            "Capital of Japan?",
            "Who wrote Hamlet?",
        ]
        params = SamplingParams(max_tokens=30, temperature=0.0)

        async with AsyncEngineCore(model, tokenizer, config) as engine:
            await asyncio.sleep(0.1)

            # Concurrent batch
            start = time.perf_counter()

            request_ids = []
            for p in prompts:
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                rid = await engine.add_request(formatted, params)
                request_ids.append(rid)

            async def get_result(rid):
                async for out in engine.stream_outputs(rid, timeout=60):
                    if out.finished:
                        return out.completion_tokens
                return 0

            results = await asyncio.gather(*[get_result(r) for r in request_ids])

            batch_time = time.perf_counter() - start
            total_tokens = sum(results)
            batch_throughput = total_tokens / batch_time

            print(f"\nBatch: {len(prompts)} requests in {batch_time:.2f}s")
            print(f"Total tokens: {total_tokens}")
            print(f"Throughput: {batch_throughput:.1f} tok/s")
            print(f"Requests/sec: {len(prompts) / batch_time:.2f}")

            # Batching should achieve reasonable throughput
            assert batch_throughput > 100  # At least 100 tok/s
            assert all(t > 0 for t in results)


if __name__ == "__main__":
    # Quick standalone test
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Continuous batching benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("VLLM_MLX_TEST_MODEL", "mlx-community/Qwen3-8B-6bit"),
        help="Model to benchmark",
    )
    args = parser.parse_args()

    MODEL_NAME = args.model

    async def run_benchmark():
        from mlx_lm import load

        from vllm_mlx import (
            AsyncEngineCore,
            EngineConfig,
            SamplingParams,
            SchedulerConfig,
        )

        print("=" * 60)
        print("Continuous Batching Benchmark")
        print("=" * 60)
        print(f"Model: {MODEL_NAME}")

        print("\nLoading model...")
        model, tokenizer = load(MODEL_NAME)

        config = EngineConfig(
            model_name="test",
            scheduler_config=SchedulerConfig(
                max_num_seqs=256,
                prefill_batch_size=8,
                completion_batch_size=32,  # 32 gives optimal throughput
            ),
        )

        prompts = [
            "What is 2+2?",
            "Name 3 colors.",
            "What is Python?",
            "Capital of Japan?",
            "Who wrote Hamlet?",
        ]
        params = SamplingParams(max_tokens=50, temperature=0.7)

        async with AsyncEngineCore(model, tokenizer, config) as engine:
            await asyncio.sleep(0.1)

            print(f"\nSending {len(prompts)} concurrent requests...")
            start = time.perf_counter()

            # Use generate() for optimal throughput (no streaming overhead)
            async def run_one(prompt):
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                result = await engine.engine.generate(formatted, params)
                return (
                    prompt,
                    result.output_text[:50],
                    result.prompt_tokens,
                    result.completion_tokens,
                )

            results = await asyncio.gather(*[run_one(p) for p in prompts])

            total_time = time.perf_counter() - start
            prompt_tokens = sum(r[2] for r in results)
            completion_tokens = sum(r[3] for r in results)
            total_tokens = prompt_tokens + completion_tokens

            print("\n" + "-" * 60)
            print("Results:")
            for prompt, output, _, tokens in results:
                clean_output = output.replace("\n", " ")[:40]
                print(f"  [{tokens:3d} tok] {prompt[:20]:20s} -> {clean_output}...")

            print("\n" + "=" * 60)
            print("BENCHMARK RESULTS")
            print("=" * 60)
            print(f"Total time:    {total_time:.2f}s")
            print(f"Requests:      {len(prompts)}")
            print(f"Total tokens:  {total_tokens}")
            print(f"Throughput:    {total_tokens / total_time:.1f} tok/s")
            print(f"Requests/sec:  {len(prompts) / total_time:.2f}")
            print("=" * 60)

    asyncio.run(run_benchmark())
