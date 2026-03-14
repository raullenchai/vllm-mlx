# SPDX-License-Identifier: Apache-2.0
"""
Test that MLLM stream_chat holds the generation lock.

Bug: SimpleEngine.stream_chat() MLLM path does not acquire
_generation_lock, allowing concurrent MLLM streaming requests
to race on shared Metal command buffers.

Every other generation method acquires the lock:
  - generate()        → async with self._generation_lock
  - stream_generate() → async with self._generation_lock
  - chat()            → async with self._generation_lock
  - stream_chat() LLM → calls stream_generate() which locks
  - stream_chat() MLLM → BUG: no lock

This test verifies the fix by running two concurrent stream_chat
calls on a mock MLLM SimpleEngine and asserting they serialize
(no overlap in execution).
"""

import asyncio
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from vllm_mlx.engine.simple import SimpleEngine


@dataclass
class FakeChunk:
    text: str
    finish_reason: str | None = None
    prompt_tokens: int = 5


def make_fake_mllm_model(execution_log: list, label: str, delay: float = 0.05):
    """Create a fake MLLM model whose stream_chat records execution order."""

    def stream_chat(messages, max_tokens, temperature, top_p, **kwargs):
        # Each call yields 3 chunks with a blocking sleep between them.
        # If two generators interleave, the execution_log will show
        # mixed labels like [A-start, B-start, A-1, B-1, ...].
        # With proper locking, it should be [A-start, A-1, ..., A-end, B-start, ...]
        import time

        execution_log.append(f"{label}-start")
        for i in range(3):
            time.sleep(delay)  # simulate work on Metal
            is_last = i == 2
            # Log end BEFORE the final yield since stream_chat breaks
            # after seeing finish_reason and never calls next() again.
            if is_last:
                execution_log.append(f"{label}-end")
            execution_log.append(f"{label}-{i}")
            yield FakeChunk(
                text=f"tok{i}",
                finish_reason="stop" if is_last else None,
            )

    model = MagicMock()
    model.stream_chat = stream_chat
    return model


@pytest.mark.asyncio
async def test_mllm_stream_chat_holds_lock():
    """Two concurrent MLLM stream_chat calls must not interleave."""
    execution_log: list[str] = []

    # Create engine in MLLM mode without loading a real model
    with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=True):
        engine = SimpleEngine(model_name="fake-mllm")

    engine._loaded = True
    engine._model = make_fake_mllm_model(execution_log, "A", delay=0.02)

    model_b = make_fake_mllm_model(execution_log, "B", delay=0.02)

    messages = [{"role": "user", "content": "describe the image"}]

    async def run_stream(label: str):
        # Swap model for B after A starts (both use same engine/lock)
        if label == "B":
            engine._model = model_b

        results = []
        async for chunk in engine.stream_chat(
            messages=messages, max_tokens=10, temperature=0.0, top_p=1.0
        ):
            results.append(chunk)
        return results

    # Launch both concurrently
    task_a = asyncio.create_task(run_stream("A"))
    # Small delay so A grabs the lock first
    await asyncio.sleep(0.001)
    task_b = asyncio.create_task(run_stream("B"))

    results_a, results_b = await asyncio.gather(task_a, task_b)

    # Both should complete successfully
    assert len(results_a) == 3
    assert len(results_b) == 3

    # Key assertion: A must fully complete before B starts.
    # Find indices in the log.
    a_end_idx = execution_log.index("A-end")
    b_start_idx = execution_log.index("B-start")
    assert a_end_idx < b_start_idx, (
        f"MLLM stream_chat calls interleaved! Lock not held.\n"
        f"Execution order: {execution_log}"
    )


@pytest.mark.asyncio
async def test_mllm_stream_chat_serializes_with_chat():
    """MLLM stream_chat and chat() must not run concurrently."""
    execution_log: list[str] = []

    with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=True):
        engine = SimpleEngine(model_name="fake-mllm")

    engine._loaded = True

    # stream_chat model
    engine._model = make_fake_mllm_model(execution_log, "STREAM", delay=0.02)

    # Patch chat path on model
    @dataclass
    class FakeChatOutput:
        text: str = "response"
        prompt_tokens: int = 5
        completion_tokens: int = 3
        finish_reason: str = "stop"

    def fake_chat(**kwargs):
        import time

        execution_log.append("CHAT-start")
        time.sleep(0.03)
        execution_log.append("CHAT-end")
        return FakeChatOutput()

    messages = [{"role": "user", "content": "describe"}]

    async def run_stream():
        async for _ in engine.stream_chat(
            messages=messages, max_tokens=10, temperature=0.0, top_p=1.0
        ):
            pass

    async def run_chat():
        # Replace model's chat method for the non-streaming call
        engine._model.chat = fake_chat
        await engine.chat(messages=messages, max_tokens=10, temperature=0.0, top_p=1.0)

    task_stream = asyncio.create_task(run_stream())
    await asyncio.sleep(0.001)
    task_chat = asyncio.create_task(run_chat())

    await asyncio.gather(task_stream, task_chat)

    # stream must finish before chat starts
    stream_end = execution_log.index("STREAM-end")
    chat_start = execution_log.index("CHAT-start")
    assert stream_end < chat_start, (
        f"stream_chat and chat() ran concurrently!\nExecution order: {execution_log}"
    )


@pytest.mark.asyncio
async def test_mllm_stream_chat_serializes_with_generate():
    """MLLM stream_chat and generate() must not run concurrently."""
    execution_log: list[str] = []

    with patch("vllm_mlx.engine.simple.is_mllm_model", return_value=True):
        engine = SimpleEngine(model_name="fake-mllm")

    engine._loaded = True
    engine._model = make_fake_mllm_model(execution_log, "STREAM", delay=0.02)

    @dataclass
    class FakeGenOutput:
        text: str = "generated"
        tokens: list = None
        prompt_tokens: int = 5
        completion_tokens: int = 3
        finish_reason: str = "stop"

        def __post_init__(self):
            if self.tokens is None:
                self.tokens = [1, 2, 3]

    def fake_generate(**kwargs):
        import time

        execution_log.append("GEN-start")
        time.sleep(0.03)
        execution_log.append("GEN-end")
        return FakeGenOutput()

    engine._model.generate = fake_generate

    messages = [{"role": "user", "content": "describe"}]

    async def run_stream():
        async for _ in engine.stream_chat(
            messages=messages, max_tokens=10, temperature=0.0, top_p=1.0
        ):
            pass

    async def run_generate():
        await engine.generate(prompt="test", max_tokens=10)

    task_stream = asyncio.create_task(run_stream())
    await asyncio.sleep(0.001)
    task_gen = asyncio.create_task(run_generate())

    await asyncio.gather(task_stream, task_gen)

    stream_end = execution_log.index("STREAM-end")
    gen_start = execution_log.index("GEN-start")
    assert stream_end < gen_start, (
        f"stream_chat and generate() ran concurrently!\n"
        f"Execution order: {execution_log}"
    )
