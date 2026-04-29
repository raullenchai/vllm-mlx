# SPDX-License-Identifier: Apache-2.0
"""Tests for DFlashEngine DDTree routing."""

import concurrent.futures
import sys
import types

import pytest

from vllm_mlx.engine.dflash import DFlashEngine


@pytest.mark.asyncio
async def test_dflash_engine_routes_to_ddtree(monkeypatch):
    fake_engine = types.ModuleType("vllm_mlx.speculative.ddtree.engine")

    def fake_generate_ddtree(**kwargs):
        assert kwargs["tree_budget"] == 4
        assert kwargs["block_size"] == 2
        assert kwargs["stop_strings"] == ["STOP"]
        assert "</tool_call>" in kwargs["stop_after_strings"]
        return {
            "text": "ok",
            "generated_token_ids": [1, 2],
            "prompt_tokens": 3,
            "generated_tokens": 2,
            "finish_reason": "stop",
            "proposed_tokens": 4,
            "accepted_tokens": 3,
            "speculative_steps": 1,
            "avg_acceptance_ratio": 0.75,
            "block_size_history": [2],
            "avg_tree_node_count": 5.0,
            "ddtree_fast_path_ratio": 1.0,
            "tree_budget": 4,
            "generation_tps": 123.0,
        }

    fake_generate_ddtree.__name__ = "generate_ddtree"
    fake_engine.generate_ddtree = fake_generate_ddtree
    monkeypatch.setitem(
        sys.modules,
        "vllm_mlx.speculative.ddtree.engine",
        fake_engine,
    )

    engine = DFlashEngine(
        model_name="dummy",
        drafter_path="dummy-drafter",
        block_size=2,
        ddtree_budget=4,
    )
    engine._loaded = True
    engine._model = object()
    engine._drafter = object()
    engine._tokenizer = object()
    engine._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    try:
        outputs = [
            output
            async for output in engine.stream_generate(
                "prompt",
                max_tokens=8,
                temperature=0,
                top_p=1,
                stop=["STOP"],
                tools_requested=True,
            )
        ]
    finally:
        await engine.stop()

    assert outputs[-1].text == "ok"
    assert outputs[-1].completion_tokens == 2

    stats = engine.get_stats()
    assert stats["dflash"]["mode"] == "ddtree"
    assert stats["dflash"]["ddtree_budget"] == 4
    assert stats["dflash"]["ddtree_requests"] == 1
    assert stats["dflash"]["ddtree_last_generation_tps"] == 123.0
