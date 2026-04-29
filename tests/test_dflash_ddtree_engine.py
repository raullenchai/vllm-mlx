# SPDX-License-Identifier: Apache-2.0
"""Tests for DFlashEngine DDTree routing."""

import concurrent.futures
import sys
import types

import pytest

from vllm_mlx.engine.dflash import DFlashEngine
from vllm_mlx.speculative.ddtree.engine import _looks_like_tool_call_draft


@pytest.mark.asyncio
async def test_dflash_engine_routes_to_ddtree(monkeypatch):
    fake_engine = types.ModuleType("vllm_mlx.speculative.ddtree.engine")

    def fake_generate_ddtree(**kwargs):
        assert kwargs["tree_budget"] == 4
        assert kwargs["block_size"] is None
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


@pytest.mark.asyncio
async def test_dflash_engine_passes_ngram_first_config(monkeypatch):
    fake_engine = types.ModuleType("vllm_mlx.speculative.ddtree.engine")

    def fake_generate_ddtree(**kwargs):
        assert kwargs["tree_budget"] == 4
        assert kwargs["ngram_num_draft_tokens"] == 6
        assert kwargs["ngram_size"] == 3
        assert kwargs["ngram_min_matches"] == 1
        assert kwargs["ngram_disable_threshold"] == 0.5
        assert kwargs["ngram_disable_window"] == 2
        assert kwargs["ngram_disable_cooldown"] == 7
        return {
            "text": "ok",
            "generated_token_ids": [1],
            "prompt_tokens": 3,
            "prefill_seconds": 0.25,
            "generated_tokens": 1,
            "finish_reason": "stop",
            "proposed_tokens": 2,
            "accepted_tokens": 1,
            "speculative_steps": 1,
            "avg_acceptance_ratio": 0.5,
            "block_size_history": [2],
            "avg_tree_node_count": 4.0,
            "ddtree_fast_path_ratio": 1.0,
            "tree_budget": 4,
            "generation_tps": 99.0,
            "ngram_acceptance_ratio": 0.75,
            "ngram_cycles_completed": 3,
            "ngram_fallback_cycles": 2,
            "ngram_tool_guard_cycles": 1,
        }

    fake_engine.generate_ddtree = fake_generate_ddtree
    monkeypatch.setitem(
        sys.modules,
        "vllm_mlx.speculative.ddtree.engine",
        fake_engine,
    )

    engine = DFlashEngine(
        model_name="dummy",
        drafter_path="dummy-drafter",
        ddtree_budget=4,
        fallback_mode="ngram",
        ngram_num_draft_tokens=6,
        ngram_size=3,
        ngram_min_matches=1,
        ngram_disable_threshold=0.5,
        ngram_disable_window=2,
        ngram_disable_cooldown=7,
    )
    engine._loaded = True
    engine._model = object()
    engine._drafter = object()
    engine._tokenizer = object()
    engine._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    active_stats = None
    try:
        outputs = []
        async for output in engine.stream_generate(
            "prompt",
            max_tokens=8,
            temperature=0,
            top_p=1,
        ):
            outputs.append(output)
            active_stats = engine.get_stats()
    finally:
        await engine.stop()

    assert outputs[-1].text == "ok"

    stats = engine.get_stats()
    assert active_stats is not None
    assert active_stats["requests"][0]["tokens_per_second"] == 99.0
    assert active_stats["requests"][0]["ttft_s"] == 0.25
    assert active_stats["requests"][0]["completion_tokens"] == 1
    assert stats["dflash"]["mode"] == "ddtree-ngram"
    assert stats["dflash"]["ngram_first_enabled"] is True
    assert stats["dflash"]["ngram_num_draft_tokens"] == 6
    assert stats["dflash"]["ngram_last_acceptance_ratio"] == 0.75
    assert stats["dflash"]["ngram_last_cycles"] == 3
    assert stats["dflash"]["ngram_last_fallback_cycles"] == 2
    assert stats["dflash"]["ngram_last_tool_guard_cycles"] == 1


def test_ddtree_ngram_tool_call_guard_detects_xml_markers():
    class FakeTokenizer:
        def decode(self, tokens, *args, **kwargs):
            if 4 in tokens:
                return "<tool_call><function=bash>"
            return "plain text"

    assert _looks_like_tool_call_draft(FakeTokenizer(), [1, 2], 3, [4])
    assert not _looks_like_tool_call_draft(FakeTokenizer(), [1, 2], 3, [5])
