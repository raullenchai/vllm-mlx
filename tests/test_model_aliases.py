# SPDX-License-Identifier: Apache-2.0
"""Tests for the model alias registry."""

from vllm_mlx.model_aliases import list_aliases, resolve_model


def test_known_alias_resolves():
    assert resolve_model("qwen3.5-9b") == "mlx-community/Qwen3.5-9B-4bit"
    assert resolve_model("llama3-3b") == "mlx-community/Llama-3.2-3B-Instruct-4bit"


def test_full_path_passes_through():
    assert resolve_model("mlx-community/Foo-Bar") == "mlx-community/Foo-Bar"
    assert resolve_model("/Users/me/local-model") == "/Users/me/local-model"


def test_unknown_name_passes_through():
    assert resolve_model("nonexistent-model") == "nonexistent-model"


def test_list_aliases_nonempty():
    aliases = list_aliases()
    assert len(aliases) >= 15
    assert "qwen3.5-9b" in aliases
