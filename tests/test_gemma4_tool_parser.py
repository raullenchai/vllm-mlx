# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for Gemma 4 tool call parser.

Covers:
- Bare numeric/bool/null/float args:        {a:3,b:4}
- Quoted string args:                       {city:<|"|>Paris<|"|>}
- Mixed bare + quoted args:                 {a:3,b:<|"|>hi<|"|>}
- Content stripping (no leakage of markup)
- Multi-tool calls in one output
- Streaming dedup behavior
- Text-format fallback ([Calling tool: ...])

Fix history:
- 2026-04-07: original parser only handled `key:<|"|>value<|"|>` form;
  numeric args like `{a:3,b:4}` parsed as empty dict {}.
"""
import json

import pytest

from vllm_mlx.tool_parsers.gemma4_tool_parser import (
    Gemma4ToolParser,
    _parse_gemma4_args,
)

# ---------------------------------------------------------------------------
# _parse_gemma4_args — unit tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args_str,expected",
    [
        # bare numeric — the original bug
        ("a:3,b:4", {"a": 3, "b": 4}),
        # bare bool / null / float
        ("flag:true,n:42", {"flag": True, "n": 42}),
        ("x:null", {"x": None}),
        ("rate:0.5", {"rate": 0.5}),
        ("flag:false", {"flag": False}),
        # quoted string (existing form)
        ('city:<|"|>Paris<|"|>', {"city": "Paris"}),
        # mixed: numeric + quoted
        ('a:3,b:<|"|>hi<|"|>', {"a": 3, "b": "hi"}),
        # multiple quoted strings
        ('first:<|"|>Alice<|"|>,last:<|"|>Smith<|"|>', {"first": "Alice", "last": "Smith"}),
        # mixed: 3 fields, all types
        (
            'flag:true,n:42,name:<|"|>Bob<|"|>',
            {"flag": True, "n": 42, "name": "Bob"},
        ),
        # quoted string containing punctuation
        ('msg:<|"|>hello, world<|"|>', {"msg": "hello, world"}),
        # negative integer
        ("n:-5", {"n": -5}),
        # empty arg dict
        ("", {}),
    ],
)
def test_parse_gemma4_args(args_str, expected):
    assert _parse_gemma4_args(args_str) == expected


# ---------------------------------------------------------------------------
# extract_tool_calls — full markup
# ---------------------------------------------------------------------------


def test_extract_bare_numeric_args():
    """Original bug: {a:3,b:4} was returning empty arguments."""
    parser = Gemma4ToolParser()
    out = "<|tool_call>call:add{a:3,b:4}<tool_call|>"
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert len(res.tool_calls) == 1
    tc = res.tool_calls[0]
    assert tc["name"] == "add"
    args = json.loads(tc["arguments"])
    assert args == {"a": 3, "b": 4}
    # Content must NOT leak any markup
    assert res.content is None


def test_extract_quoted_string_args():
    parser = Gemma4ToolParser()
    out = '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>'
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    args = json.loads(res.tool_calls[0]["arguments"])
    assert args == {"city": "Paris"}
    assert res.content is None


def test_extract_mixed_args():
    parser = Gemma4ToolParser()
    out = '<|tool_call>call:mix{a:3,b:<|"|>hi<|"|>}<tool_call|>'
    res = parser.extract_tool_calls(out)
    args = json.loads(res.tool_calls[0]["arguments"])
    assert args == {"a": 3, "b": "hi"}
    assert res.content is None


def test_no_tool_call_returns_content_unchanged():
    parser = Gemma4ToolParser()
    out = "Hello, the answer is 42."
    res = parser.extract_tool_calls(out)
    assert res.tools_called is False
    assert res.content == out
    assert res.tool_calls == []


def test_multiple_tool_calls():
    parser = Gemma4ToolParser()
    out = (
        "<|tool_call>call:add{a:1,b:2}<tool_call|>"
        "<|tool_call>call:multiply{a:3,b:4}<tool_call|>"
    )
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    assert len(res.tool_calls) == 2
    assert res.tool_calls[0]["name"] == "add"
    assert json.loads(res.tool_calls[0]["arguments"]) == {"a": 1, "b": 2}
    assert res.tool_calls[1]["name"] == "multiply"
    assert json.loads(res.tool_calls[1]["arguments"]) == {"a": 3, "b": 4}
    assert res.content is None


def test_tool_call_with_surrounding_content():
    parser = Gemma4ToolParser()
    out = (
        "Let me check the weather. "
        '<|tool_call>call:get_weather{city:<|"|>NYC<|"|>}<tool_call|>'
        " That should help."
    )
    res = parser.extract_tool_calls(out)
    assert res.tools_called is True
    # Surrounding text preserved, markup stripped
    assert res.content is not None
    assert "<|tool_call>" not in res.content
    assert "<tool_call|>" not in res.content
    assert "weather" in res.content
    assert "should help" in res.content


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_streaming_emits_completed_tool_call_once():
    parser = Gemma4ToolParser()
    parser.reset()
    full = "<|tool_call>call:add{a:3,b:4}<tool_call|>"

    # Feed token-by-token-ish (split in halves)
    midpoint = len(full) // 2
    delta1 = full[:midpoint]
    delta2 = full[midpoint:]

    r1 = parser.extract_tool_calls_streaming("", delta1, delta1)
    # In the middle of an incomplete tool call — should suppress
    assert r1 is None

    r2 = parser.extract_tool_calls_streaming(delta1, full, delta2)
    # Now complete — should emit
    assert r2 is not None
    assert "tool_calls" in r2
    assert len(r2["tool_calls"]) == 1
    tc = r2["tool_calls"][0]
    assert tc["function"]["name"] == "add"
    assert json.loads(tc["function"]["arguments"]) == {"a": 3, "b": 4}

    # Subsequent calls with no new completed tools — no re-emit
    r3 = parser.extract_tool_calls_streaming(full, full, "")
    assert r3 is None


def test_streaming_passthrough_when_no_markup():
    parser = Gemma4ToolParser()
    parser.reset()
    r = parser.extract_tool_calls_streaming("", "Hello world", "Hello world")
    assert r == {"content": "Hello world"}
