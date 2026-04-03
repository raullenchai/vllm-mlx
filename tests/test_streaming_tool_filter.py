"""Tests for StreamingToolCallFilter - suppresses tool call XML during streaming."""

import unittest

from vllm_mlx.api.utils import StreamingToolCallFilter


class TestStreamingToolCallFilter(unittest.TestCase):
    def test_normal_text_passes_through(self):
        f = StreamingToolCallFilter()
        assert f.process("Hello world") == "Hello world"

    def test_minimax_tool_call_suppressed(self):
        f = StreamingToolCallFilter()
        f.process("<minimax:tool_call>")
        f.process('<invoke name="read">')
        f.process('<parameter name="path">/tmp/test.txt</parameter>')
        f.process("</invoke>")
        result = f.process("</minimax:tool_call>")
        assert result == ""

    def test_text_after_tool_call_emits(self):
        f = StreamingToolCallFilter()
        f.process("<minimax:tool_call>content</minimax:tool_call>")
        assert f.process("After") == "After"

    def test_text_before_and_after_same_delta(self):
        f = StreamingToolCallFilter()
        result = f.process("Before <minimax:tool_call>inside</minimax:tool_call>After")
        assert result == "Before After"

    def test_split_across_deltas(self):
        f = StreamingToolCallFilter()
        r1 = f.process("Before <minim")
        r2 = f.process("ax:tool_call>inside</minimax:tool_call>After")
        assert r1 + r2 == "Before After"

    def test_qwen_format_suppressed(self):
        f = StreamingToolCallFilter()
        result = f.process('Text <tool_call>{"name":"fn"}</tool_call> more')
        assert result == "Text  more"

    def test_multiple_tool_calls(self):
        f = StreamingToolCallFilter()
        result = f.process(
            "A <minimax:tool_call>x</minimax:tool_call>"
            " B <minimax:tool_call>y</minimax:tool_call> C"
        )
        assert result == "A  B  C"

    def test_flush_partial_tag_emits(self):
        f = StreamingToolCallFilter()
        r = f.process("text <minim")
        fl = f.flush()
        assert r + fl == "text <minim"

    def test_flush_unterminated_block_discards(self):
        f = StreamingToolCallFilter()
        f.process("<minimax:tool_call>partial content")
        assert f.flush() == ""

    def test_large_tool_call_content(self):
        """Simulates a Read tool returning a large file."""
        f = StreamingToolCallFilter()
        big = "x" * 10000
        result = f.process(f"Before <minimax:tool_call>{big}</minimax:tool_call>After")
        assert result == "Before After"

    def test_think_tags_not_filtered(self):
        f = StreamingToolCallFilter()
        result = f.process("<think>reasoning here</think>answer")
        assert "<think>" in result
        assert "reasoning here" in result

    def test_mixed_think_and_tool_call(self):
        f = StreamingToolCallFilter()
        result = f.process(
            "<think>thinking</think>"
            "<minimax:tool_call>tool stuff</minimax:tool_call>"
            "final answer"
        )
        assert "<think>thinking</think>" in result
        assert "tool stuff" not in result
        assert "final answer" in result

    def test_gradual_token_by_token(self):
        """Simulate token-by-token streaming."""
        f = StreamingToolCallFilter()
        parts = [
            "Hello ",
            "<",
            "mini",
            "max:",
            "tool_call",
            ">",
            '<invoke name="test">',
            "</invoke>",
            "</minimax:tool_call>",
            " world",
        ]
        result = ""
        for part in parts:
            result += f.process(part)
        result += f.flush()
        assert result == "Hello  world", f"Got: {result!r}"

    def test_empty_deltas(self):
        f = StreamingToolCallFilter()
        assert f.process("") == ""
        assert f.process("text") == "text"
        assert f.process("") == ""

    def test_calling_tool_bracket_suppressed(self):
        """Qwen3 bracket-style: [Calling tool: func({...})]\n"""
        f = StreamingToolCallFilter()
        result = f.process('[Calling tool: search({"q": "test"})]\n')
        assert result == ""

    def test_calling_tool_multiline_json(self):
        """Multi-line JSON args in bracket-style tool call."""
        f = StreamingToolCallFilter()
        r1 = f.process('[Calling tool: search({"q": "test",')
        r2 = f.process(' "limit": 5})]\n')
        r3 = f.process("After")
        assert r1 + r2 + r3 == "After"

    def test_buffer_cap_on_unclosed_block(self):
        """Buffer should be capped if tool call block never closes."""
        from vllm_mlx.api.utils import _MAX_TOOL_BUFFER_BYTES

        f = StreamingToolCallFilter()
        f.process("<minimax:tool_call>")
        # Feed data exceeding the cap
        chunk = "x" * 10000
        for _ in range(_MAX_TOOL_BUFFER_BYTES // 10000 + 2):
            f.process(chunk)
        # After cap, filter should have exited the block
        assert not f._in_block
        # New text should pass through
        assert f.process("after") == "after"


if __name__ == "__main__":
    unittest.main()
