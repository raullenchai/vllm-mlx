# SPDX-License-Identifier: Apache-2.0
"""
Shared chat template application logic for both SimpleEngine and BatchedEngine.

This module eliminates duplication between engines for chat template handling,
ensuring consistent behavior for enable_thinking, tools, and fallback logic.
"""

import copy
import json
import logging

logger = logging.getLogger(__name__)


def _build_tool_injection_text(tools: list[dict]) -> str:
    """Build a compact tool definition string for system prompt injection.

    When a chat template doesn't support the ``tools`` parameter natively,
    we inject tool definitions into the system message so the model can
    still see them.

    Args:
        tools: List of tool definitions in OpenAI function-calling format.

    Returns:
        A formatted string describing available tools and calling format.
    """
    lines = ["# Available Tools", ""]
    for tool in tools:
        func = tool.get("function", tool)
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])

        lines.append(f"## {name}")
        if desc:
            lines.append(f"{desc}")
        if props:
            lines.append(f"Parameters: {json.dumps(props, ensure_ascii=False)}")
        if required:
            lines.append(f"Required: {json.dumps(required)}")
        lines.append("")

    lines.append("To call a tool, output exactly this format:")
    lines.append("<tool_call>")
    lines.append('{"name": "tool_name", "arguments": {"param": "value"}}')
    lines.append("</tool_call>")

    return "\n".join(lines)


def _inject_tools_into_messages(
    messages: list[dict], tools: list[dict]
) -> list[dict]:
    """Inject tool definitions into the system message.

    If the first message has role ``system``, append to its content.
    Otherwise, prepend a new system message with the tool definitions.

    Args:
        messages: Original messages (not mutated).
        tools: Tool definitions to inject.

    Returns:
        A shallow copy of messages with tool definitions injected.
    """
    injection = _build_tool_injection_text(tools)
    msgs = copy.copy(messages)

    if msgs and msgs[0].get("role") == "system":
        first = dict(msgs[0])
        first["content"] = first.get("content", "") + "\n\n" + injection
        msgs[0] = first
    else:
        msgs.insert(0, {"role": "system", "content": injection})

    return msgs


def apply_chat_template(
    template_applicator,
    messages: list[dict],
    tools: list[dict] | None = None,
    enable_thinking: bool | None = None,
    model_name: str = "",
) -> str:
    """Apply a chat template to messages with consistent fallback behavior.

    This is the shared implementation used by both SimpleEngine and BatchedEngine
    to ensure parity in template application, especially for ``enable_thinking``
    and ``tools`` parameters.

    Args:
        template_applicator: Object with ``apply_chat_template`` method
            (tokenizer or processor).
        messages: List of chat messages in OpenAI format.
        tools: Converted tool definitions for the template, or None.
        enable_thinking: Whether to enable thinking mode.
            - True/False: explicit control
            - None: auto-detect (True except for coder models)
        model_name: Model name string, used for auto-detection of
            ``enable_thinking`` when set to None.

    Returns:
        The formatted prompt string.  Falls back to a plain
        ``role: content`` format if the applicator has no
        ``apply_chat_template`` method.
    """
    if not hasattr(template_applicator, "apply_chat_template"):
        # Fallback for models without apply_chat_template
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return prompt + "\nassistant:"

    if enable_thinking is None:
        enable_thinking = "coder" not in model_name.lower()

    template_kwargs: dict = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    if tools:
        template_kwargs["tools"] = tools

    try:
        return template_applicator.apply_chat_template(messages, **template_kwargs)
    except TypeError as e:
        # Template doesn't support tools/enable_thinking params.
        logger.debug("Chat template TypeError, retrying without extras: %s", e)
        for key in ["tools", "enable_thinking"]:
            template_kwargs.pop(key, None)

        if tools:
            # Inject tool definitions into system message so the model
            # can still see them even though the template rejects the
            # tools parameter.
            logger.info(
                "Chat template doesn't support tools param — "
                "injecting %d tool definitions into system prompt",
                len(tools),
            )
            injected = _inject_tools_into_messages(messages, tools)
            return template_applicator.apply_chat_template(
                injected, **template_kwargs
            )

        return template_applicator.apply_chat_template(messages, **template_kwargs)
