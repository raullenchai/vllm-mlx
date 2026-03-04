# SPDX-License-Identifier: Apache-2.0
"""
Shared chat template application logic for both SimpleEngine and BatchedEngine.

This module eliminates duplication between engines for chat template handling,
ensuring consistent behavior for enable_thinking, tools, and fallback logic.
"""

import logging

logger = logging.getLogger(__name__)


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
        # Some templates don't support tools/enable_thinking; retry without them.
        logger.debug("Chat template TypeError, retrying without extras: %s", e)
        for key in ["tools", "enable_thinking"]:
            template_kwargs.pop(key, None)
        return template_applicator.apply_chat_template(messages, **template_kwargs)
