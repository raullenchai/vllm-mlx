#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Simple text generation example using vllm-mlx.

This example demonstrates basic LLM inference on Apple Silicon
using the MLX backend.
"""

from vllm_mlx.models import MLXLanguageModel


def main():
    # Use a small quantized model for quick testing
    model_name = "mlx-community/Llama-3.2-3B-Instruct-4bit"

    print(f"Loading model: {model_name}")
    model = MLXLanguageModel(model_name)
    model.load()

    print("\n" + "=" * 50)
    print("Model loaded! Starting generation...")
    print("=" * 50 + "\n")

    # Simple generation
    prompt = "What is the meaning of life?"
    print(f"Prompt: {prompt}\n")

    output = model.generate(
        prompt,
        max_tokens=200,
        temperature=0.7,
    )

    print(f"Response:\n{output.text}")
    print(f"\nFinish reason: {output.finish_reason}")

    # Streaming generation
    print("\n" + "=" * 50)
    print("Streaming generation:")
    print("=" * 50 + "\n")

    prompt = "Write a haiku about coding:"
    print(f"Prompt: {prompt}\n")
    print("Response: ", end="", flush=True)

    for chunk in model.stream_generate(
        prompt,
        max_tokens=100,
        temperature=0.8,
    ):
        print(chunk.text, end="", flush=True)

    print("\n")

    # Chat interface
    print("=" * 50)
    print("Chat interface:")
    print("=" * 50 + "\n")

    messages = [{"role": "user", "content": "Hello! Can you introduce yourself?"}]

    response = model.chat(messages, max_tokens=150)
    print(f"User: {messages[0]['content']}")
    print(f"Assistant: {response.text}")


if __name__ == "__main__":
    main()
