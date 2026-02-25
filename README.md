# vLLM-MLX

**vLLM-like inference for Apple Silicon** - GPU-accelerated Text, Image, Video & Audio on Mac

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Apple Silicon](https://img.shields.io/badge/Apple-Silicon-black.svg)](https://support.apple.com/en-us/HT211814)
[![GitHub](https://img.shields.io/badge/GitHub-waybarrios%2Fvllm--mlx-blue?logo=github)](https://github.com/waybarrios/vllm-mlx)

## Overview

vllm-mlx brings native Apple Silicon GPU acceleration to vLLM by integrating:

- **[MLX](https://github.com/ml-explore/mlx)**: Apple's ML framework with unified memory and Metal kernels
- **[mlx-lm](https://github.com/ml-explore/mlx-lm)**: Optimized LLM inference with KV cache and quantization
- **[mlx-vlm](https://github.com/Blaizzy/mlx-vlm)**: Vision-language models for multimodal inference
- **[mlx-audio](https://github.com/Blaizzy/mlx-audio)**: Speech-to-Text and Text-to-Speech with native voices
- **[mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings)**: Text embeddings for semantic search and RAG

## Features

- **Multimodal** - Text, Image, Video & Audio in one platform
- **Native GPU acceleration** on Apple Silicon (M1, M2, M3, M4)
- **Native TTS voices** - Spanish, French, Chinese, Japanese + 5 more languages
- **OpenAI API compatible** - drop-in replacement for OpenAI client
- **Anthropic Messages API** - native `/v1/messages` endpoint for Claude Code and OpenCode
- **Embeddings** - OpenAI-compatible `/v1/embeddings` endpoint with mlx-embeddings
- **Reasoning Models** - extract thinking process from Qwen3, DeepSeek-R1
- **MCP Tool Calling** - integrate external tools via Model Context Protocol
- **Paged KV Cache** - memory-efficient caching with prefix sharing
- **Continuous Batching** - high throughput for multiple concurrent users

## Quick Start

### Installation

**Using uv (recommended):**

```bash
# Install as CLI tool (system-wide)
uv tool install git+https://github.com/waybarrios/vllm-mlx.git

# Or install in a project/virtual environment
uv pip install git+https://github.com/waybarrios/vllm-mlx.git
```

**Using pip:**

```bash
# Install from GitHub
pip install git+https://github.com/waybarrios/vllm-mlx.git

# Or clone and install in development mode
git clone https://github.com/waybarrios/vllm-mlx.git
cd vllm-mlx
pip install -e .
```

### Start Server

```bash
# Simple mode (single user, max throughput)
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000

# Continuous batching (multiple users)
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --continuous-batching

# With API key authentication
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --port 8000 --api-key your-secret-key
```

### Use with OpenAI SDK

```python
from openai import OpenAI

# Without API key (local development)
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# With API key (production)
client = OpenAI(base_url="http://localhost:8000/v1", api_key="your-secret-key")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Use with Anthropic SDK

vllm-mlx exposes an Anthropic-compatible `/v1/messages` endpoint, so tools like Claude Code and OpenCode can connect directly.

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:8000", api_key="not-needed")

response = client.messages.create(
    model="default",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.content[0].text)
```

To use with Claude Code:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=not-needed
claude
```

See [Anthropic Messages API docs](docs/guides/server.md#anthropic-messages-api) for streaming, tool calling, system messages, and token counting.

### Multimodal (Images & Video)

```bash
vllm-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit --port 8000
```

```python
response = client.chat.completions.create(
    model="default",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    }]
)
```

### Audio (TTS/STT)

```bash
# Install audio dependencies
pip install vllm-mlx[audio]
python -m spacy download en_core_web_sm
brew install espeak-ng  # macOS, for non-English languages
```

```bash
# Text-to-Speech (English)
python examples/tts_example.py "Hello, how are you?" --play

# Text-to-Speech (Spanish)
python examples/tts_multilingual.py "Hola mundo" --lang es --play

# List available models and languages
python examples/tts_multilingual.py --list-models
python examples/tts_multilingual.py --list-languages
```

**Supported TTS Models:**
| Model | Languages | Description |
|-------|-----------|-------------|
| Kokoro | EN, ES, FR, JA, ZH, IT, PT, HI | Fast, 82M params, 11 voices |
| Chatterbox | 15+ languages | Expressive, voice cloning |
| VibeVoice | EN | Realtime, low latency |
| VoxCPM | ZH, EN | High quality Chinese/English |

### Reasoning Models

Extract the thinking process from reasoning models like Qwen3 and DeepSeek-R1:

```bash
# Start server with reasoning parser
vllm-mlx serve mlx-community/Qwen3-8B-4bit --reasoning-parser qwen3
```

```python
response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is 17 × 23?"}]
)

# Access reasoning separately from the answer
print("Thinking:", response.choices[0].message.reasoning)
print("Answer:", response.choices[0].message.content)
```

**Supported Parsers:**
| Parser | Models | Description |
|--------|--------|-------------|
| `qwen3` | Qwen3 series | Requires both `<think>` and `</think>` tags |
| `deepseek_r1` | DeepSeek-R1 | Handles implicit `<think>` tag |

### Embeddings

Generate text embeddings for semantic search, RAG, and similarity:

```bash
# Start server with an embedding model pre-loaded
vllm-mlx serve mlx-community/Llama-3.2-3B-Instruct-4bit --embedding-model mlx-community/all-MiniLM-L6-v2-4bit
```

```python
# Generate embeddings using the OpenAI SDK
embeddings = client.embeddings.create(
    model="mlx-community/all-MiniLM-L6-v2-4bit",
    input=["Hello world", "How are you?"]
)
print(f"Dimensions: {len(embeddings.data[0].embedding)}")
```

See [Embeddings Guide](docs/guides/embeddings.md) for details on supported models and lazy loading.

## Documentation

For full documentation, see the [docs](docs/) directory:

- **Getting Started**
  - [Installation](docs/getting-started/installation.md)
  - [Quick Start](docs/getting-started/quickstart.md)

- **User Guides**
  - [OpenAI-Compatible Server](docs/guides/server.md)
  - [Anthropic Messages API](docs/guides/server.md#anthropic-messages-api)
  - [Python API](docs/guides/python-api.md)
  - [Multimodal (Images & Video)](docs/guides/multimodal.md)
  - [Audio (STT/TTS)](docs/guides/audio.md)
  - [Embeddings](docs/guides/embeddings.md)
  - [Reasoning Models](docs/guides/reasoning.md)
  - [MCP & Tool Calling](docs/guides/mcp-tools.md)
  - [Continuous Batching](docs/guides/continuous-batching.md)

- **Reference**
  - [CLI Commands](docs/reference/cli.md)
  - [Supported Models](docs/reference/models.md)
  - [Configuration](docs/reference/configuration.md)

- **Benchmarks**
  - [LLM Benchmarks](docs/benchmarks/llm.md)
  - [Image Benchmarks](docs/benchmarks/image.md)
  - [Video Benchmarks](docs/benchmarks/video.md)
  - [Audio Benchmarks](docs/benchmarks/audio.md)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           vLLM API Layer                                │
│                    (OpenAI-compatible interface)                         │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            MLXPlatform                                  │
│               (vLLM platform plugin for Apple Silicon)                  │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
        ┌─────────────┬────────────┴────────────┬─────────────┐
        ▼             ▼                         ▼             ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│    mlx-lm     │ │   mlx-vlm     │ │   mlx-audio   │ │mlx-embeddings │
│(LLM inference)│ │ (Vision+LLM)  │ │  (TTS + STT)  │ │ (Embeddings)  │
└───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘
        │             │                         │             │
        └─────────────┴─────────────────────────┴─────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              MLX                                        │
│                (Apple ML Framework - Metal kernels)                      │
└─────────────────────────────────────────────────────────────────────────┘
```

## Performance

**LLM Performance (M4 Max, 128GB):**

| Model | Speed | Memory |
|-------|-------|--------|
| Qwen3-0.6B-8bit | 402 tok/s | 0.7 GB |
| Llama-3.2-1B-4bit | 464 tok/s | 0.7 GB |
| Llama-3.2-3B-4bit | 200 tok/s | 1.8 GB |

**Continuous Batching (5 concurrent requests):**

| Model | Single | Batched | Speedup |
|-------|--------|---------|---------|
| Qwen3-0.6B-8bit | 328 tok/s | 1112 tok/s | **3.4x** |
| Llama-3.2-1B-4bit | 299 tok/s | 613 tok/s | **2.0x** |

**Audio - Speech-to-Text (M4 Max, 128GB):**

| Model | RTF* | Use Case |
|-------|------|----------|
| whisper-tiny | **197x** | Real-time, low latency |
| whisper-large-v3-turbo | **55x** | Best quality/speed balance |
| whisper-large-v3 | **24x** | Highest accuracy |

*RTF = Real-Time Factor. RTF of 100x means 1 minute transcribes in ~0.6 seconds.

See [benchmarks](docs/benchmarks/) for detailed results.

## Gemma 3 Support

vllm-mlx includes native support for Gemma 3 vision models. Gemma 3 is automatically detected as MLLM.

### Usage

```bash
# Start server with Gemma 3
vllm-mlx serve mlx-community/gemma-3-27b-it-4bit --port 8000

# Verify it loaded as MLLM (not LLM)
curl http://localhost:8000/health
# Should show: "model_type": "mllm"
```

### Long Context Patch (mlx-vlm)

Gemma 3's default `sliding_window=1024` limits context to ~10K tokens on Apple Silicon (Metal GPU timeout at higher context). To enable longer context (up to ~50K tokens), patch mlx-vlm:

**Location:** `~/.../site-packages/mlx_vlm/models/gemma3/language.py`

Find the `make_cache` method and replace with:

```python
def make_cache(self):
    import os
    # Set GEMMA3_SLIDING_WINDOW=8192 for ~40K context
    # Set GEMMA3_SLIDING_WINDOW=0 for ~50K context (full KVCache)
    sliding_window = int(os.environ.get('GEMMA3_SLIDING_WINDOW', self.config.sliding_window))

    caches = []
    for i in range(self.config.num_hidden_layers):
        if (
            i % self.config.sliding_window_pattern
            == self.config.sliding_window_pattern - 1
        ):
            caches.append(KVCache())
        elif sliding_window == 0:
            caches.append(KVCache())  # Full context for all layers
        else:
            caches.append(RotatingKVCache(max_size=sliding_window, keep=0))
    return caches
```

**Usage:**

```bash
# Default (~10K max context)
vllm-mlx serve mlx-community/gemma-3-27b-it-4bit --port 8000

# Extended context (~40K max)
GEMMA3_SLIDING_WINDOW=8192 vllm-mlx serve mlx-community/gemma-3-27b-it-4bit --port 8000

# Maximum context (~50K max)
GEMMA3_SLIDING_WINDOW=0 vllm-mlx serve mlx-community/gemma-3-27b-it-4bit --port 8000
```

**Benchmark Results (M4 Max 128GB):**

| Setting | Max Context | Memory |
|---------|-------------|--------|
| Default (1024) | ~10K tokens | ~16GB |
| `GEMMA3_SLIDING_WINDOW=8192` | ~40K tokens | ~25GB |
| `GEMMA3_SLIDING_WINDOW=0` | ~50K tokens | ~35GB |

## Contributing

We welcome contributions! See [Contributing Guide](docs/development/contributing.md) for details.

- Bug fixes and improvements
- Performance optimizations
- Documentation improvements
- Benchmarks on different Apple Silicon chips

Submit PRs to: [https://github.com/waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx)

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.

## Benchmarks — MiniMax-M2.5 on M3 Ultra

All benchmarks on **Mac Studio M3 Ultra (256GB)** running **MiniMax-M2.5-MLX-4bit (229B MoE, ~120GB)** with `--reasoning-parser deepseek_r1 --tool-call-parser minimax --enable-auto-tool-choice --enable-prefix-cache`.

Server mode: Simple (no continuous batching)

### Tier 0 vs Tier 1 Comparison

#### TTFT (Time To First Token)

| Prompt Size | Tier 0 | Tier 1 | Delta |
|-------------|--------|--------|-------|
| Short (~50 tok) | 0.376s | 0.374s | -0.5% |
| Medium (~500 tok) | 0.818s | 0.793s | **-3.1%** |
| Long (~2K tok) | 1.422s | 1.424s | ~same |

#### Decode Throughput

| Output Length | Tier 0 | Tier 1 | Delta |
|--------------|--------|--------|-------|
| 128 tokens | 53.2 tok/s | 53.1 tok/s | ~same |
| 512 tokens | 52.3 tok/s | 52.1 tok/s | ~same |
| 2048 tokens | 49.8 tok/s | 49.8 tok/s | same |

No regression — frequency-aware eviction adds negligible overhead.

#### Prefix Cache (Multi-Turn Conversation)

| Turn | Tier 0 | Tier 1 | Delta |
|------|--------|--------|-------|
| Turn 1 | 0.526s | 0.524s | ~same |
| Turn 2 | 0.775s | 0.608s | **-21.5%** |
| Turn 3 | 0.987s | 0.905s | **-8.3%** |
| Turn 4 | 1.120s | 1.062s | **-5.2%** |
| **T4/T1 ratio** | **2.09x** | **2.03x** | **improved** |

Frequency-aware LRU-LFU hybrid keeps high-frequency blocks (system prompt, early turns) cached longer. Turn 2 sees the largest improvement because the system prompt blocks now survive eviction pressure from Turn 1's generation.

#### Tool Calling

| Test | Tier 0 | Tier 1 | Tools Called |
|------|--------|--------|-------------|
| Simple (weather) | OK, 2.00s | OK, 2.08s | get_weather |
| Multi-arg (search) | OK, 2.51s | OK, 2.51s | search_web |
| Code execution | OK, 3.99s | OK, 4.03s | run_python |
| Multi-tool selection | OK, 3.04s | OK, 2.99s | get_weather, search_web |

Accuracy: **4/4 (100%)** on both tiers | Avg latency: **2.89s → 2.90s** (no regression)

The streaming tool truncation fix is a correctness improvement — it prevents silent data loss when streams are interrupted mid-markup (e.g., `<minimax:tool_call>` without closing tag). This doesn't appear in normal benchmarks but eliminates a class of silent failures in production.

#### Reasoning Separation (deepseek_r1 parser)

| Test | Tier 0 | Tier 1 | Status |
|------|--------|--------|--------|
| Math (17*23) | 868 / 222 chars | 1008 / 210 chars | OK |
| Logic | 2338 / 0 chars | 2367 / 0 chars | PARTIAL |
| Code | 1960 / 0 chars | 1989 / 0 chars | PARTIAL |

Properly separated: **1/3** on both tiers (reasoning parser unchanged — improvement planned for later tier)

#### Long Generation Stability

| Metric | Tier 0 | Tier 1 |
|--------|--------|--------|
| Max tokens | 8192 | 8192 |
| Completed | Yes | Yes |
| Decode speed | 32.6 tok/s | 31.9 tok/s |
| Total time | 251.7s | 257.2s |
| Output chars | 36,609 | 27,944 |

### Key Takeaways

1. **Prefix cache improved** — Turn 2 TTFT dropped **21.5%**, T4/T1 ratio improved from 2.09x to 2.03x
2. **Zero regressions** — TTFT, decode throughput, tool accuracy, and stability all unchanged
3. **Tool truncation fix is latent** — prevents silent data loss on interrupted streams (correctness, not perf)
4. **Bigger gains under pressure** — frequency-aware eviction benefits grow as cache fills up; this benchmark doesn't fully saturate the cache

### Run the Benchmark

```bash
# Start server
python -m vllm_mlx.cli serve <model> \
  --tool-call-parser minimax --enable-auto-tool-choice \
  --reasoning-parser deepseek_r1 --enable-prefix-cache

# Run all tests
python benchmark_minmax.py --output results.json

# Run specific test
python benchmark_minmax.py --test ttft
python benchmark_minmax.py --test tool_call
```

---

## Optimization Roadmap

Prioritized optimization plan based on analysis of upstream vLLM features, adapted for MLX on Apple Silicon. Focused on large-model serving (MiniMax-M2.5 229B MoE, ~120GB on M3 Ultra 256GB).

See [docs/plans/](docs/plans/) for detailed implementation plans.

### Tier 0: Quick Wins (Merged)

Low-risk, high-impact changes. [Full plan](docs/plans/tier0-pinned-prefix-cache.md).

| Optimization | Impact | Status |
|-------------|--------|--------|
| **GC Control** -- disable Python GC during generation | Eliminates latency spikes | Merged |
| **Detokenizer Caching** -- incremental token counting | O(n^2) -> O(1) per chunk | Merged |
| **Schema Error Hardening** -- graceful fallback on bad schemas | Prevents server crashes | Merged |
| **Pinned Prefix Cache** -- prevent system prompt eviction | Turn4/Turn1 TTFT 2.09x -> ~1.0x | Merged |

### Tier 1: High Impact Optimizations (Partially Merged)

| Optimization | Expected Impact | Status |
|-------------|----------------|--------|
| **Fix Streaming Tool Truncation** -- parser-aware fallback for incomplete tool calls | Eliminates silently lost MiniMax/Llama/Mistral tool calls | Merged |
| **Frequency-Aware Cache Eviction** -- LRU-LFU hybrid for multi-turn | System prompt blocks survive 100x longer under pressure | Merged |
| **KV Block Reuse Ordering** -- prefer high-frequency blocks on hash collision | Higher cache hit ratio for hybrid models | Merged |
| **Jump-Forward Decoding** -- skip constrained tokens in guided gen | 2-5x faster JSON generation | Planned |
| **Structural Tags for Tool Calling** -- guarantee valid JSON args | <1% tool call failure rate | Planned |
| **LogitsProcessor Interface** -- clean extensibility for custom logits | Plugin architecture for constrained decoding | Planned |

### Tier 2: Evaluate Later (High Potential, High Complexity)

| Optimization | Expected Impact | Upstream Ref |
|-------------|----------------|-------------|
| **Beam Search + Structured Output** -- best-quality tool arguments | Highest quality constrained generation | [vLLM #35022](https://github.com/vllm-project/vllm/pull/35022) |
| **Draft Model Speculative Decoding** -- faster generation overall | 1.5-2x decode speedup | [vLLM #32662](https://github.com/vllm-project/vllm/pull/32662) |
| **Inter-Prefill-Budget** -- better TTFT under concurrent load | 37% TTFT reduction for concurrent requests | [vLLM #33743](https://github.com/vllm-project/vllm/pull/33743) |
| **Tool Calling Redesign** -- better architecture for tool call parsing | Cleaner, more maintainable tool call path | [vLLM #22977](https://github.com/vllm-project/vllm/pull/22977) |

### Estimated Combined Impact (Tier 0 + Tier 1)

For 10-turn agentic conversations with tool calling (e.g., OpenClaw on MiniMax-M2.5):

- **TTFT**: 2x improvement (pinned prefix cache + GC control)
- **Tool call reliability**: 95% -> 99%+ (structural tags + streaming fix)
- **Throughput consistency**: Eliminate GC-induced spikes
- **Long generation**: O(n^2) -> O(1) per-chunk overhead

---

## Citation

If you use vLLM-MLX in your research or project, please cite:

```bibtex
@software{vllm_mlx2025,
  author = {Barrios, Wayner},
  title = {vLLM-MLX: Apple Silicon MLX Backend for vLLM},
  year = {2025},
  url = {https://github.com/waybarrios/vllm-mlx},
  note = {Native GPU-accelerated LLM and vision-language model inference on Apple Silicon}
}
```

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
- [mlx-lm](https://github.com/ml-explore/mlx-lm) - LLM inference library
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) - Vision-language models
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) - Text-to-Speech and Speech-to-Text
- [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) - Text embeddings
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput LLM serving
